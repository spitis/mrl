import mrl
import gym
from mrl.replays.core.shared_buffer import SharedMemoryTrajectoryBuffer as Buffer
import numpy as np
import pickle
import os
from mrl.utils.misc import flatten_state

class OnlineHERBuffer(mrl.Module):

  def __init__(
      self,
      module_name='replay_buffer'
    ):
    """
    Buffer that does online hindsight relabeling.
    """

    super().__init__(module_name, required_agent_modules=['env'], locals=locals())

    self.size = None
    self.goal_shape = None
    self.buffer = None
    self.save_buffer = None # can be manually set to save this replay buffer irrespective of config
    self.modalities = ['observation']
    self.goal_modalities = ['desired_goal']
    
  def _setup(self):
    self.size = self.config.replay_size

    env = self.env

    if type(env.observation_space) == gym.spaces.Dict:
      if env.goal_env:
        self.goal_modalities = [m for m in self.config.goal_modalities]
        self.goal_shape = (env.goal_dim,)
      state_shape = (env.state_dim,)
      self.modalities = [m for m in self.config.modalities]
    else:
      state_shape = env.observation_space.shape

    items = [("state", state_shape),
             ("action", env.action_space.shape), ("reward", (1,)),
             ("next_state", state_shape), ("done", (1,))]

    if self.goal_shape is not None:
      items += [("previous_ag", self.goal_shape), # for reward shaping
                ("ag", self.goal_shape), # achieved goal
                ("bg", self.goal_shape), # behavioral goal (i.e., intrinsic if curious agent)
                ("dg", self.goal_shape)] # desired goal (even if ignored behaviorally)

    self.buffer = Buffer(self.size, items)
    self._subbuffers = [[] for _ in range(self.env.num_envs)]
    self.n_envs = self.env.num_envs

    # HER mode can differ if demo or normal replay buffer
    if 'demo' in self.module_name:
      self.fut, self.act, self.ach, self.beh = parse_hindsight_mode(self.config.demo_her)
    else:
      self.fut, self.act, self.ach, self.beh = parse_hindsight_mode(self.config.her)

  def _process_experience(self, exp):
    if getattr(self, 'logger'):
      self.logger.add_tabular('Replay buffer size', len(self.buffer))
    done = np.expand_dims(exp.done, 1)  # format for replay buffer
    reward = np.expand_dims(exp.reward, 1)  # format for replay buffer

    action = exp.action

    if self.goal_shape:
      state = flatten_state(exp.state, self.modalities)
      next_state = flatten_state(exp.next_state, self.modalities)
      if hasattr(self, 'achieved_goal'):
        previous_achieved = self.achieved_goal(exp.state)
        achieved = self.achieved_goal(exp.next_state)
      else:
        previous_achieved = exp.state['achieved_goal']
        achieved = exp.next_state['achieved_goal']
      desired = flatten_state(exp.state, self.goal_modalities)
      if hasattr(self, 'ag_curiosity') and self.ag_curiosity.current_goals is not None:
        behavioral = self.ag_curiosity.current_goals
        # recompute online reward
        reward = self.env.compute_reward(achieved, behavioral, {'s':state, 'a':action, 'ns':next_state}).reshape(-1, 1)
      else:
        behavioral = desired
      for i in range(self.n_envs):
        self._subbuffers[i].append([
            state[i], action[i], reward[i], next_state[i], done[i], previous_achieved[i], achieved[i],
            behavioral[i], desired[i]
        ])
    else:
      state = exp.state
      next_state = exp.next_state
      for i in range(self.n_envs):
        self._subbuffers[i].append(
            [state[i], action[i], reward[i], next_state[i], done[i]])

    for i in range(self.n_envs):
      if exp.trajectory_over[i]:
        trajectory = [np.stack(a) for a in zip(*self._subbuffers[i])]
        self.buffer.add_trajectory(*trajectory)
        self._subbuffers[i] = []

  def sample(self, batch_size, to_torch=True):
    if hasattr(self, 'prioritized_replay'):
      batch_idxs = self.prioritized_replay(batch_size)
    else:
      batch_idxs = np.random.randint(self.buffer.size, size=batch_size)

    if self.goal_shape:
      if "demo" in self.module_name:
        has_config_her = self.config.get('demo_her')
      else:
        has_config_her = self.config.get('her')
      
      if has_config_her:

        if self.config.env_steps > self.config.future_warm_up:
          fut_batch_size, act_batch_size, ach_batch_size, beh_batch_size, real_batch_size = np.random.multinomial(
              batch_size, [self.fut, self.act, self.ach, self.beh, 1.])
        else:
          fut_batch_size, act_batch_size, ach_batch_size, beh_batch_size, real_batch_size = batch_size, 0, 0, 0, 0

        fut_idxs, act_idxs, ach_idxs, beh_idxs, real_idxs = np.array_split(batch_idxs, 
          np.cumsum([fut_batch_size, act_batch_size, ach_batch_size, beh_batch_size]))

        # Sample the real batch (i.e., goals = behavioral goals)
        states, actions, rewards, next_states, dones, previous_ags, ags, goals, _ =\
            self.buffer.sample(real_batch_size, batch_idxs=real_idxs)

        # Sample the future batch
        states_fut, actions_fut, _, next_states_fut, dones_fut, previous_ags_fut, ags_fut, _, _, goals_fut =\
          self.buffer.sample_future(fut_batch_size, batch_idxs=fut_idxs)

        # Sample the actual batch
        states_act, actions_act, _, next_states_act, dones_act, previous_ags_act, ags_act, _, _, goals_act =\
          self.buffer.sample_from_goal_buffer('dg', act_batch_size, batch_idxs=act_idxs)

        # Sample the achieved batch
        states_ach, actions_ach, _, next_states_ach, dones_ach, previous_ags_ach, ags_ach, _, _, goals_ach =\
          self.buffer.sample_from_goal_buffer('ag', ach_batch_size, batch_idxs=ach_idxs)

        # Sample the behavioral batch
        states_beh, actions_beh, _, next_states_beh, dones_beh, previous_ags_beh, ags_beh, _, _, goals_beh =\
          self.buffer.sample_from_goal_buffer('bg', beh_batch_size, batch_idxs=beh_idxs)

        # Concatenate the five
        states = np.concatenate([states, states_fut, states_act, states_ach, states_beh], 0)
        actions = np.concatenate([actions, actions_fut, actions_act, actions_ach, actions_beh], 0)
        ags = np.concatenate([ags, ags_fut, ags_act, ags_ach, ags_beh], 0)
        goals = np.concatenate([goals, goals_fut, goals_act, goals_ach, goals_beh], 0)
        next_states = np.concatenate([next_states, next_states_fut, next_states_act, next_states_ach, next_states_beh], 0)

        # Recompute reward online
        if hasattr(self, 'goal_reward'):
          rewards = self.goal_reward(ags, goals, {'s':states, 'a':actions, 'ns':next_states}).reshape(-1, 1).astype(np.float32)
        else:
          rewards = self.env.compute_reward(ags, goals, {'s':states, 'a':actions, 'ns':next_states}).reshape(-1, 1).astype(np.float32)

        if self.config.get('never_done'):
          dones = np.zeros_like(rewards, dtype=np.float32)
        elif self.config.get('first_visit_succ'):
          dones = np.round(rewards + 1.)
        else:
          raise ValueError("Never done or first visit succ must be set in goal environments to use HER.")
          dones = np.concatenate([dones, dones_fut, dones_act, dones_ach, dones_beh], 0)

        if self.config.sparse_reward_shaping:
          previous_ags = np.concatenate([previous_ags, previous_ags_fut, previous_ags_act, previous_ags_ach, previous_ags_beh], 0)
          previous_phi = -np.linalg.norm(previous_ags - goals, axis=1, keepdims=True)
          current_phi  = -np.linalg.norm(ags - goals, axis=1, keepdims=True)
          rewards_F = self.config.gamma * current_phi - previous_phi
          rewards += self.config.sparse_reward_shaping * rewards_F

      else:
        # Uses the original desired goals
        states, actions, rewards, next_states, dones, _ , _, _, goals =\
                                                    self.buffer.sample(batch_size, batch_idxs=batch_idxs)

      if self.config.slot_based_state:
        # TODO: For now, we flatten according to config.slot_state_dims
        I, J = self.config.slot_state_dims
        states = np.concatenate((states[:, I, J], goals), -1)
        next_states = np.concatenate((next_states[:, I, J], goals), -1)
      else:
        states = np.concatenate((states, goals), -1)
        next_states = np.concatenate((next_states, goals), -1)
      gammas = self.config.gamma * (1.-dones)

    elif self.config.get('n_step_returns') and self.config.n_step_returns > 1:
      states, actions, rewards, next_states, dones = self.buffer.sample_n_step_transitions(
        batch_size, self.config.n_step_returns, self.config.gamma, batch_idxs=batch_idxs
      )
      gammas = self.config.gamma**self.config.n_step_returns * (1.-dones)

    else:
      states, actions, rewards, next_states, dones = self.buffer.sample(
          batch_size, batch_idxs=batch_idxs)
      gammas = self.config.gamma * (1.-dones)

    if hasattr(self, 'state_normalizer'):
      states = self.state_normalizer(states, update=False).astype(np.float32)
      next_states = self.state_normalizer(
          next_states, update=False).astype(np.float32)
    
    if to_torch:
      return (self.torch(states), self.torch(actions),
            self.torch(rewards), self.torch(next_states),
            self.torch(gammas))
    else:
      return (states, actions, rewards, next_states, gammas)

  def __len__(self):
    return len(self.buffer)

  def save(self, save_folder):
    if self.config.save_replay_buf or self.save_buffer:
      state = self.buffer._get_state()
      with open(os.path.join(save_folder, "{}.pickle".format(self.module_name)), 'wb') as f:
        pickle.dump(state, f)

  def load(self, save_folder):
    load_path = os.path.join(save_folder, "{}.pickle".format(self.module_name))
    if os.path.exists(load_path):
      with open(load_path, 'rb') as f:
        state = pickle.load(f)
      self.buffer._set_state(state)
    else:
      self.logger.log_color('WARNING', 'Replay buffer is not being loaded / was not saved.', color='cyan')
      self.logger.log_color('WARNING', 'Replay buffer is not being loaded / was not saved.', color='red')
      self.logger.log_color('WARNING', 'Replay buffer is not being loaded / was not saved.', color='yellow')

def parse_hindsight_mode(hindsight_mode : str):
  if 'future_' in hindsight_mode:
    _, fut = hindsight_mode.split('_')
    fut = float(fut) / (1. + float(fut))
    act = 0.
    ach = 0.
    beh = 0.
  elif 'futureactual_' in hindsight_mode:
    _, fut, act = hindsight_mode.split('_')
    non_hindsight_frac = 1. / (1. + float(fut) + float(act))
    fut = float(fut) * non_hindsight_frac
    act = float(act) * non_hindsight_frac
    ach = 0.
    beh = 0.
  elif 'futureachieved_' in hindsight_mode:
    _, fut, ach = hindsight_mode.split('_')
    non_hindsight_frac = 1. / (1. + float(fut) + float(ach))
    fut = float(fut) * non_hindsight_frac
    act = 0.
    ach = float(ach) * non_hindsight_frac
    beh = 0.
  elif 'rfaa_' in hindsight_mode:
    _, real, fut, act, ach = hindsight_mode.split('_')
    denom = (float(real) + float(fut) + float(act) + float(ach))
    fut = float(fut) / denom
    act = float(act) / denom
    ach = float(ach) / denom
    beh = 0.
  elif 'rfaab_' in hindsight_mode:
    _, real, fut, act, ach, beh = hindsight_mode.split('_')
    denom = (float(real) + float(fut) + float(act) + float(ach) + float(beh))
    fut = float(fut) / denom
    act = float(act) / denom
    ach = float(ach) / denom
    beh = float(beh) / denom
  else:
    fut = 0.
    act = 0.
    ach = 0.
    beh = 0.

  return fut, act, ach, beh
