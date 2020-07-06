import mrl
import numpy as np
import torch
import gym
import os
import pickle
from experiments.coda.coda_generic import enlarge_dataset_spriteworld, enlarge_dataset_generic
from mrl.replays.online_her_buffer import OnlineHERBuffer
from mrl.replays.old_replay_buffer import OldReplayBuffer
from mrl.replays.core.replay_buffer import ReplayBuffer as Buffer
from mrl.utils.misc import batch_block_diag, batch_block_diag_many

class CodaBuffer(OnlineHERBuffer):
  """
  This is a special CODA version of the HER buffer
  """
  def __init__(self,
               module_name='replay_buffer',
               max_coda_ratio=0.5,
               make_coda_data_every=1000,
               num_coda_source_transitions=5000,
               num_coda_source_pairs=2000,
               coda_samples_per_pair=5,
               coda_buffer_size=None,
               coda_on_goal_components=False,
               add_bad_dcs = False,
               spriteworld_config=None,
               reward_fn=None, # used for non-Spriteworld
               num_procs=8):
    """
    Args:
      max_coda_ratio: maximum ratio of coda:real data when sampling from the buffer
      make_coda_data_every: how often to make coda data
      num_coda_source_transitions: how many random pairs from the main buffer we should use to make coda samples
      num_coda_source_pairs: using sampled transitions, how many source pairs should we form for doing CODA on?
      coda_samples_per_pair: with each source pair, how many coda samples should we make? # NOTE: these are deduplicated, so it's upper bound
      spriteworld_config: config for the spriteworld env
      num_procs: num cpus to use for generating coda sample
    """
    super().__init__()
    self.max_coda_ratio = max_coda_ratio
    self.make_coda_data_every = make_coda_data_every
    self.num_coda_source_transitions = num_coda_source_transitions
    self.num_coda_source_pairs = num_coda_source_pairs
    self.coda_samples_per_pair = coda_samples_per_pair
    self.coda_buffer = None
    self.coda_buffer_size = coda_buffer_size
    self.add_bad_dcs = add_bad_dcs
    self.num_procs = num_procs
    self.reward_fn=reward_fn
    self.coda_on_goal_components = coda_on_goal_components

    self.spriteworld_config = spriteworld_config # NONE == This is not Spriteworld
    if spriteworld_config is not None:
      raise NotImplementedError('not supported by public CoDA Code')
      assert self.reward_fn is None
      self.state_to_sprites = SpriteMaker()

  def _setup(self):
    """
    Keeps an separate, internal coda_buffer with coda experiences
    """

    super()._setup()

    env = self.env
    if type(env.observation_space) == gym.spaces.Dict:
      observation_space = env.observation_space.spaces["observation"]
    else:
      observation_space = env.observation_space

    items = [("state", observation_space.shape),
             ("action", env.action_space.shape), ("reward", (1,)),
             ("next_state", observation_space.shape), ("done", (1,))]

    self.coda_buffer = Buffer(self.coda_buffer_size or self.size, items)

    assert self.config.never_done # TODO: Make this work in general case
    
    if not self.config.slot_based_state and self.config.slot_state_dims:
      self.invert_batch_block_diag_mask = ([], [])
      for i, state_dims in enumerate(self.config.slot_state_dims):
        for state_dim in state_dims:
          self.invert_batch_block_diag_mask[0].append(i)
          self.invert_batch_block_diag_mask[1].append(state_dim)

    if self.config.slot_action_dims:
      self.invert_batch_block_diag_mask_actions = ([], [])
      for i, action_dims in enumerate(self.config.slot_action_dims):
        for action_dim in action_dims:
          self.invert_batch_block_diag_mask_actions[0].append(i)
          self.invert_batch_block_diag_mask_actions[1].append(action_dim)


  def _optimize(self):
    """
    Populates the internal coda_buffer using samples from the real buffer. 
    """
    config = self.config
    if not (len(self) and config.opt_steps % self.make_coda_data_every == 0):
      return

    # return if too early
    if hasattr(self, 'coda_attention_model') and len(self) < config.min_experience_to_train_coda_attn + 5000:
      return

    if self.spriteworld_config is not None:
      return self._optimize_spriteworld()

    sample = self.buffer.sample(self.num_coda_source_transitions)
    states, actions, next_states = sample[0], sample[1], sample[3]

    #og_state_set = set([tuple(s.flatten().round(2)) for s in states])

    # Convert states to the correct format
    if not config.slot_based_state and config.slot_state_dims:
      state_slots = [states[:, s][:,None] for s in config.slot_state_dims]
      states = batch_block_diag_many(*state_slots)
      next_state_slots = [next_states[:, s][:,None] for s in config.slot_state_dims]
      next_states = batch_block_diag_many(*next_state_slots)

    if self.config.slot_action_dims:
      action_slots = [actions[:, s][:,None] for s in config.slot_action_dims]
      actions = batch_block_diag_many(*action_slots)

    new_s1s, new_a1s, new_s2s = enlarge_dataset_generic(
        states, actions, next_states,
        self.num_coda_source_pairs,
        self.coda_samples_per_pair,
        batch_get_mask=self.get_coda_mask,
        pool=True,
        max_cpus=self.num_procs,
        add_bad_dcs=self.add_bad_dcs)

    if not len(new_s1s):
      return

    if not config.slot_based_state and config.slot_state_dims:
      new_s1s = new_s1s[:,self.invert_batch_block_diag_mask[0], self.invert_batch_block_diag_mask[1]]
      new_s2s = new_s2s[:,self.invert_batch_block_diag_mask[0], self.invert_batch_block_diag_mask[1]]

    if not config.slot_based_state and config.slot_action_dims:
      new_a1s = new_a1s[:,self.invert_batch_block_diag_mask_actions[0], self.invert_batch_block_diag_mask_actions[1]]

    #remove duplicates of original data
    #valid_idxs = []
    #for i, s in enumerate(new_s1s):
    # if not (tuple(s.flatten().round(2)) in og_state_set):
    #   valid_idxs.append(i)
    #
    #new_s1s = new_s1s[valid_idxs]
    #new_a1s = new_a1s[valid_idxs]
    #new_s2s = new_s2s[valid_idxs]

    assert config.never_done
    r = np.ones((len(new_s1s), 1))
    d = np.zeros((len(new_s1s), 1))

    self.coda_buffer.add_batch(new_s1s, new_a1s, r, new_s2s, d)

  def _optimize_spriteworld(self):
    """
    Populates the internal coda_buffer using samples from the real buffer. 
    """
    states, actions, rewards, next_states, _ = super().sample(self.num_coda_source_transitions, to_torch=False)

    sprites_for_base_data = []  # TODO this is spriteworld specific
    og_state_set = []
    sars_data = []

    for s, a, r, s2 in zip(states, actions, rewards, next_states):
      sprites_for_base_data.append(self.state_to_sprites(s))  # TODO this is spriteworld specific
      og_state_set.append(tuple(s.round(2)))
      sars_data.append((s, (a + 1.) / 2., r, s2)) # TODO the action part is spriteworld specific

    og_state_set = set(og_state_set)

    coda_data = enlarge_dataset_spriteworld(
        sars_data,
        sprites_for_base_data,
        self.spriteworld_config,  # TODO this is spriteworld specific
        self.num_coda_source_pairs,
        self.coda_samples_per_pair,
        flattened=True,
        custom_get_mask=self.get_coda_mask,
        pool=True,
        max_cpus=self.num_procs)

    # remove duplicates of original data
     # TODO the action part in next line is spriteworld specific
    coda_data = [(s, (a * 2) - 1., r, s2) for s, a, r, s2 in coda_data if not tuple(s.round(2)) in og_state_set]
    s, a, r, s2 = [np.array(x) for x in zip(*coda_data)]
    r = r.reshape(-1, 1)

    assert self.config.never_done
    d = np.zeros_like(r)
    
    assert not self.goal_space # Don't use goal space with sprite world. 
    self.coda_buffer.add_batch(s, a, r, s2, d)

  def sample(self, batch_size, to_torch=True):
    """
    Samples both from real buffer and from coda buffer and concatenates results.
    """
    assert (to_torch)

    coda_ratio = min(len(self.coda_buffer) / (len(self) + len(self.coda_buffer)), self.max_coda_ratio)

    if coda_ratio < 0.01:
      return super().sample(batch_size)

    coda_batch_size, real_batch_size = np.random.multinomial(batch_size, [coda_ratio, 1. - coda_ratio])

    coda_states, coda_actions, coda_rewards, coda_next_states, coda_gammas = self.sample_coda(coda_batch_size)
    states, actions, rewards, next_states, gammas = super().sample(real_batch_size)


    states = torch.cat((states, coda_states), 0)
    actions = torch.cat((actions, coda_actions), 0)
    rewards = torch.cat((rewards, coda_rewards), 0)
    next_states = torch.cat((next_states, coda_next_states), 0)
    gammas = torch.cat((gammas, coda_gammas), 0)

    return states, actions, rewards, next_states, gammas

  def sample_coda(self, batch_size, to_torch=True):
    """
    Samples from coda_buffer (no real data).
    """
    states, actions, rewards, next_states, dones = self.coda_buffer.sample(batch_size)

    assert np.all(np.logical_not(dones))
    
    # Reward relabeling
    if self.goal_space is not None:
      assert self.reward_fn is not None

      dg_buffer = self.buffer.BUFF.buffer_dg
      dg_idxs = np.random.randint(len(dg_buffer), size=(batch_size // 2,))
      dgs = dg_buffer.get_batch(dg_idxs)

      ag_buffer = self.buffer.BUFF.buffer_ag
      ag_idxs = np.random.randint(len(ag_buffer), size=(batch_size - batch_size // 2,))
      ags = ag_buffer.get_batch(ag_idxs)

      goals = np.concatenate((dgs, ags), axis=0)

      if self.coda_on_goal_components:
        goal_components = [goals[:, i] for i in self.config.slot_goal_dims]
        [np.random.shuffle(g) for g in goal_components]
        goals = np.concatenate(goal_components, axis=1)

      rewards = self.reward_fn(states, actions, next_states, goals)

      if self.config.slot_based_state:
        # TODO: For now, we flatten according to config.slot_state_dims
        I, J = self.config.slot_state_dims
        states = np.concatenate((states[:, I, J], goals), -1)
        next_states = np.concatenate((next_states[:, I, J], goals), -1)
      else:
        states = np.concatenate((states, goals), -1)
        next_states = np.concatenate((next_states, goals), -1)

    elif self.reward_fn is not None:
      rewards = self.reward_fn(states, actions, next_states)
      
    else:
      # This is spriteworld and things have already been relabeled
      assert self.spriteworld_config is not None

    if hasattr(self, 'done_fn'):
      dones = self.done_fn(states, actions, next_states)

    gammas = self.config.gamma * (1 - dones)
    
    if hasattr(self, 'state_normalizer'):
      states = self.state_normalizer(states, update=False).astype(np.float32)
      next_states = self.state_normalizer(next_states, update=False).astype(np.float32)

    if to_torch:
      return (self.torch(states), self.torch(actions), self.torch(rewards), self.torch(next_states), self.torch(gammas))
    else:
      return (states, actions, rewards, next_states, gammas)

  def save(self, save_folder):
    if self.config.save_replay_buf or self.save_buffer:
      state = self.buffer._get_state()
      with open(os.path.join(save_folder, "{}.pickle".format(self.module_name)), 'wb') as f:
        pickle.dump(state, f)

      state = self.coda_buffer._get_state()
      with open(os.path.join(save_folder, "{}.pickle".format(self.module_name + '_coda')), 'wb') as f:
        pickle.dump(state, f)

  def load(self, save_folder):
    load_path = os.path.join(save_folder, "{}.pickle".format(self.module_name))
    if os.path.exists(load_path):
      with open(load_path, 'rb') as f:
        state = pickle.load(f)
      self.buffer._set_state(state)
    else:
      self.logger.log_color('###############################################################', '', color='red')
      self.logger.log_color('WARNING', 'Replay buffer is not being loaded / was not saved.', color='cyan')
      self.logger.log_color('WARNING', 'Replay buffer is not being loaded / was not saved.', color='red')
      self.logger.log_color('WARNING', 'Replay buffer is not being loaded / was not saved.', color='yellow')
      self.logger.log_color('###############################################################', '', color='red')
    
    load_path = os.path.join(save_folder, "{}.pickle".format(self.module_name + '_coda'))
    if os.path.exists(load_path):
      with open(load_path, 'rb') as f:
        state = pickle.load(f)
      self.coda_buffer._set_state(state)
    else:
      self.logger.log_color('###############################################################', '', color='red')
      self.logger.log_color('WARNING', 'Coda buffer is not being loaded / was not saved.', color='cyan')
      self.logger.log_color('WARNING', 'Coda buffer is not being loaded / was not saved.', color='red')
      self.logger.log_color('WARNING', 'Coda buffer is not being loaded / was not saved.', color='yellow')
      self.logger.log_color('###############################################################', '', color='red')



class CodaOldBuffer(OldReplayBuffer):
  """
  This is a special CODA version of the HER buffer
  """
  def __init__(self,
               module_name='replay_buffer',
               max_coda_ratio=0.5,
               make_coda_data_every=1000,
               num_coda_source_transitions=5000,
               num_coda_source_pairs=2000,
               coda_samples_per_pair=5,
               coda_buffer_size=None,
               coda_on_goal_components=False,
               add_bad_dcs = False,
               spriteworld_config=None,
               reward_fn=None, # used for non-Spriteworld
               num_procs=8):
    """
    Args:
      max_coda_ratio: maximum ratio of coda:real data when sampling from the buffer
      make_coda_data_every: how often to make coda data
      num_coda_source_transitions: how many random pairs from the main buffer we should use to make coda samples
      num_coda_source_pairs: using sampled transitions, how many source pairs should we form for doing CODA on?
      coda_samples_per_pair: with each source pair, how many coda samples should we make? # NOTE: these are deduplicated, so it's upper bound
      spriteworld_config: config for the spriteworld env
      num_procs: num cpus to use for generating coda sample
    """
    super().__init__()
    self.max_coda_ratio = max_coda_ratio
    self.make_coda_data_every = make_coda_data_every
    self.num_coda_source_transitions = num_coda_source_transitions
    self.num_coda_source_pairs = num_coda_source_pairs
    self.coda_samples_per_pair = coda_samples_per_pair
    self.coda_buffer = None
    self.coda_buffer_size = coda_buffer_size
    self.add_bad_dcs = add_bad_dcs
    self.num_procs = num_procs
    self.reward_fn=reward_fn
    self.coda_on_goal_components = coda_on_goal_components

    self.spriteworld_config = spriteworld_config # NONE == This is not Spriteworld
    if spriteworld_config is not None:
      assert self.reward_fn is None
      self.state_to_sprites = SpriteMaker()

  def _setup(self):
    """
    Keeps an separate, internal coda_buffer with coda experiences
    """

    super()._setup()

    env = self.env
    if type(env.observation_space) == gym.spaces.Dict:
      observation_space = env.observation_space.spaces["observation"]
    else:
      observation_space = env.observation_space

    items = [("state", observation_space.shape),
             ("action", env.action_space.shape), ("reward", (1,)),
             ("next_state", observation_space.shape), ("done", (1,))]

    self.coda_buffer = Buffer(self.coda_buffer_size or self.size, items)

    assert self.config.never_done # TODO: Make this work in general case


    if not self.config.slot_based_state and self.config.slot_state_dims:
      self.invert_batch_block_diag_mask = ([], [])
      for i, state_dims in enumerate(self.config.slot_state_dims):
        for state_dim in state_dims:
          self.invert_batch_block_diag_mask[0].append(i)
          self.invert_batch_block_diag_mask[1].append(state_dim)

    if self.config.slot_action_dims:
      self.invert_batch_block_diag_mask_actions = ([], [])
      for i, action_dims in enumerate(self.config.slot_action_dims):
        for action_dim in action_dims:
          self.invert_batch_block_diag_mask_actions[0].append(i)
          self.invert_batch_block_diag_mask_actions[1].append(action_dim)

  def _optimize(self):
    """
    Populates the internal coda_buffer using samples from the real buffer. 
    """
    config = self.config
    if not (len(self) and config.opt_steps % self.make_coda_data_every == 0):
      return

    # return if too early
    if hasattr(self, 'coda_attention_model') and len(self) < config.min_experience_to_train_coda_attn + 5000:
      return

    if self.spriteworld_config is not None:
      return self._optimize_spriteworld()

    sample = self.buffer.sample(self.num_coda_source_transitions)
    states, actions, next_states = sample[0], sample[1], sample[3]

    #og_state_set = set([tuple(s.flatten().round(2)) for s in states])

    # Convert states to the correct format
    if not config.slot_based_state and config.slot_state_dims:
      state_slots = [states[:, s][:,None] for s in config.slot_state_dims]
      states = batch_block_diag_many(*state_slots)
      next_state_slots = [next_states[:, s][:,None] for s in config.slot_state_dims]
      next_states = batch_block_diag_many(*next_state_slots)

    if config.slot_action_dims:
      action_slots = [actions[:, s][:,None] for s in config.slot_action_dims]
      actions = batch_block_diag_many(*action_slots)

    new_s1s, new_a1s, new_s2s = enlarge_dataset_generic(
        states, actions, next_states,
        self.num_coda_source_pairs,
        self.coda_samples_per_pair,
        batch_get_mask=self.get_coda_mask,
        pool=True,
        max_cpus=self.num_procs,
        add_bad_dcs=self.add_bad_dcs)

    if not len(new_s1s):
      return

    if not config.slot_based_state and config.slot_state_dims:
      new_s1s = new_s1s[:,self.invert_batch_block_diag_mask[0], self.invert_batch_block_diag_mask[1]]
      new_s2s = new_s2s[:,self.invert_batch_block_diag_mask[0], self.invert_batch_block_diag_mask[1]]

    if not config.slot_based_state and config.slot_action_dims:
      new_a1s = new_a1s[:,self.invert_batch_block_diag_mask_actions[0], self.invert_batch_block_diag_mask_actions[1]]

    #remove duplicates of original data
    #valid_idxs = []
    #for i, s in enumerate(new_s1s):
    # if not (tuple(s.flatten().round(2)) in og_state_set):
    #   valid_idxs.append(i)
    #
    #new_s1s = new_s1s[valid_idxs]
    #new_a1s = new_a1s[valid_idxs]
    #new_s2s = new_s2s[valid_idxs]

    assert config.never_done
    r = np.ones((len(new_s1s), 1))
    d = np.zeros((len(new_s1s), 1))

    self.coda_buffer.add_batch(new_s1s, new_a1s, r, new_s2s, d)

  def _optimize_spriteworld(self):
    """
    Populates the internal coda_buffer using samples from the real buffer. 
    """
    states, actions, rewards, next_states, _ = super().sample(self.num_coda_source_transitions, to_torch=False)

    sprites_for_base_data = []  # TODO this is spriteworld specific
    og_state_set = []
    sars_data = []

    for s, a, r, s2 in zip(states, actions, rewards, next_states):
      sprites_for_base_data.append(self.state_to_sprites(s))  # TODO this is spriteworld specific
      og_state_set.append(tuple(s.round(2)))
      sars_data.append((s, (a + 1.) / 2., r, s2)) # TODO the action part is spriteworld specific

    og_state_set = set(og_state_set)

    coda_data = enlarge_dataset_spriteworld(
        sars_data,
        sprites_for_base_data,
        self.spriteworld_config,  # TODO this is spriteworld specific
        self.num_coda_source_pairs,
        self.coda_samples_per_pair,
        flattened=True,
        custom_get_mask=self.get_coda_mask,
        pool=True,
        max_cpus=self.num_procs)

    # remove duplicates of original data
     # TODO the action part in next line is spriteworld specific
    coda_data = [(s, (a * 2) - 1., r, s2) for s, a, r, s2 in coda_data if not tuple(s.round(2)) in og_state_set]
    s, a, r, s2 = [np.array(x) for x in zip(*coda_data)]
    r = r.reshape(-1, 1)

    assert self.config.never_done
    d = np.zeros_like(r)
    
    assert not self.goal_space # Don't use goal space with sprite world. 
    self.coda_buffer.add_batch(s, a, r, s2, d)

  def sample(self, batch_size, to_torch=True):
    """
    Samples both from real buffer and from coda buffer and concatenates results.
    """
    assert (to_torch)

    coda_ratio = min(len(self.coda_buffer) / (len(self) + len(self.coda_buffer)), self.max_coda_ratio)

    if coda_ratio < 0.01:
      return super().sample(batch_size)

    coda_batch_size, real_batch_size = np.random.multinomial(batch_size, [coda_ratio, 1. - coda_ratio])

    coda_states, coda_actions, coda_rewards, coda_next_states, coda_gammas = self.sample_coda(coda_batch_size)
    states, actions, rewards, next_states, gammas = super().sample(real_batch_size)


    states = torch.cat((states, coda_states), 0)
    actions = torch.cat((actions, coda_actions), 0)
    rewards = torch.cat((rewards, coda_rewards), 0)
    next_states = torch.cat((next_states, coda_next_states), 0)
    gammas = torch.cat((gammas, coda_gammas), 0)

    return states, actions, rewards, next_states, gammas

  def sample_coda(self, batch_size, to_torch=True):
    """
    Samples from coda_buffer (no real data).
    """
    states, actions, rewards, next_states, dones = self.coda_buffer.sample(batch_size)

    assert np.all(np.logical_not(dones))
    
    # Reward relabeling
    if self.goal_space is not None:
      assert self.reward_fn is not None

      dg_buffer = self.buffer.BUFF.buffer_dg
      dg_idxs = np.random.randint(len(dg_buffer), size=(batch_size // 2,))
      dgs = dg_buffer.get_batch(dg_idxs)

      ag_buffer = self.buffer.BUFF.buffer_ag
      ag_idxs = np.random.randint(len(ag_buffer), size=(batch_size - batch_size // 2,))
      ags = ag_buffer.get_batch(ag_idxs)

      goals = np.concatenate((dgs, ags), axis=0)

      if self.coda_on_goal_components:
        goal_components = [goals[:, i] for i in self.config.slot_goal_dims]
        [np.random.shuffle(g) for g in goal_components]
        goals = np.concatenate(goal_components, axis=1)

      rewards = self.reward_fn(states, actions, next_states, goals)

      if self.config.slot_based_state:
        # TODO: For now, we flatten according to config.slot_state_dims
        I, J = self.config.slot_state_dims
        states = np.concatenate((states[:, I, J], goals), -1)
        next_states = np.concatenate((next_states[:, I, J], goals), -1)
      else:
        states = np.concatenate((states, goals), -1)
        next_states = np.concatenate((next_states, goals), -1)

    elif self.reward_fn is not None:
      rewards = self.reward_fn(states, actions, next_states)
      
    else:
      # This is spriteworld and things have already been relabeled
      assert self.spriteworld_config is not None

    if hasattr(self, 'done_fn'):
      dones = self.done_fn(states, actions, next_states)

    gammas = self.config.gamma * (1 - dones)

    if hasattr(self, 'state_normalizer'):
      states = self.state_normalizer(states, update=False).astype(np.float32)
      next_states = self.state_normalizer(next_states, update=False).astype(np.float32)

    if to_torch:
      return (self.torch(states), self.torch(actions), self.torch(rewards), self.torch(next_states), self.torch(gammas))
    else:
      return (states, actions, rewards, next_states, gammas)

  def save(self, save_folder):
    if self.config.save_replay_buf or self.save_buffer:
      state = self.buffer._get_state()
      with open(os.path.join(save_folder, "{}.pickle".format(self.module_name)), 'wb') as f:
        pickle.dump(state, f)

      state = self.coda_buffer._get_state()
      with open(os.path.join(save_folder, "{}.pickle".format(self.module_name + '_coda')), 'wb') as f:
        pickle.dump(state, f)

  def load(self, save_folder):
    load_path = os.path.join(save_folder, "{}.pickle".format(self.module_name))
    if os.path.exists(load_path):
      with open(load_path, 'rb') as f:
        state = pickle.load(f)
      self.buffer._set_state(state)
    else:
      self.logger.log_color('###############################################################', '', color='red')
      self.logger.log_color('WARNING', 'Replay buffer is not being loaded / was not saved.', color='cyan')
      self.logger.log_color('WARNING', 'Replay buffer is not being loaded / was not saved.', color='red')
      self.logger.log_color('WARNING', 'Replay buffer is not being loaded / was not saved.', color='yellow')
      self.logger.log_color('###############################################################', '', color='red')
    
    load_path = os.path.join(save_folder, "{}.pickle".format(self.module_name + '_coda'))
    if os.path.exists(load_path):
      with open(load_path, 'rb') as f:
        state = pickle.load(f)
      self.coda_buffer._set_state(state)
    else:
      self.logger.log_color('###############################################################', '', color='red')
      self.logger.log_color('WARNING', 'Coda buffer is not being loaded / was not saved.', color='cyan')
      self.logger.log_color('WARNING', 'Coda buffer is not being loaded / was not saved.', color='red')
      self.logger.log_color('WARNING', 'Coda buffer is not being loaded / was not saved.', color='yellow')
      self.logger.log_color('###############################################################', '', color='red')