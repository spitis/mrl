import mrl
import numpy as np
import gym
from mrl.replays.core.replay_buffer import RingBuffer
from mrl.utils.misc import AttrDict
import pickle
import os
from sklearn import mixture
from scipy.stats import rankdata

class EntropyPrioritizedOnlineHERBuffer(mrl.Module):

  def __init__(
      self,
      module_name='prioritized_replay',
      rank_method='dense',
      temperature=1.0
  ):
    """
    Buffer that stores entropy of trajectories for prioritized replay
    """

    super().__init__(module_name, required_agent_modules=['env','replay_buffer'], locals=locals())

    self.goal_space = None
    self.buffer = None
    self.rank_method = rank_method
    self.temperature = temperature
    self.traj_len = None
    self.has_fit_density = False

  def _setup(self):
    self.ag_buffer = self.replay_buffer.buffer.BUFF.buffer_ag

    env = self.env
    assert type(env.observation_space) == gym.spaces.Dict
    self.goal_space = env.observation_space.spaces["desired_goal"]

    # Note: for now we apply entropy estimation on the achieved goal (ag) space
    # Define the buffers to store for prioritization
    items = [("entropy", (1,)), ("priority", (1,))]
    self.buffer = AttrDict()
    for name, shape in items:
      self.buffer['buffer_' + name] = RingBuffer(self.ag_buffer.maxlen, shape=shape)

    self._subbuffers = [[] for _ in range(self.env.num_envs)]
    self.n_envs = self.env.num_envs

    # Define the placeholder for mixture model to estimate trajectory
    self.clf = 0

  def fit_density_model(self):
    if not self.has_fit_density:
      self.has_fit_density = True
    ag = self.ag_buffer.data[0:self.size].copy()
    X_train = ag.reshape(-1, self.traj_len * ag.shape[-1]) # [num_episodes, episode_len * goal_dim]

    self.clf = mixture.BayesianGaussianMixture(weight_concentration_prior_type="dirichlet_distribution", n_components=3)
    self.clf.fit(X_train)
    pred = -self.clf.score_samples(X_train)

    self.pred_min = pred.min()
    pred = pred - self.pred_min
    pred = np.clip(pred, 0, None)
    self.pred_sum = pred.sum()
    pred = pred / self.pred_sum
    self.pred_avg = (1 / pred.shape[0])
    pred = np.repeat(pred, self.traj_len, axis=0)

    self.buffer.buffer_entropy.data[:self.size] = pred.reshape(-1,1).copy()

  def _process_experience(self, exp):
    # Compute the entropy 
    # TODO: Include previous achieved goal too? or use that instead of ag?
    achieved = exp.next_state['achieved_goal']
    for i in range(self.n_envs):
      self._subbuffers[i].append([achieved[i]])
    
    for i in range(self.n_envs):
      if exp.trajectory_over[i]:
        # TODO: Compute the entropy of the trajectory
        traj_len = len(self._subbuffers[i])
        if self.traj_len is None:
          self.traj_len = traj_len
        else:
          # Current implementation assumes the same length for all trajectories
          assert(traj_len == self.traj_len)

        if not isinstance(self.clf, int):
          ag = [np.stack(a) for a in zip(*self._subbuffers[i])][0] # [episode_len, goal_dim]
          X = ag.reshape(-1, ag.shape[0]*ag.shape[1])
          pred = -self.clf.score_samples(X)

          pred = pred - self.pred_min
          pred = np.clip(pred, 0, None)
          pred = pred / self.pred_sum # Shape (1,)

          entropy = np.ones((traj_len,1)) * pred
        else:
          # Not enough data to train mixture density yet, set entropy to be zero
          entropy = np.zeros((traj_len, 1))
        
        priority = np.zeros((traj_len,1))
        trajectory = [entropy, priority]
        
        # TODO: Update the trajectory with entropy
        self.add_trajectory(*trajectory)

        self._subbuffers[i] = []

        # TODO: Update the rank here before adding it to the trajectory?
        self.update_priority()

  def add_trajectory(self, *items):
    """
    Append a trajectory of transitions to the buffer.

    :param items: a list of batched transition values to append to the replay buffer,
        in the item order that we initialized the ReplayBuffer with.
    """
    for buffer, batched_values in zip(self.buffer.values(), items):
      buffer.append_batch(batched_values)

  def update_priority(self):
    """
    After adding a trajectory to the replay buffer, update the ranking of transitions
    """
    # Note: 'dense' assigns the next highest element with the rank immediately 
    # after those assigned to the tied elements.
    entropy_transition_total = self.buffer.buffer_entropy.data[:self.size]
    entropy_rank = rankdata(entropy_transition_total, method=self.rank_method)
    entropy_rank = (entropy_rank - 1).reshape(-1, 1)
    self.buffer.buffer_priority.data[:self.size] = entropy_rank

  def __call__(self, batch_size):
    """
    Samples batch_size number of indices from main replay_buffer.

    Args:
      batch_size (int): size of the batch to sample
    
    Returns:
      batch_idxs: a 1-D numpy array of length batch_size containing indices
                  sampled in prioritized manner
    """
    if self.rank_method == 'none':
      entropy_trajectory = self.buffer.buffer_entropy.data[:self.size]
    else:
      entropy_trajectory = self.buffer.buffer_priority.data[:self.size]
    
    # Factorize out sampling into sampling trajectory according to priority/entropy
    # then sample time uniformly independently
    entropy_trajectory = entropy_trajectory.reshape(-1, self.traj_len)[:,0]
    p_trajectory = np.power(entropy_trajectory, 1/(self.temperature+1e-2))
    
    # If the density model hasn't been fitted yet, we have p_trajectory all 0's
    # And hence treat them as uniform:
    if not self.has_fit_density:
      p_trajectory = np.ones(p_trajectory.shape) / len(p_trajectory)
    else:
      assert(p_trajectory.sum() != 0.0)
      p_trajectory = p_trajectory / p_trajectory.sum()
    
    num_trajectories = p_trajectory.shape[0]
    batch_tidx = np.random.choice(num_trajectories, size=batch_size, p=p_trajectory)
    batch_idxs = self.traj_len * batch_tidx + np.random.choice(self.traj_len, size=batch_size)

    return batch_idxs

  @property
  def size(self):
    return len(self.ag_buffer)

  def save(self, save_folder):
    if self.config.save_replay_buf:
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
      self.logger.log_color('###############################################################', '', color='red')
      self.logger.log_color('WARNING', 'Replay buffer is not being loaded / was not saved.', color='cyan')
      self.logger.log_color('WARNING', 'Replay buffer is not being loaded / was not saved.', color='red')
      self.logger.log_color('WARNING', 'Replay buffer is not being loaded / was not saved.', color='yellow')
      self.logger.log_color('###############################################################', '', color='red')