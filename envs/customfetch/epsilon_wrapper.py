import numpy as np
from gym import spaces

class EpsilonWrapper(object):

  def __init__(self, env, attrs=('distance_threshold', 'rotation_threshold'), compute_reward_with_internal=None):
    """Attrs is list of attributes (strings like "distance_threshold"). Only valid ones are used. """

    self.env = env
    if hasattr(self.env, 'mode'):
      assert self.env.mode == 0

    if compute_reward_with_internal is not None:
      self.compute_reward_with_internal = compute_reward_with_internal

    obs = self.env.reset()
    self.internal_len = obs['achieved_goal'].shape[0]

    self.attrs = []
    self.defaults = []

    for attr in attrs:
      if hasattr(self.env, attr):
        self.attrs.append(attr)
        self.defaults.append(getattr(self.env, attr))
    
    self.defaults = np.array(self.defaults)

    self.observation_space = spaces.Dict(dict(
        desired_goal=spaces.Box(-np.inf, np.inf, shape=(self.internal_len + len(self.attrs),), dtype='float32'),
        achieved_goal=spaces.Box(-np.inf, np.inf, shape=(self.internal_len + len(self.attrs),), dtype='float32'),
        observation=spaces.Box(-np.inf, np.inf, shape=(obs['observation'].shape[0],), dtype='float32'),
    ))


  def __getattr__(self, attr):
    if attr in self.__dict__:
      return getattr(self, attr)
    return getattr(self.env, attr)

  def compute_reward(self, achieved_goal, goal, info):
    internal_achieved = achieved_goal[:self.internal_len]
    internal_goal = goal[:self.internal_len]
    
    if self.compute_reward_with_internal:
      for attr, eps in zip(self.attrs, self.defaults):
        setattr(self.env, attr, eps)
      internal_reward = self.env.compute_reward(internal_achieved, internal_goal, info)
      return internal_reward

    # Otherwise, use epsilon in the goal to determine external reward
    for attr, eps in zip(self.attrs, goal[self.internal_len:]):
      setattr(self.env, attr, eps)
    reward = self.env.compute_reward(internal_achieved, internal_goal, info)
    return reward

  
  def step(self, action):
    obs, reward, done, info = self.env.step(action)
    obs['achieved_goal'] = np.concatenate([obs['achieved_goal'], self.defaults])
    obs['desired_goal'] = np.concatenate([obs['desired_goal'], self.defaults])
    reward = self.compute_reward(obs['achieved_goal'], obs['desired_goal'], info)
    return obs, reward, done, info


  def _get_obs(self):
    """just adds a large epsilon (content doesn't super matter)"""
    obs = self.env._get_obs()
    obs['achieved_goal'] = np.concatenate([obs['achieved_goal'], self.defaults])
    obs['desired_goal'] = np.concatenate([obs['desired_goal'], self.defaults])
    return obs

  def _sample_goal(self):
    """just adds a large epsilon (content doesn't super matter)"""
    goal = self.env._sample_goal()
    goal = np.concatenate([goal, self.defaults])
    return goal

  def reset(self):
    obs = self.env.reset()
    obs['achieved_goal'] = np.concatenate([obs['achieved_goal'], self.defaults])
    obs['desired_goal'] = np.concatenate([obs['desired_goal'], self.defaults])
    return obs

class OldEpsilonWrapper(object):

  def __init__(self, env, epsilon, compute_reward_with_internal=None):
    """Epsilon is float or np.array, specifying default epsilon"""

    self.env = env
    if hasattr(self.env, 'mode'):
      assert self.env.mode == 0

    if compute_reward_with_internal is not None:
      self.compute_reward_with_internal = compute_reward_with_internal

    obs = self.env.reset()
    self.default_epsilon = np.ones_like(obs['desired_goal']) * epsilon
    
    self.observation_space = spaces.Dict(dict(
        desired_goal=spaces.Box(-np.inf, np.inf, shape=(obs['achieved_goal'].shape[0]*2,), dtype='float32'),
        achieved_goal=spaces.Box(-np.inf, np.inf, shape=(obs['achieved_goal'].shape[0]*2,), dtype='float32'),
        observation=spaces.Box(-np.inf, np.inf, shape=(obs['observation'].shape[0],), dtype='float32'),
    ))


  def __getattr__(self, attr):
    if attr in self.__dict__:
      return getattr(self, attr)
    return getattr(self.env, attr)

  def compute_reward(self, achieved_goal, goal, info):
    internal_len = len(achieved_goal) // 2
    internal_achieved = achieved_goal[:internal_len]
    internal_goal = goal[:internal_len]
    
    if self.compute_reward_with_internal:
      internal_reward = self.env.compute_reward(internal_achieved, internal_goal, info)
      return internal_reward

    # Otherwise, use epsilon to determine external reward
    epsilon = goal[internal_len:]

    success = np.all(np.abs(internal_achieved - internal_goal) < epsilon)
    return success - 1.

  
  def step(self, action):
    obs, reward, done, info = self.env.step(action)
    obs['achieved_goal'] = np.concatenate([obs['achieved_goal'], self.default_epsilon])
    obs['desired_goal'] = np.concatenate([obs['desired_goal'], self.default_epsilon])
    reward = self.compute_reward(obs['achieved_goal'], obs['desired_goal'], info)
    return obs, reward, done, info


  def _get_obs(self):
    """just adds a large epsilon (content doesn't super matter)"""
    obs = self.env._get_obs()
    obs['achieved_goal'] = np.concatenate([obs['achieved_goal'], self.default_epsilon])
    obs['desired_goal'] = np.concatenate([obs['desired_goal'], self.default_epsilon])
    return obs

  def _sample_goal(self):
    """just adds a large epsilon (content doesn't super matter)"""
    goal = self.env._sample_goal()
    goal = np.concatenate([goal, self.default_epsilon])
    return goal

  def reset(self):
    obs = self.env.reset()
    obs['achieved_goal'] = np.concatenate([obs['achieved_goal'], self.default_epsilon])
    obs['desired_goal'] = np.concatenate([obs['desired_goal'], self.default_epsilon])
    return obs