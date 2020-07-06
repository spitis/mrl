import numpy as np
import os
from gym.envs.robotics.hand.manipulate import ManipulateEnv
from gym.envs.robotics.hand.reach import HandReachEnv

MANIPULATE_BLOCK_XML = os.path.join('hand', 'manipulate_block.xml')
MANIPULATE_PEN_XML = os.path.join('hand', 'manipulate_pen.xml')
MANIPULATE_EGG_XML = os.path.join('hand', 'manipulate_egg.xml')

######## BLOCK ########
class HandBlockEnv(ManipulateEnv):
  def __init__(self, max_step = 100, target_position='random', target_rotation='xyz', reward_type='sparse', distance_threshold=0.01, rotation_threshold=0.1):
    self.num_step = 0
    self.max_step = max_step
    super(HandBlockEnv, self).__init__(
      model_path=MANIPULATE_BLOCK_XML, target_position=target_position,
      target_rotation=target_rotation,
      target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
      reward_type='sparse', distance_threshold=distance_threshold, rotation_threshold=rotation_threshold)
  
  def step(self, action):
    obs, reward, _, info = super().step(action)
    self.num_step += 1
    done = True if self.num_step >= self.max_step else False

    return obs, reward, done, info
    
  def reset(self):
    obs = super().reset()
    self.num_step = 0
    return obs

######## PEN ########
class HandPenEnv(ManipulateEnv):
  def __init__(self, max_step = 100, target_position='random', target_rotation='xyz', reward_type='sparse', distance_threshold=0.05, rotation_threshold=0.1):
    self.num_step = 0
    self.max_step = max_step
    super(HandPenEnv, self).__init__(
      model_path=MANIPULATE_PEN_XML, target_position=target_position,
      target_rotation=target_rotation,
      target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
      randomize_initial_rotation=False, reward_type=reward_type,
      ignore_z_target_rotation=True, distance_threshold=distance_threshold, rotation_threshold=rotation_threshold)
  
  def step(self, action):
    obs, reward, _, info = super().step(action)
    self.num_step += 1
    done = True if self.num_step >= self.max_step else False

    return obs, reward, done, info
    
  def reset(self):
    obs = super().reset()
    self.num_step = 0
    return obs

######## EGG ########
class HandEggEnv(ManipulateEnv):
  def __init__(self, max_step = 100, target_position='random', target_rotation='xyz', reward_type='sparse', distance_threshold=0.01, rotation_threshold=0.1):
    self.num_step = 0
    self.max_step = max_step
    super(HandEggEnv, self).__init__(
      model_path=MANIPULATE_EGG_XML, target_position=target_position,
      target_rotation=target_rotation,
      target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
      reward_type=reward_type, distance_threshold=distance_threshold, rotation_threshold=rotation_threshold)

  def step(self, action):
    obs, reward, _, info = super().step(action)
    self.num_step += 1
    done = True if self.num_step >= self.max_step else False

    return obs, reward, done, info

  def reset(self):
    obs = super().reset()
    self.num_step = 0
    return obs

######## REACH ########
class HandReachFullEnv(HandReachEnv):
  def __init__(self, max_step = 50, distance_threshold=0.01, n_substeps=20, reward_type='sparse',):
    self.num_step = 0
    self.max_step = max_step
    super(HandReachFullEnv, self).__init__(
      distance_threshold=distance_threshold, n_substeps=n_substeps, reward_type=reward_type,
    )

  def step(self, action):
    obs, reward, _, info = super().step(action)
    self.num_step += 1
    done = True if self.num_step >= self.max_step else False

    return obs, reward, done, info

  def reset(self):
    obs = super().reset()
    self.num_step = 0
    return obs


