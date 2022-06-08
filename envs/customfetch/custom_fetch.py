import numpy as np
from gym.envs.robotics.fetch_env import FetchEnv, goal_distance
from gym.envs.robotics.fetch.push import FetchPushEnv
from gym.envs.robotics.fetch.slide import FetchSlideEnv
from gym.envs.robotics.fetch.pick_and_place import FetchPickAndPlaceEnv
from gym.envs.robotics import fetch_env
from gym import spaces
import os
from gym.utils import EzPickle
from enum import Enum
from sklearn.metrics.pairwise import euclidean_distances
from scipy.linalg import block_diag

from gym.envs.robotics import rotations, utils

dir_path = os.path.dirname(os.path.realpath(__file__))
STACKXML = os.path.join(dir_path, 'xmls', 'FetchStack#.xml')
ORIGPUSHXML = os.path.join(dir_path, 'xmls', 'Push.xml')
ORIGSLIDEXML = os.path.join(dir_path, 'xmls', 'Slide.xml')
PUSHXML = os.path.join(dir_path, 'xmls', 'CustomPush.xml')
PPXML = os.path.join(dir_path, 'xmls', 'CustomPP.xml')
SLIDEXML = os.path.join(dir_path, 'xmls', 'CustomSlide.xml')
SLIDE_N_XML = os.path.join(dir_path, 'xmls', 'FetchSlide#.xml')
PUSH_N_XML = os.path.join(dir_path, 'xmls', 'FetchPush#.xml')

INIT_Q_POSES = [
    [1.3, 0.6, 0.41, 1., 0., 0., 0.],
    [1.3, 0.9, 0.41, 1., 0., 0., 0.],
    [1.2, 0.68, 0.41, 1., 0., 0., 0.],
    [1.4, 0.82, 0.41, 1., 0., 0., 0.],
    [1.4, 0.68, 0.41, 1., 0., 0., 0.],
    [1.2, 0.82, 0.41, 1., 0., 0., 0.],
]
INIT_Q_POSES_SLIDE = [
    [1.3, 0.7, 0.42, 1., 0., 0., 0.],
    [1.3, 0.9, 0.42, 1., 0., 0., 0.],
    [1.25, 0.8, 0.42, 1., 0., 0., 0.],
    [1.35, 0.8, 0.42, 1., 0., 0., 0.],
    [1.35, 0.7, 0.42, 1., 0., 0., 0.],
    [1.25, 0.9, 0.42, 1., 0., 0., 0.],
]


class GoalType(Enum):
  OBJ = 1
  GRIP = 2
  OBJ_GRIP = 3
  ALL = 4
  OBJSPEED = 5
  OBJSPEED2 = 6
  OBJ_GRIP_GRIPPER = 7


def compute_reward(achieved_goal, goal, internal_goal, distance_threshold, per_dim_threshold,
                   compute_reward_with_internal, mode):
  # Always require internal success.
  internal_success = 0.
  if internal_goal == GoalType.OBJ_GRIP:
    if len(achieved_goal.shape) == 1:
      d = goal_distance(achieved_goal[:6], goal[:6])
    else:
      d = goal_distance(achieved_goal[:, :6], goal[:, :6])
  elif internal_goal in [GoalType.GRIP, GoalType.OBJ]:
    if len(achieved_goal.shape) == 1:
      d = goal_distance(achieved_goal[:3], goal[:3])
    else:
      d = goal_distance(achieved_goal[:, :3], goal[:, :3])
  else:
    raise

  internal_success = (d <= distance_threshold).astype(np.float32)

  if compute_reward_with_internal:
    return internal_success - (1. - mode)

  # use per_dim_thresholds for other dimensions
  success = np.all(np.abs(achieved_goal - goal) < per_dim_threshold, axis=-1)
  success = np.logical_and(internal_success, success).astype(np.float32)
  return success - (1. - mode)


def get_obs(sim, external_goal, goal, subtract_obj_velp=True):
  # positions
  grip_pos = sim.data.get_site_xpos('robot0:grip')
  dt = sim.nsubsteps * sim.model.opt.timestep
  grip_velp = sim.data.get_site_xvelp('robot0:grip') * dt
  robot_qpos, robot_qvel = utils.robot_get_obs(sim)

  object_pos = sim.data.get_site_xpos('object0').ravel()
  # rotations
  object_rot = rotations.mat2euler(sim.data.get_site_xmat('object0')).ravel()
  # velocities
  object_velp = (sim.data.get_site_xvelp('object0') * dt).ravel()
  object_velr = (sim.data.get_site_xvelr('object0') * dt).ravel()
  # gripper state
  object_rel_pos = object_pos - grip_pos
  if subtract_obj_velp:
    object_velp -= grip_velp

  gripper_state = robot_qpos[-2:]
  gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric

  items = [
      grip_pos,
      object_pos,
      object_rel_pos,
      gripper_state,
      object_rot,
      object_velp,
      object_velr,
      grip_velp,
      gripper_vel,
  ]

  obs = np.concatenate(items)

  if external_goal == GoalType.ALL:
    achieved_goal = np.concatenate([
        object_pos,
        grip_pos,
        object_rel_pos,
        gripper_state,
        object_rot,
        object_velp,
        object_velr,
        grip_velp,
        gripper_vel,
    ])
  elif external_goal == GoalType.OBJ:
    achieved_goal = object_pos
  elif external_goal == GoalType.OBJ_GRIP:
    achieved_goal = np.concatenate([object_pos, grip_pos])
  elif external_goal == GoalType.OBJ_GRIP_GRIPPER:
    achieved_goal = np.concatenate([object_pos, grip_pos, gripper_state])
  elif external_goal == GoalType.OBJSPEED:
    achieved_goal = np.concatenate([object_pos, object_velp])
  elif external_goal == GoalType.OBJSPEED2:
    achieved_goal = np.concatenate([object_pos, object_velp, object_velr])
  else:
    raise ValueError('unsupported goal type!')

  return {
      'observation': obs,
      'achieved_goal': achieved_goal,
      'desired_goal': goal.copy(),
  }


def sample_goal(initial_gripper_xpos, np_random, target_range, target_offset, height_offset, internal_goal,
                external_goal, grip_offset, gripper_goal):
  obj_goal = initial_gripper_xpos[:3] + np_random.uniform(-target_range, target_range, size=3)
  obj_goal += target_offset
  obj_goal[2] = height_offset

  if internal_goal in [GoalType.GRIP, GoalType.OBJ_GRIP]:
    grip_goal = initial_gripper_xpos[:3] + np_random.uniform(-0.15, 0.15, size=3) + np.array([0., 0., 0.15])
    obj_rel_goal = obj_goal - grip_goal
  else:
    grip_goal = obj_goal + grip_offset
    obj_rel_goal = -grip_offset

  if external_goal == GoalType.ALL:
    return np.concatenate([obj_goal, grip_goal, obj_rel_goal, gripper_goal, [0.] * 14])
  elif external_goal == GoalType.OBJ:
    return obj_goal
  elif external_goal == GoalType.OBJ_GRIP_GRIPPER:
    return np.concatenate([obj_goal, grip_goal, gripper_goal])
  elif external_goal == GoalType.OBJ_GRIP:
    return np.concatenate([obj_goal, grip_goal])
  elif external_goal == GoalType.OBJSPEED:
    return np.concatenate([obj_goal, [0.] * 3])
  elif external_goal == GoalType.OBJSPEED2:
    return np.concatenate([obj_goal, [0.] * 6])
  elif external_goal == GoalType.GRIP:
    raise NotImplementedError

  raise ValueError("BAD external goal value")


class StackEnv(fetch_env.FetchEnv, EzPickle):
  def __init__(self,
               max_step=50,
               n=1,
               internal_goal=GoalType.OBJ,
               external_goal=GoalType.OBJ,
               mode="-1/0",
               compute_reward_with_internal=False,
               per_dim_threshold=None,
               hard=False,
               distance_threshold=0.,
               range_min=None,
               range_max=None):
    self.internal_goal = internal_goal
    self.external_goal = external_goal
    self.n = n
    self.hard = hard

    initial_qpos = {
        'robot0:slide0': 0.405,
        'robot0:slide1': 0.48,
        'robot0:slide2': 0.0,
    }
    for i in range(self.n):
      k = 'object{}:joint'.format(i)
      initial_qpos[k] = INIT_Q_POSES[i]

    fetch_env.FetchEnv.__init__(self,
                                STACKXML.replace('#', '{}'.format(n)),
                                has_object=True,
                                block_gripper=False,
                                n_substeps=20,
                                gripper_extra_height=0.2,
                                target_in_the_air=False,
                                target_offset=0.0,
                                obj_range=0.15,
                                target_range=0.1,
                                distance_threshold=0.05,
                                initial_qpos=initial_qpos,
                                reward_type='sparse')

    EzPickle.__init__(self)

    self.max_step = max(50 * (n - 1), 50)
    self.num_step = 0

    self.mode = 0
    if mode == "0/1" or mode == 1:
      self.mode = 1

    if self.external_goal == internal_goal:
      self.compute_reward_with_internal = True
    else:
      self.compute_reward_with_internal = compute_reward_with_internal

    self.per_dim_threshold = np.sqrt(self.distance_threshold**2 / 3)
    if per_dim_threshold:
      raise NotImplementedError

  def compute_reward(self, achieved_goal, goal, info):
    # Compute distance between goal and the achieved goal.

    if self.external_goal == GoalType.OBJ_GRIP:
      if len(achieved_goal.shape) == 1:
        actual_internal_goals = np.split(goal[:-3], self.n)
        achieved_internal_goals = np.split(achieved_goal[:-3], self.n)
      else:
        actual_internal_goals = np.split(goal[:, :-3], self.n, axis=1)
        achieved_internal_goals = np.split(achieved_goal[:, :-3], self.n, axis=1)
    elif self.external_goal == GoalType.OBJ:
      if len(achieved_goal.shape) == 1:
        actual_internal_goals = np.split(goal, self.n)
        achieved_internal_goals = np.split(achieved_goal, self.n)
      else:
        actual_internal_goals = np.split(goal, self.n, axis=1)
        achieved_internal_goals = np.split(achieved_goal, self.n, axis=1)
    else:
      raise

    if len(achieved_goal.shape) == 1:
      success = 1.
    else:
      success = np.ones(achieved_goal.shape[0], dtype=np.float32)

    for b, g in zip(achieved_internal_goals, actual_internal_goals):
      d = goal_distance(b, g)
      success *= (d <= self.distance_threshold).astype(np.float32)

    if self.compute_reward_with_internal:
      return success - (1. - self.mode)

    # use per_dim_thresholds for other dimensions
    if len(achieved_goal.shape) == 1:
      d = goal_distance(achieved_goal[-3:], goal[-3:])
    else:
      d = goal_distance(achieved_goal[:, -3:], goal[:, -3:])
    success *= (d <= self.distance_threshold).astype(np.float32)

    return success - (1. - self.mode)

  def _get_obs(self):
    # positions
    grip_pos = self.sim.data.get_site_xpos('robot0:grip')
    dt = self.sim.nsubsteps * self.sim.model.opt.timestep
    grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt
    robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)

    obj_feats = []
    obj_poses = []

    for i in range(self.n):
      obj_labl = 'object{}'.format(i)
      object_pos = self.sim.data.get_site_xpos(obj_labl).ravel()
      object_pos[2] = max(object_pos[2], self.height_offset)
      # rotations
      object_rot = rotations.mat2euler(self.sim.data.get_site_xmat(obj_labl)).ravel()
      # velocities
      object_velp = (self.sim.data.get_site_xvelp(obj_labl) * dt).ravel()
      object_velr = (self.sim.data.get_site_xvelr(obj_labl) * dt).ravel()
      # gripper state
      object_rel_pos = object_pos - grip_pos
      #object_velp -= grip_velp

      obj_feats.append([object_pos, object_rel_pos, object_rot, object_velp, object_velr])
      obj_poses.append(object_pos)

    gripper_state = robot_qpos[-2:]
    gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric

    if self.external_goal == GoalType.OBJ_GRIP:
      achieved_goal = np.concatenate(obj_poses + [grip_pos])
    else:
      achieved_goal = np.concatenate(obj_poses)
    obs = np.concatenate([grip_pos, gripper_state, grip_velp, gripper_vel] + sum(obj_feats, []))

    return {
        'observation': obs,
        'achieved_goal': achieved_goal,
        'desired_goal': self.goal.copy(),
    }

  def _render_callback(self):
    # Visualize target.
    sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos)
    if self.external_goal == GoalType.OBJ_GRIP:
      goals = np.split(self.goal[:-3], self.n)
    else:
      goals = np.split(self.goal, self.n)

    for i in range(self.n):
      site_id = self.sim.model.site_name2id('target{}'.format(i))
      self.sim.model.site_pos[site_id] = goals[i] - sites_offset[i]
    self.sim.forward()

  def _reset_sim(self):
    self.sim.set_state(self.initial_state)

    # Only a little randomize about the start state
    for i in range(self.n):
      object_qpos = self.sim.data.get_joint_qpos('object{}:joint'.format(i))
      object_qpos[:2] += self.np_random.uniform(-0.03, 0.03, size=2)
      self.sim.data.set_joint_qpos('object{}:joint'.format(i), object_qpos)

    bad_poses = [self.initial_gripper_xpos[:2]]
    # Randomize start positions of boxes.
    # for i in range(self.n):
    #   object_xpos = self.initial_gripper_xpos[:2]
    #   while min([np.linalg.norm(object_xpos - p) for p in bad_poses]) < 0.1:
    #       object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
    #   bad_poses.append(object_xpos)

    #   object_qpos = self.sim.data.get_joint_qpos('object{}:joint'.format(i))
    #   assert object_qpos.shape == (7,)
    #   object_qpos[:2] = object_xpos
    #   self.sim.data.set_joint_qpos('object{}:joint'.format(i), object_qpos)

    self.sim.forward()
    return True

  def _sample_goal(self):
    bottom_box = self.initial_gripper_xpos[:3] + self.np_random.uniform(-self.target_range, self.target_range, size=3)
    bottom_box[2] = self.height_offset  #self.sim.data.get_joint_qpos('object0:joint')[:3]

    goal = []
    for i in range(self.n):
      goal.append(bottom_box + (np.array([0., 0., 0.05]) * i))

    if self.external_goal == GoalType.OBJ_GRIP:
      goal.append(goal[-1] + np.array([-0.01, 0., 0.008]))

    return np.concatenate(goal)

  def step(self, action):
    obs, reward, _, info = super().step(action)
    self.num_step += 1
    done = True if self.num_step >= self.max_step else False
    if done: info['TimeLimit.truncated'] = True

    if self.mode == 1 and reward:
      done = True

    info['is_success'] = np.allclose(reward, self.mode)
    return obs, reward, done, info

  def reset(self):
    obs = super().reset()
    self.num_step = 0
    return obs


class PushEnv(FetchPushEnv):
  def __init__(self,
               max_step=51,
               internal_goal=GoalType.OBJ,
               external_goal=GoalType.OBJ,
               mode="-1/0",
               compute_reward_with_internal=False,
               per_dim_threshold=None,
               hard=False,
               distance_threshold=0.,
               n=0,
               range_min=None,
               range_max=None):
    self.internal_goal = internal_goal
    self.external_goal = external_goal
    if hard or n > 0:
      raise ValueError("Hard not supported")
    super().__init__(reward_type='sparse')

    if distance_threshold > 1e-5:
      self.distance_threshold = distance_threshold

    if self.internal_goal == GoalType.OBJ_GRIP:
      self.distance_threshold *= np.sqrt(2)

    self.max_step = max_step
    self.num_step = 0
    self.mode = 0
    if mode == "0/1" or mode == 1:
      self.mode = 1

    if self.external_goal == internal_goal:
      self.compute_reward_with_internal = True
    else:
      self.compute_reward_with_internal = compute_reward_with_internal

    self.per_dim_threshold = np.sqrt(self.distance_threshold**2 / 3)
    if per_dim_threshold:
      self.per_dim_threshold = per_dim_threshold
    print('PER DIM THRESHOLD:', self.per_dim_threshold)

  def compute_reward(self, achieved_goal, goal, info):
    return compute_reward(achieved_goal, goal, self.internal_goal, self.distance_threshold, self.per_dim_threshold,
                          self.compute_reward_with_internal, self.mode)

  def _render_callback(self):
    # Visualize target.
    sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
    site_id = self.sim.model.site_name2id('target0')

    goal = self.goal[:3]

    self.sim.model.site_pos[site_id] = goal - sites_offset[0]
    self.sim.forward()

  def _get_obs(self):
    return get_obs(self.sim, self.external_goal, self.goal)

  def _sample_goal(self):
    return sample_goal(self.initial_gripper_xpos, self.np_random, self.target_range, self.target_offset,
                       self.height_offset, self.internal_goal, self.external_goal, np.array([0., 0., 0.05]), [0., 0.])

  def step(self, action):
    obs, reward, _, info = super().step(action)
    self.num_step += 1
    done = True if self.num_step >= self.max_step else False
    if done: info['TimeLimit.truncated'] = True

    if self.mode == 1 and reward:
      done = True

    info['is_success'] = np.allclose(reward, self.mode)
    return obs, reward, done, info

  def reset(self):
    obs = super().reset()
    self.num_step = 0
    return obs


class SlideEnv(FetchSlideEnv):
  def __init__(self,
               max_step=51,
               internal_goal=GoalType.OBJ,
               external_goal=GoalType.OBJ,
               mode="-1/0",
               compute_reward_with_internal=False,
               per_dim_threshold=None,
               hard=False,
               distance_threshold=0.,
               n=0,
               range_min=None,
               range_max=None):
    self.internal_goal = internal_goal
    self.external_goal = external_goal

    self.subtract_obj_velp = True
    if self.external_goal in [GoalType.OBJSPEED, GoalType.OBJSPEED2]:
      self.subtract_obj_velp = False

    if hard or n > 0:
      raise ValueError("Hard not supported")
    super().__init__(reward_type='sparse')

    if distance_threshold > 1e-5:
      self.distance_threshold = distance_threshold

    self.max_step = max_step
    self.num_step = 0
    self.mode = 0
    if mode == "0/1" or mode == 1:
      self.mode = 1

    if self.external_goal == internal_goal:
      self.compute_reward_with_internal = True
    else:
      self.compute_reward_with_internal = compute_reward_with_internal

    self.per_dim_threshold = np.sqrt(self.distance_threshold**2 / 3)
    if isinstance(per_dim_threshold, float) and per_dim_threshold > 1e-3:
      self.per_dim_threshold = per_dim_threshold
    print('PER DIM THRESHOLD:', self.per_dim_threshold)

  def compute_reward(self, achieved_goal, goal, info):
    return compute_reward(achieved_goal, goal, self.internal_goal, self.distance_threshold, self.per_dim_threshold,
                          self.compute_reward_with_internal, self.mode)

  def _render_callback(self):
    # Visualize target.
    sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
    site_id = self.sim.model.site_name2id('target0')

    goal = self.goal[:3]

    self.sim.model.site_pos[site_id] = goal - sites_offset[0]
    self.sim.forward()

  def _get_obs(self):
    return get_obs(self.sim, self.external_goal, self.goal, self.subtract_obj_velp)

  def _sample_goal(self):
    return sample_goal(self.initial_gripper_xpos, self.np_random, self.target_range, self.target_offset,
                       self.height_offset, self.internal_goal, self.external_goal, np.array([0., 0., 0.05]), [0., 0.])

  def step(self, action):
    obs, reward, _, info = super().step(action)
    self.num_step += 1
    done = True if self.num_step >= self.max_step else False
    if done: info['TimeLimit.truncated'] = True

    if self.mode == 1 and reward:
      done = True

    info['is_success'] = np.allclose(reward, self.mode)
    return obs, reward, done, info

  def reset(self):
    obs = super().reset()
    self.num_step = 0
    return obs


class PickPlaceEnv(FetchPickAndPlaceEnv):
  def __init__(self,
               max_step=51,
               internal_goal=GoalType.OBJ,
               external_goal=GoalType.OBJ,
               mode="-1/0",
               compute_reward_with_internal=False,
               per_dim_threshold=None,
               hard=False,
               distance_threshold=0.,
               n=0.5,
               range_min=0.2,
               range_max=0.45):
    self.internal_goal = internal_goal
    self.external_goal = external_goal
    if hard:
      self.minimum_air = range_min
      self.maximum_air = range_max
    else:
      self.minimum_air = 0.
      self.maximum_air = range_max
    self.in_air_percentage = n
    super().__init__(reward_type='sparse')

    if distance_threshold > 1e-5:
      self.distance_threshold = distance_threshold

    self.max_step = max_step
    self.num_step = 0
    self.mode = 0
    if mode == "0/1" or mode == 1:
      self.mode = 1

    if self.external_goal == internal_goal:
      self.compute_reward_with_internal = True
    else:
      self.compute_reward_with_internal = compute_reward_with_internal

    self.per_dim_threshold = np.sqrt(self.distance_threshold**2 / 3)
    if per_dim_threshold:
      self.per_dim_threshold = per_dim_threshold
    print('PER DIM THRESHOLD:', self.per_dim_threshold)

  def compute_reward(self, achieved_goal, goal, info):
    return compute_reward(achieved_goal, goal, self.internal_goal, self.distance_threshold, self.per_dim_threshold,
                          self.compute_reward_with_internal, self.mode)

  def _render_callback(self):
    # Visualize target.
    sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
    site_id = self.sim.model.site_name2id('target0')

    goal = self.goal[:3]

    self.sim.model.site_pos[site_id] = goal - sites_offset[0]
    self.sim.forward()

  def _get_obs(self):
    return get_obs(self.sim, self.external_goal, self.goal)

  def _sample_goal(self):
    height_offset = self.height_offset
    if self.np_random.uniform() < self.in_air_percentage:
      height_offset += self.np_random.uniform(self.minimum_air, self.maximum_air)
    return sample_goal(self.initial_gripper_xpos, self.np_random, self.target_range, self.target_offset, height_offset,
                       self.internal_goal, self.external_goal, np.array([-0.01, 0., 0.008]), [0.024273, 0.024273])

  def step(self, action):
    obs, reward, _, info = super().step(action)
    self.num_step += 1
    done = True if self.num_step >= self.max_step else False
    if done: info['TimeLimit.truncated'] = True

    if self.mode == 1 and reward:
      done = True

    info['is_success'] = np.allclose(reward, self.mode)
    return obs, reward, done, info

  def reset(self):
    obs = super().reset()
    self.num_step = 0
    return obs


###########
# Environments with random weight matrix goal projection
# Same as the environments above, but the goal vector returned is now multiplied
# by a (fixed) random vector, initialized when the env is instantiated
##########


class PushRandGoalDistEnv(PushEnv):
  def __init__(self,
               max_step=51,
               internal_goal=GoalType.OBJ,
               external_goal=GoalType.OBJ,
               mode="-1/0",
               compute_reward_with_internal=False,
               per_dim_threshold=None,
               compute_external_reward_with_soft_threshold=0,
               seed=123):
    self.done_init = False
    # May be there's a nicer way to pass down the init to parent
    super().__init__(max_step=max_step,
                     internal_goal=internal_goal,
                     external_goal=external_goal,
                     mode=mode,
                     compute_reward_with_internal=compute_reward_with_internal,
                     per_dim_threshold=per_dim_threshold,
                     compute_external_reward_with_soft_threshold=compute_external_reward_with_soft_threshold)
    self.done_init = True
    # Additionally sample a random invertible matrix
    self.seed(seed)

    # Get the size of the goal for this configuration from parent class
    goal_shape = super()._sample_goal().shape

    W = self.np_random.randn(goal_shape[0], goal_shape[0])
    # Check if W is invertible, sample new ones if not
    while not np.isfinite(np.linalg.cond(W)):
      W = self.np_random.randn(goal_shape[0], goal_shape[0])

    self.rand_goal_W = W
    self.rand_goal_W_inv = np.linalg.inv(W)

  def _sample_goal(self):
    goal = super()._sample_goal()

    # Check if has done init yet. If not then just use the original goal space
    if self.done_init:
      # Apply random distillation
      return np.dot(self.rand_goal_W, goal)
    else:
      return goal

  def _render_callback(self):
    # Visualize target.
    sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
    site_id = self.sim.model.site_name2id('target0')

    if self.external_goal == GoalType.OBJ:
      goal = self.goal
    elif self.external_goal == GoalType.OBJSPEED:
      goal = self.goal[:3]
    else:
      goal = self.goal[3:6]

    self.sim.model.site_pos[site_id] = goal - sites_offset[0]
    self.sim.forward()

  def _get_obs(self):
    observation = super()._get_obs()

    if self.done_init:
      # Apply random distillation for the achieved goal.
      # desired_goal is already in the distillation form
      observation['achieved_goal'] = np.dot(self.rand_goal_W, observation['achieved_goal'])

    return observation

  def compute_reward(self, achieved_goal, goal, info):
    # Invert the goals back to original goal space, then use compute reward func from parent

    og_achieved_goal = np.dot(self.rand_goal_W_inv, achieved_goal)
    og_desired_goal = np.dot(self.rand_goal_W_inv, goal)

    reward = super().compute_reward(og_achieved_goal, og_desired_goal, info)

    return reward


class SlideRandGoalDistEnv(SlideEnv):
  def __init__(self,
               max_step=51,
               internal_goal=GoalType.OBJ,
               external_goal=GoalType.OBJ,
               mode="-1/0",
               compute_reward_with_internal=False,
               per_dim_threshold=None,
               seed=123):
    self.done_init = False
    super().__init__(max_step=max_step,
                     internal_goal=internal_goal,
                     external_goal=external_goal,
                     mode=mode,
                     compute_reward_with_internal=compute_reward_with_internal,
                     per_dim_threshold=per_dim_threshold)
    self.done_init = True
    # Additionally sample a random invertible matrix
    self.seed(seed)

    # Get the size of the goal for this configuration from parent class
    goal_shape = super()._sample_goal().shape

    W = self.np_random.randn(goal_shape[0], goal_shape[0])
    # Check if W is invertible, sample new ones if not
    while not np.isfinite(np.linalg.cond(W)):
      W = self.np_random.randn(goal_shape[0], goal_shape[0])

    self.rand_goal_W = W
    self.rand_goal_W_inv = np.linalg.inv(W)

  def _sample_goal(self):
    goal = super()._sample_goal()

    # Check if has done init yet. If not then just use the original goal space
    if self.done_init:
      # Apply random distillation
      return np.dot(self.rand_goal_W, goal)
    else:
      return goal

  def _get_obs(self):
    observation = super()._get_obs()

    if self.done_init:
      # Apply random distillation for the achieved goal.
      # desired_goal is already in the distillation form
      observation['achieved_goal'] = np.dot(self.rand_goal_W, observation['achieved_goal'])

    return observation

  def compute_reward(self, achieved_goal, goal, info):
    # Invert the goals back to original goal space, then use compute reward func from parent

    og_achieved_goal = np.dot(self.rand_goal_W_inv, achieved_goal)
    og_desired_goal = np.dot(self.rand_goal_W_inv, goal)

    reward = super().compute_reward(og_achieved_goal, og_desired_goal, info)

    return reward


class PushGoalVisualizer(fetch_env.FetchEnv, EzPickle):
  def __init__(self):
    initial_qpos = {
        'robot0:slide0': 0.405,
        'robot0:slide1': 0.48,
        'robot0:slide2': 0.0,
        'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
    }
    fetch_env.FetchEnv.__init__(self,
                                PUSHXML,
                                has_object=True,
                                block_gripper=True,
                                n_substeps=20,
                                gripper_extra_height=0.0,
                                target_in_the_air=False,
                                target_offset=0.0,
                                obj_range=0.15,
                                target_range=0.15,
                                distance_threshold=0.05,
                                initial_qpos=initial_qpos,
                                reward_type='sparse')

    EzPickle.__init__(self)

  def _render_callback(self):
    # Visualize target.
    sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
    site_id = self.sim.model.site_name2id('target0')

    self.sim.model.site_pos[site_id] = self.goal - sites_offset[0]
    site_id = self.sim.model.site_name2id('target1')
    self.sim.model.site_pos[site_id] = self.goal - sites_offset[0] + np.array([0., 0., 0.05])

    self.sim.forward()


class PPGoalVisualizer(fetch_env.FetchEnv, EzPickle):
  def __init__(self):
    initial_qpos = {
        'robot0:slide0': 0.405,
        'robot0:slide1': 0.48,
        'robot0:slide2': 0.0,
        'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
    }
    fetch_env.FetchEnv.__init__(self,
                                PPXML,
                                has_object=True,
                                block_gripper=False,
                                n_substeps=20,
                                gripper_extra_height=0.2,
                                target_in_the_air=True,
                                target_offset=0.0,
                                obj_range=0.15,
                                target_range=0.15,
                                distance_threshold=0.05,
                                initial_qpos=initial_qpos,
                                reward_type='sparse')

    EzPickle.__init__(self)

  def _render_callback(self):
    # Visualize target.
    sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
    site_id = self.sim.model.site_name2id('target0')

    self.sim.model.site_pos[site_id] = self.goal - sites_offset[0]
    site_id = self.sim.model.site_name2id('target1')
    self.sim.model.site_pos[site_id] = self.goal - sites_offset[0] + np.array([-0.01, 0., 0.008])

    self.sim.forward()


class SlideGoalVisualizer(fetch_env.FetchEnv, EzPickle):
  def __init__(self):
    initial_qpos = {
        'robot0:slide0': 0.05,
        'robot0:slide1': 0.48,
        'robot0:slide2': 0.0,
        'table0:slide0': 0.7,
        'table0:slide1': 0.3,
        'table0:slide2': 0.0,
        'object0:joint': [1.7, 1.1, 0.4, 1., 0., 0., 0.],
    }
    fetch_env.FetchEnv.__init__(self,
                                SLIDEXML,
                                has_object=True,
                                block_gripper=True,
                                n_substeps=20,
                                gripper_extra_height=-0.02,
                                target_in_the_air=False,
                                target_offset=np.array([0.4, 0.0, 0.0]),
                                obj_range=0.1,
                                target_range=0.3,
                                distance_threshold=0.05,
                                initial_qpos=initial_qpos,
                                reward_type='sparse')

    EzPickle.__init__(self)

  def _render_callback(self):
    # Visualize target.
    sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
    site_id = self.sim.model.site_name2id('target0')

    self.sim.model.site_pos[site_id] = self.goal - sites_offset[0]
    site_id = self.sim.model.site_name2id('target1')
    self.sim.model.site_pos[site_id] = self.goal - sites_offset[0]

    self.sim.forward()


class PushLeft(fetch_env.FetchEnv, EzPickle):
  def __init__(self, reward_type='sparse'):
    initial_qpos = {
        'robot0:slide0': 0.405,
        'robot0:slide1': 0.48,
        'robot0:slide2': 0.0,
        'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
    }
    self.max_step = 50
    self.num_step = 0
    fetch_env.FetchEnv.__init__(self,
                                ORIGPUSHXML,
                                has_object=True,
                                block_gripper=True,
                                n_substeps=20,
                                gripper_extra_height=0.0,
                                target_in_the_air=False,
                                target_offset=0.0,
                                obj_range=0.15,
                                target_range=0.15,
                                distance_threshold=0.05,
                                initial_qpos=initial_qpos,
                                reward_type=reward_type)
    EzPickle.__init__(self)

  def _env_setup(self, initial_qpos):
    for name, value in initial_qpos.items():
      self.sim.data.set_joint_qpos(name, value)
    utils.reset_mocap_welds(self.sim)
    self.sim.forward()

    delta = np.array([-0.2, 0., 0.])
    # Move end effector into position.
    gripper_target = np.array([-0.498, 0.005, -0.431 + self.gripper_extra_height
                               ]) + delta + self.sim.data.get_site_xpos('robot0:grip')
    gripper_rotation = np.array([1., 0., 1., 0.])
    self.sim.data.set_mocap_pos('robot0:mocap', gripper_target)
    self.sim.data.set_mocap_quat('robot0:mocap', gripper_rotation)
    for _ in range(10):
      self.sim.step()

    # Extract information for sampling goals.
    self.initial_gripper_xpos = np.array([1.34182673, 0.74891285, 0.41317183])
    if self.has_object:
      self.height_offset = self.sim.data.get_site_xpos('object0')[2]

  def step(self, action):
    obs, reward, _, info = super().step(action)
    self.num_step += 1
    done = True if self.num_step >= self.max_step else False
    return obs, reward, done, info

  def reset(self):
    obs = super().reset()
    self.num_step = 0
    return obs

  def _sample_goal(self):
    if self.has_object:
      if self.np_random.random() < 0.15:
        goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-self.target_range, self.target_range, size=3)
      else:
        goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-self.target_range, 0., size=3)
      goal += self.target_offset
      goal[2] = self.height_offset
      if self.target_in_the_air and self.np_random.uniform() < 0.5:
        goal[2] += self.np_random.uniform(0, 0.45)
    else:
      goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-0.15, 0.15, size=3)
    return goal.copy()


class PushRight(fetch_env.FetchEnv, EzPickle):
  def __init__(self, reward_type='sparse'):
    initial_qpos = {
        'robot0:slide0': 0.405,
        'robot0:slide1': 0.48,
        'robot0:slide2': 0.0,
        'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
    }
    self.max_step = 50
    self.num_step = 0
    fetch_env.FetchEnv.__init__(self,
                                ORIGPUSHXML,
                                has_object=True,
                                block_gripper=True,
                                n_substeps=20,
                                gripper_extra_height=0.0,
                                target_in_the_air=False,
                                target_offset=0.0,
                                obj_range=0.15,
                                target_range=0.15,
                                distance_threshold=0.05,
                                initial_qpos=initial_qpos,
                                reward_type=reward_type)
    EzPickle.__init__(self)

  def _env_setup(self, initial_qpos):
    for name, value in initial_qpos.items():
      self.sim.data.set_joint_qpos(name, value)
    utils.reset_mocap_welds(self.sim)
    self.sim.forward()

    delta = np.array([0.2, 0., 0.])
    # Move end effector into position.
    gripper_target = np.array([-0.498, 0.005, -0.431 + self.gripper_extra_height
                               ]) + delta + self.sim.data.get_site_xpos('robot0:grip')
    gripper_rotation = np.array([1., 0., 1., 0.])
    self.sim.data.set_mocap_pos('robot0:mocap', gripper_target)
    self.sim.data.set_mocap_quat('robot0:mocap', gripper_rotation)
    for _ in range(10):
      self.sim.step()

    # Extract information for sampling goals.
    self.initial_gripper_xpos = np.array([1.34182673, 0.74891285, 0.41317183])
    if self.has_object:
      self.height_offset = self.sim.data.get_site_xpos('object0')[2]

  def step(self, action):
    obs, reward, _, info = super().step(action)
    self.num_step += 1
    done = True if self.num_step >= self.max_step else False

    return obs, reward, done, info

  def reset(self):
    obs = super().reset()
    self.num_step = 0
    return obs

  def _sample_goal(self):
    if self.has_object:
      if self.np_random.random() < 0.15:
        goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-self.target_range, self.target_range, size=3)
      else:
        goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(0., self.target_range, size=3)
      goal += self.target_offset
      goal[2] = self.height_offset
      if self.target_in_the_air and self.np_random.uniform() < 0.5:
        goal[2] += self.np_random.uniform(0, 0.45)
    else:
      goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-0.15, 0.15, size=3)
    return goal.copy()


class DisentangledFetchPushEnv(FetchPushEnv):
  def _get_obs(self):
    # positions
    grip_pos = self.sim.data.get_site_xpos('robot0:grip')
    dt = self.sim.nsubsteps * self.sim.model.opt.timestep
    grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt
    robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)
    if self.has_object:
        object_pos = self.sim.data.get_site_xpos('object0')
        # rotations
        object_rot = rotations.mat2euler(self.sim.data.get_site_xmat('object0'))
        # velocities
        object_velp = self.sim.data.get_site_xvelp('object0') * dt
        object_velr = self.sim.data.get_site_xvelr('object0') * dt
        # gripper state
        object_rel_pos = object_pos - grip_pos
        object_velp -= grip_velp
    else:
        object_pos = object_rot = object_velp = object_velr = object_rel_pos = np.zeros(0)
    gripper_state = robot_qpos[-2:]
    gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric

    achieved_goal = np.squeeze(object_pos.copy())
    obs = np.concatenate([
        grip_pos, gripper_state, grip_velp, gripper_vel, 
        object_pos.ravel(), object_rot.ravel(), object_velp.ravel(), object_velr.ravel(), 
    ])

    return {
        'observation': obs.copy(),
        'achieved_goal': achieved_goal.copy(),
        'desired_goal': self.goal.copy(),
    }


class DictPush(FetchPushEnv):
  def __init__(self):
    super().__init__()
    obs = self._get_obs()
    self.observation_space = spaces.Dict(dict(
      observation=spaces.Box(-np.inf, np.inf, shape=(), dtype='float32'),
      desired_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
      achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
      gripper=spaces.Box(-np.inf, np.inf, shape=obs['gripper'].shape, dtype='float32'),
      object=spaces.Box(-np.inf, np.inf, shape=obs['object'].shape, dtype='float32'),
      relative=spaces.Box(-np.inf, np.inf, shape=obs['relative'].shape, dtype='float32'),
    ))

  def _get_obs(self):
    dt = self.sim.nsubsteps * self.sim.model.opt.timestep

    # gripper state
    grip_pos = self.sim.data.get_site_xpos('robot0:grip')
    grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt
    robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)
    gripper_state = robot_qpos[-2:]
    gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric

    # object state
    object_pos = self.sim.data.get_site_xpos('object0')
    # rotations
    object_rot = rotations.mat2euler(self.sim.data.get_site_xmat('object0'))
    # velocities
    object_velp = self.sim.data.get_site_xvelp('object0') * dt
    object_velr = self.sim.data.get_site_xvelr('object0') * dt
    # relative states
    object_rel_pos = object_pos - grip_pos
    object_rel_vel = object_velp - grip_velp

    achieved_goal = np.squeeze(object_pos)

    return {
      'observation': np.zeros(0), # hack to get around gym's GoalEnv checks
      'gripper': np.concatenate([grip_pos, gripper_state, grip_velp, gripper_vel]),
      'object': np.concatenate([object_pos, object_rot, object_velp, object_velr]),
      'relative': np.concatenate([object_rel_pos, object_rel_vel]),
      'achieved_goal': achieved_goal,
      'desired_goal': self.goal.copy(),
    }

  def achieved_goal(self, observation):
    obj = observation['object']
    if len(obj.shape) == 1:
      return obj[:3]
    return obj[:, :3]

class DictPushAndReach(FetchPushEnv):
  def __init__(self):
    super().__init__()
    obs = self._get_obs()
    self.observation_space = spaces.Dict(dict(
      observation=spaces.Box(-np.inf, np.inf, shape=(), dtype='float32'),
      desired_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
      achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
      gripper=spaces.Box(-np.inf, np.inf, shape=obs['gripper'].shape, dtype='float32'),
      object=spaces.Box(-np.inf, np.inf, shape=obs['object'].shape, dtype='float32'),
      relative=spaces.Box(-np.inf, np.inf, shape=obs['relative'].shape, dtype='float32'),
      gripper_goal=spaces.Box(-np.inf, np.inf, shape=(3,), dtype='float32'),
      object_goal=spaces.Box(-np.inf, np.inf, shape=(3,), dtype='float32')
    ))

  def _get_obs(self):
    dt = self.sim.nsubsteps * self.sim.model.opt.timestep

    # gripper state
    grip_pos = self.sim.data.get_site_xpos('robot0:grip')
    grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt
    robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)
    gripper_state = robot_qpos[-2:]
    gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric

    # object state
    object_pos = self.sim.data.get_site_xpos('object0')
    # rotations
    object_rot = rotations.mat2euler(self.sim.data.get_site_xmat('object0'))
    # velocities
    object_velp = self.sim.data.get_site_xvelp('object0') * dt
    object_velr = self.sim.data.get_site_xvelr('object0') * dt
    # relative states
    object_rel_pos = object_pos - grip_pos
    object_rel_vel = object_velp - grip_velp

    achieved_goal = np.squeeze(object_pos)

    return {
      'observation': np.zeros(0), # hack to get around gym's GoalEnv checks
      'gripper': np.concatenate([grip_pos, gripper_state, grip_velp, gripper_vel]),
      'object': np.concatenate([object_pos, object_rot, object_velp, object_velr]),
      'relative': np.concatenate([object_rel_pos, object_rel_vel]),
      'achieved_goal': achieved_goal,
      'desired_goal': self.goal.copy(),
    }

  def achieved_goal(self, observation):
    obj = observation['object']
    if len(obj.shape) == 1:
      return obj[:3]
    return obj[:, :3]


class DisentangledFetchSlideEnv(fetch_env.FetchEnv, EzPickle):
  def __init__(self, distance_threshold=0.05, reward_type='sparse'):
    initial_qpos = {
        'robot0:slide0': 0.05,
        'robot0:slide1': 0.48,
        'robot0:slide2': 0.0,
        'object0:joint': [1.7, 1.1, 0.41, 1., 0., 0., 0.],
    }
    assert False, "NEEEDS TO BE REVISED"
    fetch_env.FetchEnv.__init__(self,
                                ORIGSLIDEXML,
                                has_object=True,
                                block_gripper=True,
                                n_substeps=20,
                                gripper_extra_height=-0.02,
                                target_in_the_air=False,
                                target_offset=np.array([0.4, 0.0, 0.0]),
                                obj_range=0.1,
                                target_range=0.3,
                                distance_threshold=distance_threshold,
                                initial_qpos=initial_qpos,
                                reward_type=reward_type)
    EzPickle.__init__(self)

  def _get_obs(self):
    # positions
    grip_pos = self.sim.data.get_site_xpos('robot0:grip')
    dt = self.sim.nsubsteps * self.sim.model.opt.timestep
    grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt
    robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)
    if self.has_object:
      object_pos = self.sim.data.get_site_xpos('object0')
      # rotations
      object_rot = rotations.mat2euler(self.sim.data.get_site_xmat('object0'))
      # velocities
      object_velp = self.sim.data.get_site_xvelp('object0') * dt
      object_velr = self.sim.data.get_site_xvelr('object0') * dt
      # gripper state
      object_rel_pos = object_pos - grip_pos
      object_velp -= grip_velp
    else:
      object_pos = object_rot = object_velp = object_velr = object_rel_pos = np.zeros(0)
    gripper_state = robot_qpos[-2:]
    gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric

    if not self.has_object:
      achieved_goal = grip_pos.copy()
    else:
      achieved_goal = np.squeeze(object_pos.copy())

    grip_obs = np.concatenate([
        grip_pos,
        gripper_state,
        grip_velp,
        gripper_vel,
    ])

    obj_obs = np.concatenate([
        object_pos.ravel(),
        object_rot.ravel(),
        object_velp.ravel(),
        object_velr.ravel(),
    ])

    grip_obs_padded = np.concatenate((grip_obs, np.zeros_like(obj_obs)), 0)
    obj_obs_padded = np.concatenate((np.zeros_like(grip_obs), obj_obs), 0)

    return {
        'observation': np.stack((grip_obs_padded, obj_obs_padded), 0),
        'achieved_goal': achieved_goal.copy(),
        'desired_goal': self.goal.copy(),
    }


class DisentangledFetchPickAndPlaceEnv(FetchPickAndPlaceEnv):
  def _get_obs(self):
    assert False, "NEEEDS TO BE REVISED"
    # positions
    grip_pos = self.sim.data.get_site_xpos('robot0:grip')
    dt = self.sim.nsubsteps * self.sim.model.opt.timestep
    grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt
    robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)
    if self.has_object:
      object_pos = self.sim.data.get_site_xpos('object0')
      # rotations
      object_rot = rotations.mat2euler(self.sim.data.get_site_xmat('object0'))
      # velocities
      object_velp = self.sim.data.get_site_xvelp('object0') * dt
      object_velr = self.sim.data.get_site_xvelr('object0') * dt
      # gripper state
      object_rel_pos = object_pos - grip_pos
      object_velp -= grip_velp
    else:
      object_pos = object_rot = object_velp = object_velr = object_rel_pos = np.zeros(0)
    gripper_state = robot_qpos[-2:]
    gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric

    if not self.has_object:
      achieved_goal = grip_pos.copy()
    else:
      achieved_goal = np.squeeze(object_pos.copy())

    grip_obs = np.concatenate([
        grip_pos,
        gripper_state,
        grip_velp,
        gripper_vel,
    ])

    obj_obs = np.concatenate([
        object_pos.ravel(),
        object_rot.ravel(),
        object_velp.ravel(),
        object_velr.ravel(),
    ])

    grip_obs_padded = np.concatenate((grip_obs, np.zeros_like(obj_obs)), 0)
    obj_obs_padded = np.concatenate((np.zeros_like(grip_obs), obj_obs), 0)

    return {
        'observation': np.stack((grip_obs_padded, obj_obs_padded), 0),
        'achieved_goal': achieved_goal.copy(),
        'desired_goal': self.goal.copy(),
    }


class SlideNEnv(fetch_env.FetchEnv, EzPickle):
  def __init__(self,
               n=1,
               distance_threshold=0.075,
               **kwargs):
    self.n = n
    self.disentangled_idxs = [np.arange(10)] + [10 + 12*i + np.arange(12) for i in range(n)]
    self.disentangled_goal_idxs = [3*i + np.arange(3) for i in range(n)]
    self.ag_dims = np.concatenate([a[:3] for a in self.disentangled_idxs[1:]])
    if not distance_threshold > 1e-5:
      distance_threshold = 0.075 # default

    initial_qpos = {
        'robot0:slide0': 0.05,
        'robot0:slide1': 0.48,
        'robot0:slide2': 0.0,
    }
    for i in range(self.n):
      k = 'object{}:joint'.format(i)
      initial_qpos[k] = INIT_Q_POSES_SLIDE[i]


    fetch_env.FetchEnv.__init__(self,
                                SLIDE_N_XML.replace('#', '{}'.format(n)),
                                has_object=True,
                                block_gripper=True,
                                n_substeps=20,
                                gripper_extra_height=0.,
                                target_in_the_air=False,
                                target_offset=np.array([-0.075, 0.0, 0.0]),
                                obj_range=0.15,
                                target_range=0.30,
                                distance_threshold=distance_threshold,
                                initial_qpos=initial_qpos,
                                reward_type='sparse')
    EzPickle.__init__(self)

    self.max_step = 50 + 25 * (n - 1)
    self.num_step = 0


  def reset(self):
    obs = super().reset()
    self.num_step = 0
    return obs

  def _reset_sim(self):
    self.sim.set_state(self.initial_state)

    # Only a little randomize about the start state
    # for i in range(self.n):
    #   object_qpos = self.sim.data.get_joint_qpos('object{}:joint'.format(i))
    #   object_qpos[:2] += self.np_random.uniform(-0.03, 0.03, size=2)
    #   self.sim.data.set_joint_qpos('object{}:joint'.format(i), object_qpos)

    bad_poses = [self.initial_gripper_xpos[:2]]
    # Randomize start positions of pucks.
    for i in range(self.n):
      object_xpos = self.initial_gripper_xpos[:2]
      while min([np.linalg.norm(object_xpos - p) for p in bad_poses]) < 0.08:
          object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
      bad_poses.append(object_xpos)

      object_qpos = self.sim.data.get_joint_qpos('object{}:joint'.format(i))
      object_qvel = self.sim.data.get_joint_qvel('object{}:joint'.format(i))
      object_qpos[:2] = object_xpos
      object_qpos[2:] = np.array([0.42, 1., 0., 0., 0.])
      self.sim.data.set_joint_qpos('object{}:joint'.format(i), object_qpos)
      self.sim.data.set_joint_qvel('object{}:joint'.format(i), np.zeros_like(object_qvel))

    self.sim.forward()
    return True

  def _sample_goal(self):
    first_puck = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.target_range, self.target_range, size=2)

    goal_xys = [first_puck[:2]]
    for i in range(self.n - 1):
      object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.target_range, self.target_range, size=2)
      while min([np.linalg.norm(object_xpos - p) for p in goal_xys]) < 0.08:
        object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.target_range, self.target_range, size=2)
      goal_xys.append(object_xpos)

    goals = [np.concatenate((goal, [self.height_offset])) for goal in goal_xys]

    return np.concatenate(goals)

  def _env_setup(self, initial_qpos):
      for name, value in initial_qpos.items():
          self.sim.data.set_joint_qpos(name, value)
      utils.reset_mocap_welds(self.sim)
      self.sim.forward()

      # Move end effector into position.
      gripper_target = np.array([-0.548, 0.005, -0.431 + self.gripper_extra_height]) + self.sim.data.get_site_xpos('robot0:grip')
      gripper_rotation = np.array([1., 0., 1., 0.])
      self.sim.data.set_mocap_pos('robot0:mocap', gripper_target)
      self.sim.data.set_mocap_quat('robot0:mocap', gripper_rotation)
      for _ in range(10):
          self.sim.step()

      # Extract information for sampling goals.
      self.initial_gripper_xpos = self.sim.data.get_site_xpos('robot0:grip').copy()
      if self.has_object:
          self.height_offset = self.sim.data.get_site_xpos('object0')[2]
          
  def step(self, action):
    obs, reward, _, info = super().step(action)
    self.num_step += 1
    done = True if self.num_step >= self.max_step else False
    if done: info['TimeLimit.truncated'] = True

    info['is_success'] = np.allclose(reward, 0.)
    return obs, reward, done, info

  def _render_callback(self):
    # Visualize target.
    sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos)
    goals = np.split(self.goal, self.n)

    for i in range(self.n):
      site_id = self.sim.model.site_name2id('target{}'.format(i))
      self.sim.model.site_pos[site_id] = goals[i] - sites_offset[i]
    self.sim.forward()

  def compute_reward(self, achieved_goal, goal, info):
    # Compute distance between goal and the achieved goal.

    if len(achieved_goal.shape) == 1:
      actual_goals = np.split(goal, self.n)
      achieved_goals = np.split(achieved_goal, self.n)
      success = 1.
    else:
      actual_goals = np.split(goal, self.n, axis=1)
      achieved_goals = np.split(achieved_goal, self.n, axis=1)
      success = np.ones(achieved_goal.shape[0], dtype=np.float32)

    for b, g in zip(achieved_goals, actual_goals):
      d = goal_distance(b, g)
      success *= (d <= self.distance_threshold).astype(np.float32)

    return success - 1.  

  def _get_obs(self):
    # positions
    grip_pos = self.sim.data.get_site_xpos('robot0:grip')
    dt = self.sim.nsubsteps * self.sim.model.opt.timestep
    grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt
    robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)

    obj_feats = []
    obj_poses = []

    for i in range(self.n):
      obj_labl = 'object{}'.format(i)
      object_pos = self.sim.data.get_site_xpos(obj_labl).ravel()
      object_pos[2] = max(object_pos[2], self.height_offset)
      # rotations
      object_rot = rotations.mat2euler(self.sim.data.get_site_xmat(obj_labl)).ravel()
      # velocities
      object_velp = (self.sim.data.get_site_xvelp(obj_labl) * dt).ravel()
      object_velr = (self.sim.data.get_site_xvelr(obj_labl) * dt).ravel()
      # gripper state
      object_rel_pos = object_pos - grip_pos
      #object_velp -= grip_velp

      obj_feats.append([
        object_pos.ravel(),
        object_rot.ravel(),
        object_velp.ravel(),
        object_velr.ravel(),
      ])
      obj_poses.append(object_pos)

    gripper_state = robot_qpos[-2:]
    gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric

    achieved_goal = np.concatenate(obj_poses)

    grip_obs = np.concatenate([
        grip_pos,
        gripper_state,
        grip_velp,
        gripper_vel,
    ])

    obs = np.concatenate([grip_pos, gripper_state, grip_velp, gripper_vel] + sum(obj_feats, []))

    return {
        'observation': obs,
        'achieved_goal': achieved_goal,
        'desired_goal': self.goal.copy(),
    }


class PushNEnv(fetch_env.FetchEnv, EzPickle):
  def __init__(self,
               n=1,
               distance_threshold=0.05,
               **kwargs):
    self.n = n
    self.disentangled_idxs = [np.arange(10)] + [10 + 12*i + np.arange(12) for i in range(n)]
    self.ag_dims = np.concatenate([a[:3] for a in self.disentangled_idxs[1:]])
    if not distance_threshold > 1e-5:
      distance_threshold = 0.05 # default

    initial_qpos = {
        'robot0:slide0': 0.05,
        'robot0:slide1': 0.48,
        'robot0:slide2': 0.0,
    }
    for i in range(self.n):
      k = 'object{}:joint'.format(i)
      initial_qpos[k] = INIT_Q_POSES_SLIDE[i]


    fetch_env.FetchEnv.__init__(self,
                                PUSH_N_XML.replace('#', '{}'.format(n)),
                                has_object=True,
                                block_gripper=True,
                                n_substeps=20,
                                gripper_extra_height=0.,
                                target_in_the_air=False,
                                target_offset=np.array([-0.075, 0.0, 0.0]),
                                obj_range=0.15,
                                target_range=0.25,
                                distance_threshold=distance_threshold,
                                initial_qpos=initial_qpos,
                                reward_type='sparse')
    EzPickle.__init__(self)

    self.max_step = 50 + 25 * (n - 1)
    self.num_step = 0


  def reset(self):
    obs = super().reset()
    self.num_step = 0
    return obs

  def _reset_sim(self):
    self.sim.set_state(self.initial_state)

    # Only a little randomize about the start state
    # for i in range(self.n):
    #   object_qpos = self.sim.data.get_joint_qpos('object{}:joint'.format(i))
    #   object_qpos[:2] += self.np_random.uniform(-0.03, 0.03, size=2)
    #   self.sim.data.set_joint_qpos('object{}:joint'.format(i), object_qpos)

    bad_poses = [self.initial_gripper_xpos[:2]]
    # Randomize start positions of pucks.
    for i in range(self.n):
      object_xpos = self.initial_gripper_xpos[:2]
      while min([np.linalg.norm(object_xpos - p) for p in bad_poses]) < 0.08:
          object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
      bad_poses.append(object_xpos)

      object_qpos = self.sim.data.get_joint_qpos('object{}:joint'.format(i))
      object_qvel = self.sim.data.get_joint_qvel('object{}:joint'.format(i))
      object_qpos[:2] = object_xpos
      object_qpos[2:] = np.array([0.42, 1., 0., 0., 0.])
      self.sim.data.set_joint_qpos('object{}:joint'.format(i), object_qpos)
      self.sim.data.set_joint_qvel('object{}:joint'.format(i), np.zeros_like(object_qvel))

    self.sim.forward()
    return True

  def _sample_goal(self):
    first_puck = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.target_range, self.target_range, size=2)

    goal_xys = [first_puck[:2]]
    for i in range(self.n - 1):
      object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.target_range, self.target_range, size=2)
      while min([np.linalg.norm(object_xpos - p) for p in goal_xys]) < 0.08:
        object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.target_range, self.target_range, size=2)
      goal_xys.append(object_xpos)

    goals = [np.concatenate((goal, [self.height_offset])) for goal in goal_xys]

    return np.concatenate(goals)

  def _env_setup(self, initial_qpos):
      for name, value in initial_qpos.items():
          self.sim.data.set_joint_qpos(name, value)
      utils.reset_mocap_welds(self.sim)
      self.sim.forward()

      # Move end effector into position.
      gripper_target = np.array([-0.548, 0.005, -0.431 + self.gripper_extra_height]) + self.sim.data.get_site_xpos('robot0:grip')
      gripper_rotation = np.array([1., 0., 1., 0.])
      self.sim.data.set_mocap_pos('robot0:mocap', gripper_target)
      self.sim.data.set_mocap_quat('robot0:mocap', gripper_rotation)
      for _ in range(10):
          self.sim.step()

      # Extract information for sampling goals.
      self.initial_gripper_xpos = self.sim.data.get_site_xpos('robot0:grip').copy()
      if self.has_object:
          self.height_offset = self.sim.data.get_site_xpos('object0')[2]
          
  def step(self, action):
    obs, reward, _, info = super().step(action)
    self.num_step += 1
    done = True if self.num_step >= self.max_step else False
    if done: info['TimeLimit.truncated'] = True

    info['is_success'] = np.allclose(reward, 0.)
    return obs, reward, done, info

  def _render_callback(self):
    # Visualize target.
    sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos)
    goals = np.split(self.goal, self.n)

    for i in range(self.n):
      site_id = self.sim.model.site_name2id('target{}'.format(i))
      self.sim.model.site_pos[site_id] = goals[i] - sites_offset[i]
    self.sim.forward()

  def compute_reward(self, achieved_goal, goal, info):
    # Compute distance between goal and the achieved goal.

    if len(achieved_goal.shape) == 1:
      actual_goals = np.split(goal, self.n)
      achieved_goals = np.split(achieved_goal, self.n)
      success = 1.
    else:
      actual_goals = np.split(goal, self.n, axis=1)
      achieved_goals = np.split(achieved_goal, self.n, axis=1)
      success = np.ones(achieved_goal.shape[0], dtype=np.float32)

    for b, g in zip(achieved_goals, actual_goals):
      d = goal_distance(b, g)
      success *= (d <= self.distance_threshold).astype(np.float32)

    return success - 1.  

  def _get_obs(self):
    # positions
    grip_pos = self.sim.data.get_site_xpos('robot0:grip')
    dt = self.sim.nsubsteps * self.sim.model.opt.timestep
    grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt
    robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)

    obj_feats = []
    obj_poses = []

    for i in range(self.n):
      obj_labl = 'object{}'.format(i)
      object_pos = self.sim.data.get_site_xpos(obj_labl).ravel()
      object_pos[2] = max(object_pos[2], self.height_offset)
      # rotations
      object_rot = rotations.mat2euler(self.sim.data.get_site_xmat(obj_labl)).ravel()
      # velocities
      object_velp = (self.sim.data.get_site_xvelp(obj_labl) * dt).ravel()
      object_velr = (self.sim.data.get_site_xvelr(obj_labl) * dt).ravel()
      # gripper state
      object_rel_pos = object_pos - grip_pos
      #object_velp -= grip_velp

      obj_feats.append([
        object_pos.ravel(),
        object_rot.ravel(),
        object_velp.ravel(),
        object_velr.ravel(),
      ])
      obj_poses.append(object_pos)

    gripper_state = robot_qpos[-2:]
    gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric

    achieved_goal = np.concatenate(obj_poses)

    grip_obs = np.concatenate([
        grip_pos,
        gripper_state,
        grip_velp,
        gripper_vel,
    ])

    obs = np.concatenate([grip_pos, gripper_state, grip_velp, gripper_vel] + sum(obj_feats, []))

    return {
        'observation': obs,
        'achieved_goal': achieved_goal,
        'desired_goal': self.goal.copy(),
    }




class FetchHookSweepAllEnv(fetch_env.FetchEnv):
  """
    Mujoco env for robotics sweeping and pushing with a hook object.
    Configurable for multible variants of difficulty.

    raw_state overrides smaller state with native mjsim representation
    """
  def __init__(self, xml_file=None, place_two=False, place_random=False, goal_in_air=False, smaller_state=False, raw_state=False):
    initial_qpos = {
        'robot0:slide0': 0.405,
        'robot0:slide1': 0.48,
        'robot0:slide2': 0.0,
        'object0:joint': [1.85, 0.75, 0.4, 1., 0., 0., 0.],
        'object1:joint': [1.85, 0.75, 0.4, 1., 0., 0., 0.],
    }

    self.max_step = 75
    self.num_step = 0

    if xml_file is None:
      #Go 3 folders up to base of rl_with_teachers dir
      package_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
      xml_file = os.path.join(dir_path, 'xmls', 'sweep_all.xml')

    self.JOINTS = ('robot0:slide0', 'robot0:slide1', 'robot0:slide2', 'robot0:torso_lift_joint', 
          'robot0:head_pan_joint', 'robot0:head_tilt_joint', 'robot0:shoulder_pan_joint', 
          'robot0:shoulder_lift_joint', 'robot0:upperarm_roll_joint', 'robot0:elbow_flex_joint', 
          'robot0:forearm_roll_joint', 'robot0:wrist_flex_joint', 'robot0:wrist_roll_joint', 
          'object0:joint', 'object1:joint')

    self.place_two = place_two
    self.place_random = place_random
    self.goal_in_air = goal_in_air
    self.smaller_state = smaller_state
    self.raw_state = raw_state

    # base goal position, from which all other are derived
    self._base_goal_pos = np.array([1.85, 0.7, 0.42, 1.85, 0.8, 0.42])
    if self.goal_in_air:
      self._base_goal_pos = np.array([1.85, 0.7, 0.6, 1.85, 0.8, 0.42])
    self._goal_pos = self._base_goal_pos

    fetch_env.FetchEnv.__init__(self,
                                xml_file,
                                has_object=True,
                                block_gripper=False,
                                n_substeps=20,
                                gripper_extra_height=0.2,
                                target_in_the_air=True,
                                target_offset=0.0,
                                obj_range=None,
                                target_range=None,
                                distance_threshold=0.14,
                                initial_qpos=initial_qpos,
                                reward_type='sparse')

    self._goal_pos = self._sample_goal()
    obs = self._get_obs()
    self.observation_space = spaces.Dict(
        dict(observation=spaces.Box(-np.inf, np.inf, shape=obs['observation'].shape, dtype='float32'),
             achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs['desired_goal'].shape, dtype='float32'),
             desired_goal=spaces.Box(-np.inf, np.inf, shape=obs['desired_goal'].shape, dtype='float32')))

  def _viewer_setup(self):
    body_id = self.sim.model.body_name2id('robot0:gripper_link')
    lookat = self.sim.data.body_xpos[body_id]
    for idx, value in enumerate(lookat):
      self.viewer.cam.lookat[idx] = value
    self.viewer.cam.distance = 2.5
    self.viewer.cam.azimuth = 180.
    self.viewer.cam.elevation = -24.
    
  def render(self, mode="human", *args, **kwargs):
    # See https://github.com/openai/gym/issues/1081
    self._render_callback()
    if mode == 'rgb_array':
      self._get_viewer(mode='human').render()
      width, height = 3350, 1800
      data = self._get_viewer(mode='human').read_pixels(width, height, depth=False)
      # original image is upside-down, so flip it
      return data[::-1, :, :]
    elif mode == 'human':
      self._get_viewer(mode='human').render()

    return super().render(*args, **kwargs)


  def _render_callback(self):
    # Visualize target.
    sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos)
    goals = np.split(self.goal, 2)

    for i in range(2):
      site_id = self.sim.model.site_name2id('target{}'.format(i))
      self.sim.model.site_pos[site_id] = goals[i] - sites_offset[i]
    self.sim.forward()

  def _sample_goal(self):
    goal_pos = self._base_goal_pos.copy()

    adj = self.np_random.random(size=(3, )) * np.array([0.2, 0.2, 0.]) - np.array([0.1, 0.1, 0])
    if self.np_random.random() < 0.5:
      adj[0] += 0.31
    else:
      adj[0] -= 0.31

    if self.place_two:
      goal_pos[:3] += adj
      goal_pos[3:] += adj
    elif self.place_random:
      goal_pos[:3] += self.np_random.random(size=(3, )) * np.array([0.8, 0.8, 0.]) - np.array([0.4, 0.4, 0])
      goal_pos[3:] += self.np_random.random(size=(3, )) * np.array([0.8, 0.8, 0.]) - np.array([0.4, 0.4, 0])
    else:
      if self.np_random.random() < 0.5:
        goal_pos[:3] += adj
      else:
        goal_pos[3:] += adj

    return goal_pos

  def _reset_sim(self):
    self.sim.set_state(self.initial_state)

    while True:
      deltas = []
      for o in range(2):
        delta = self.np_random.uniform(-0.08, 0.08, (2, ))
        deltas.append(delta)
      if np.linalg.norm(deltas[0] - deltas[1]) > 0.1:
        break

    if deltas[0][1] > deltas[1][1]:
      t = deltas[0][1]
      deltas[0][1] = deltas[1][1]
      deltas[1][1] = t
    for o in range(2):
      object_qpos = self.sim.data.get_joint_qpos(f'object{o}:joint')
      assert object_qpos.shape == (7, )
      object_qpos[:2] += deltas[o]
      self.sim.data.set_joint_qpos(f'object{o}:joint', object_qpos)

    self.sim.forward()
    return True

  def _get_obs(self, force_original=False):

    if self.raw_state and not force_original:
      
      obj_poses = []

      for i in range(2):
        obj_labl = 'object{}'.format(i)
        obj_poses.append(self.sim.data.get_site_xpos(obj_labl))

      return {
        'observation': self.sim.get_state().flatten(),
        'achieved_goal': np.concatenate(obj_poses),
        'desired_goal': self.goal.copy(),
      }

    # positions
    grip_pos = self.sim.data.get_site_xpos('robot0:grip')
    dt = self.sim.nsubsteps * self.sim.model.opt.timestep
    grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt
    robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)

    obj_feats = []
    obj_poses = []

    for i in range(2):
      obj_labl = 'object{}'.format(i)
      object_pos = self.sim.data.get_site_xpos(obj_labl)
      # rotations
      object_rot = rotations.mat2euler(self.sim.data.get_site_xmat(obj_labl))
      # velocities
      object_velp = (self.sim.data.get_site_xvelp(obj_labl) * dt)
      object_velr = (self.sim.data.get_site_xvelr(obj_labl) * dt)
      # gripper state

      if self.smaller_state:
        obj_feats.append([
          object_pos,
          object_velp[:2],
        ])
      else:
        obj_feats.append([
          object_pos,
          object_rot,
          object_velp,
          object_velr,
        ])
      obj_poses.append(object_pos)

    gripper_state = robot_qpos[-2:]
    gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric
    achieved_goal = np.concatenate(obj_poses)

    if self.smaller_state:
      obs = np.concatenate([grip_pos, grip_velp] + sum(obj_feats, []))
    else:
      obs = np.concatenate([grip_pos, gripper_state, grip_velp, gripper_vel] + sum(obj_feats, []))

    return {
        'observation': obs.copy(),
        'achieved_goal': achieved_goal.copy(),
        'desired_goal': self.goal.copy(),
    }


  def compute_reward(self, achieved_goal, goal, info):
    # Compute distance between goal and the achieved goal.

    if len(achieved_goal.shape) == 1:
      actual_goals = np.split(goal, 2)
      achieved_goals = np.split(achieved_goal, 2)
      success = 1.
      on_table_bonus = np.clip(0.5 - np.abs(info['s'][2] - 0.42)*5., 0., 0.5)
    else:
      actual_goals = np.split(goal, 2, axis=1)
      achieved_goals = np.split(achieved_goal, 2, axis=1)
      success = np.ones(achieved_goal.shape[0], dtype=np.float32)
      on_table_bonus = np.clip(0.5 - np.abs(info['s'][:,2] - 0.42)*5., 0., 0.5)

    farness_penalty = 0
    for b, g in zip(achieved_goals, actual_goals):
      d = goal_distance(b, g)
      success *= (d <= self.distance_threshold).astype(np.float32)
      farness_penalty += np.clip(d / self.distance_threshold, 0., 1.)*0.1

    on_table_bonus *= (1. - success)

    return np.clip(success - 1. - farness_penalty + on_table_bonus, -1., 0.)  

  def step(self, action):
    action = np.clip(action, self.action_space.low, self.action_space.high)
    self._set_action(action)
    self.sim.step()
    self._step_callback()
    obs = self._get_obs()

    done = False
    info = {
        's': obs['observation'], 
        'a':action,
        'is_success': self._is_success(obs['achieved_goal'], self.goal),
    }
    reward = self.compute_reward(obs['achieved_goal'], self.goal, info)

    self.num_step += 1
    done = True if self.num_step >= self.max_step else False
    if done: info['TimeLimit.truncated'] = True

    info['is_success'] = np.abs(reward) < 0.3

    return obs, reward, done, info

  def reset(self):
    self._goal_pos = self._sample_goal()
    self.num_step = 0
    obs = super().reset()
    return obs

  def _env_setup(self, initial_qpos):
    for name, value in initial_qpos.items():
      self.sim.data.set_joint_qpos(name, value)
    utils.reset_mocap_welds(self.sim)
    self.sim.forward()

    gripper_target = np.array([-0.498, 0.005, -0.431 + self.gripper_extra_height
                                ]) + self.sim.data.get_site_xpos('robot0:grip')
    gripper_rotation = np.array([1., 0., 1., 0.])
    self.sim.data.set_mocap_pos('robot0:mocap', gripper_target)
    self.sim.data.set_mocap_quat('robot0:mocap', gripper_rotation)
    for _ in range(10):
      self.sim.step()

