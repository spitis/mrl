import numpy as np
import robosuite as rs
import gym
import time

from robosuite.environments.sawyer_lift import SawyerLift
from robosuite.utils import MujocoPyRenderer

class GoalBasedSawyerLift(SawyerLift, gym.GoalEnv):
  """
  Wraps SawyerLift in a GoalEnv.
  """
  def __init__(self, test=False, use_dense=True, objgrip=False):
    super().__init__(
      use_camera_obs=False,
      has_renderer=False,
      use_indicator_object=True,
      has_offscreen_renderer=False,
      horizon=50,
      ignore_done=True
    )
    self._viewer = None
    self.action_space = gym.spaces.Box(*self.action_spec)

    self._init_eef = np.array([ 0.58218024, -0.01407946,  0.90169981])

    o = self._get_observation()
    self._obs_keys = ['cube_pos', 'cube_quat', 'eef_pos', 'joint_pos', 'joint_vel', 'gripper_qpos', 'gripper_qvel']
    observation_space = gym.spaces.Box(-np.inf, np.inf, (sum([len(o[k]) for k in self._obs_keys]),))
    if objgrip: 
      self.ag = lambda obs: np.concatenate((obs[:3], obs[7:10]))
    else:
      self.ag = lambda obs: obs[:3]

    if objgrip:
      goal_space = gym.spaces.Box(-np.inf, np.inf, (6,))
    else:
      goal_space = gym.spaces.Box(-np.inf, np.inf, (3,))

    self.observation_space = gym.spaces.Dict({
     'observation': observation_space,
     'desired_goal': goal_space,
     'achieved_goal': goal_space
    })

    self.max_steps = 50
    self.num_steps = 0
    self.dist_threshold = 0.05
    if objgrip:
      self.dist_threshold = 0.07
    self.test = test
    self.use_dense = use_dense
    self.objgrip = objgrip

    self.goal = None
    self._prev_state = None

  def reset(self):
    """
    Need to override reset because the original is awfully inefficient (it reloads Mujoco Sim entirely)
    """
    self.num_steps = 0

    # From base.py
    self.sim.set_state(self.sim_state_initial)
    self.initialize_time(self.control_freq)
    self._get_reference()
    self.cur_time = 0
    self.timestep = 0
    self.done = False

    # From sawyer.py
    self.sim.data.qpos[self._ref_joint_pos_indexes] = self.mujoco_robot.init_qpos

    if self.has_gripper:
        self.sim.data.qpos[
            self._ref_gripper_joint_pos_indexes
        ] = self.gripper.init_qpos

    # From sawyer_lift.py

    # reset positions of objects
    # self.model.place_objects() <- doesn't actually work, because this operates on the model and not the sim...
    init_pos = self.sim.data.get_joint_qpos('cube')
    init_pos[:2] += np.random.uniform(-0.1, 0.1, 2)
    self.sim.data.set_joint_qpos('cube', init_pos)
    
    # reset goal position
    self.goal = init_pos[:3] + np.array([0., 0., 0.12])
    self.move_indicator(self.goal)
    if self.objgrip:
      self.goal = np.concatenate((self.goal, self.goal))

    
    # reset joint positions
    init_pos = np.array([-0.5538, -0.8208, 0.4155, 1.8409, -0.4955, 0.6482, 1.9628])
    init_pos += np.random.randn(init_pos.shape[0]) * 0.02
    self.sim.data.qpos[self._ref_joint_pos_indexes] = np.array(init_pos)
    
    # And again from base.py
    self.sim.forward()
    obs = self._get_observation()
    obs = np.concatenate([obs[k] for k in self._obs_keys])
    ag = self.ag(obs)

    self._prev_state = obs
    obs = {
        'observation': obs,
        'achieved_goal': ag,
        'desired_goal': self.goal,
    }
    return obs

  def seed(self, seed=None):
    np.random.seed(seed)

  def compute_reward(self, ag, dg, info):
    d = np.linalg.norm(ag - dg, axis=-1)
    reward = -(d >= self.dist_threshold).astype(np.float32)

    success = 1. + reward
    failure = -1 * reward

    reward -= d * success

    ns = info['ns']
    if len(ns.shape) == 1:
      reward -= (0.2*np.tanh(np.mean(np.abs(info['s'][7:17] - ns[7:17]), axis=-1)) * success) # Penalize successes that move
    else:
      reward -= (0.2*np.tanh(np.mean(np.abs(info['s'][:,7:17] - ns[:,7:17]), axis=-1)) * success) # Penalize successes that move

    # add dense reward discourages agent from being far away from the cube
    if self.use_dense:
      if len(ns.shape) == 1:
        reward -= np.linalg.norm(ns[:3] - ns[7:10]) * failure
      else:
        reward -= np.linalg.norm(ns[:,:3] - ns[:,7:10], axis=-1) * failure

    return reward

  def is_success(self, ag, dg):
    d = np.linalg.norm(ag[:3] - dg[:3], axis=-1)
    return d <= self.dist_threshold

  def step(self, action):
    obs, _, _, _ = super().step(action)
    obs = np.concatenate([obs[k] for k in self._obs_keys])
    ag = self.ag(obs)
    eef = obs[7:10]

    reward = self.compute_reward(ag, self.goal, {'s':obs, 'ns':self._prev_state})
    self._prev_state = obs
    obs = {
        'observation': obs,
        'achieved_goal': ag,
        'desired_goal': self.goal,
    }


    if not self.test:
      info = {'is_success': self.is_success(ag, self.goal)}
    elif self.test:
      info = {'is_success': self._check_success()}

    self.num_steps += 1
    done = True if self.num_steps >= self.max_steps else False
    if done: info['TimeLimit.truncated'] = True
    
    return obs, reward, done, info



  def render(self):
    """
    Fix Robosuite render method, so that Viewer is only spawned on render
    """
    if self._viewer is None:
      self._viewer = MujocoPyRenderer(self.sim)
      self._viewer.viewer.vopt.geomgroup[0] = (
          1 if self.render_collision_mesh else 0
      )
      self._viewer.viewer.vopt.geomgroup[1] = 1 if self.render_visual_mesh else 0

      # hiding the overlay speeds up rendering significantly
      self._viewer.viewer._hide_overlay = True

      self._viewer.viewer._render_every_frame = True

    time.sleep(1/self.control_freq)
    self._viewer.render()

class SawyerReach(SawyerLift, gym.GoalEnv):
  """
  Wraps SawyerLift with a toy Reaching task.
  """
  def __init__(self):
    super().__init__(
      use_camera_obs=False,
      has_renderer=False,
      use_indicator_object=True,
      has_offscreen_renderer=False,
      horizon=50,
      ignore_done=True
    )
    self._viewer = None
    self.action_space = gym.spaces.Box(*self.action_spec)

    self._init_eef = np.array([ 0.58218024, -0.01407946,  0.90169981])

    o = self._get_observation()
    self._obs_keys = ['eef_pos', 'joint_pos', 'joint_vel', 'gripper_qpos', 'gripper_qvel'] #, 'cube_pos', 'cube_quat']
    observation_space = gym.spaces.Box(-np.inf, np.inf, (sum([len(o[k]) for k in self._obs_keys]),))
    self.ag = lambda obs: obs[:3]
    goal_space = gym.spaces.Box(-np.inf, np.inf, o['eef_pos'].shape)

    self.observation_space = gym.spaces.Dict({
     'observation': observation_space,
     'desired_goal': goal_space,
     'achieved_goal': goal_space
    })

    self.max_steps = 50
    self.num_steps = 0
    self.dist_threshold = 0.05
    self._prev_state = None

  def reset(self):
    """
    Need to override reset because the original is awfully inefficient (it reloads Mujoco Sim entirely)
    """
    self.num_steps = 0

    # From base.py
    self.sim.set_state(self.sim_state_initial)
    self.initialize_time(self.control_freq)
    self._get_reference()
    self.cur_time = 0
    self.timestep = 0
    self.done = False

    # From sawyer.py
    self.sim.data.qpos[self._ref_joint_pos_indexes] = self.mujoco_robot.init_qpos

    if self.has_gripper:
        self.sim.data.qpos[
            self._ref_gripper_joint_pos_indexes
        ] = self.gripper.init_qpos

    # From sawyer_lift.py

    # reset positions of objects
    # self.model.place_objects() <- doesn't actually work, because this operates on the model and not the sim...
    init_pos = self.sim.data.get_joint_qpos('cube')
    init_pos[:2] += np.random.uniform(-0.05, 0.05, 2)
    self.sim.data.set_joint_qpos('cube', init_pos)
    
    # reset joint positions
    init_pos = np.array([-0.5538, -0.8208, 0.4155, 1.8409, -0.4955, 0.6482, 1.9628])
    init_pos += np.random.randn(init_pos.shape[0]) * 0.02
    self.sim.data.qpos[self._ref_joint_pos_indexes] = np.array(init_pos)
    
    # reset goal position
    init_pos = self._init_eef
    proposal = init_pos + np.random.uniform(-0.2, 0.2, 3) + np.array([0., 0., 0.15])
    while np.linalg.norm(init_pos - proposal) < 0.05:
      proposal = init_pos + np.random.uniform(-0.2, 0.2, 3) + np.array([0., 0., 0.15])

    self.goal = proposal
    self.move_indicator(self.goal)

    # And again from base.py
    self.sim.forward()
    obs = self._get_observation()
    obs = np.concatenate([obs[k] for k in self._obs_keys])
    ag = self.ag(obs)


    self._prev_state = obs
    obs = {
        'observation': obs,
        'achieved_goal': ag,
        'desired_goal': self.goal,
    }
    return obs

  def seed(self, seed=None):
    np.random.seed(seed)

  def compute_reward(self, achieved_goal, desired_goal, info):
    d = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
    reward = -(d >= self.dist_threshold).astype(np.float32)

    success = 1. + reward

    reward -= d * success # penalize distance when successful

    if len(achieved_goal.shape) == 1:
      reward -= (0.2*np.tanh(np.mean(np.abs(info['s'][0:10] - info['ns'][0:10]), axis=-1)) * success) # Penalize successes that move
    else:
      reward -= (0.2*np.tanh(np.mean(np.abs(info['s'][:,0:10] - info['ns'][:,0:10]), axis=-1)) * success) # Penalize successes that move
    return reward

  def is_success(self, ag, dg):
    d = np.linalg.norm(ag[:3] - dg[:3], axis=-1)
    return d <= self.dist_threshold

  def step(self, action):
    obs, _, _, _ = super().step(action)
    obs = np.concatenate([obs[k] for k in self._obs_keys])
    ag = self.ag(obs)

    reward = self.compute_reward(ag, self.goal, {'s':self._prev_state, 'ns':obs})
    self._prev_state = obs
    obs = {
        'observation': obs,
        'achieved_goal': ag,
        'desired_goal': self.goal,
    }

    info = {'is_success': self.is_success(ag, self.goal)}
    self.num_steps += 1
    done = True if self.num_steps >= self.max_steps else False
    if done: info['TimeLimit.truncated'] = True
    
    return obs, reward, done, info

  def render(self):
    """
    Fix Robosuite render method, so that Viewer is only spawned on render
    """
    if self._viewer is None:
      self._viewer = MujocoPyRenderer(self.sim)
      self._viewer.viewer.vopt.geomgroup[0] = (
          1 if self.render_collision_mesh else 0
      )
      self._viewer.viewer.vopt.geomgroup[1] = 1 if self.render_visual_mesh else 0

      # hiding the overlay speeds up rendering significantly
      self._viewer.viewer._hide_overlay = True
      self._viewer._render_every_frame = True

    time.sleep(1/self.control_freq)
    self._viewer.render()