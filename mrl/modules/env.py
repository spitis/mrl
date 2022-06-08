import mrl
from typing import Optional, Callable, List, Union
from mrl.utils.misc import AttrDict, set_global_seeds
import numpy as np

import gym

try:
  from baselines.common.atari_wrappers import make_atari, wrap_deepmind
except:
  pass
from mrl.utils.vec_env.subproc_vec_env import SubprocVecEnv
from mrl.utils.vec_env.dummy_vec_env import DummyVecEnv

import time


class EnvModule(mrl.Module):
  """
  Used to wrap state-less environments in an mrl.Module.
  Vectorizes the environment.
  """
  def __init__(
      self,
      env: Union[str, Callable],
      num_envs: int = 1,
      seed: Optional[int] = None,
      name: Optional[str] = None,
      modalities = ['observation'],
      goal_modalities = ['desired_goal'],
      episode_life=True  # for Atari
  ):

    super().__init__(name or 'env', required_agent_modules=[], locals=locals())

    self.num_envs = num_envs

    if seed is None:
      seed = int(time.time())

    if isinstance(env, str):
      sample_env = make_env_by_id(env, seed, 0, episode_life)()
      env_list = [make_env_by_id(env, seed, i, episode_life) for i in range(num_envs)]
    else:
      sample_env = make_env(env, seed, 0)()
      assert isinstance(sample_env, gym.core.Env), "Only gym environments supported for now!"
      env_list = [make_env(env, seed, i) for i in range(num_envs)]

    if num_envs == 1:
      self.env = DummyVecEnv(env_list)
    else:
      self.env = SubprocVecEnv(env_list)
    print('Initializing env!')

    self.render = self.env.render
    self.observation_space = sample_env.observation_space
    self.action_space = sample_env.action_space

    if isinstance(self.action_space, gym.spaces.Discrete):
      self.action_dim = self.action_space.n
      self.max_action = None
    else:
      assert isinstance(self.action_space, gym.spaces.Box), "Only Box/Discrete actions supported for now!"
      self.action_dim = self.action_space.shape[0]
      self.max_action = self.action_space.high[0]
      assert np.allclose(self.action_space.high,
                         -self.action_space.low), "Action high/lows must equal! Several modules rely on this"

    self.goal_env = False
    self.goal_dim = 0

    if isinstance(self.observation_space, gym.spaces.Dict):
      if goal_modalities[0] in self.observation_space.spaces:
        self.goal_env = True
        self.compute_reward = sample_env.compute_reward
        if hasattr(sample_env, 'achieved_goal'):
          self.achieved_goal = sample_env.achieved_goal
        for key in goal_modalities:
          assert key in self.env.observation_space.spaces
          self.goal_dim += int(np.prod(self.env.observation_space[key].shape))
      state_dim = 0
      for key in modalities:
        if key == 'desired_goal': continue
        assert key in self.env.observation_space.spaces
        state_dim += int(np.prod(self.env.observation_space[key].shape))
      self.state_dim = state_dim
    else:
      self.state_dim = int(np.prod(self.env.observation_space.shape))

    self.state = self.env.reset()

  def step(self, action):
    res = self.env.step(action)
    self.state = res[0]
    return res

  def reset(self, indices=None):
    if not indices:
      self.state = self.env.reset()
      return self.state
    else:
      reset_states = self.env.env_method('reset', indices=indices)
      if self.goal_env:
        for i, reset_state in zip(indices, reset_states):
          for key in reset_state:
            self.state[key][i] = reset_state[key]
      else:
        for i, reset_state in zip(indices, reset_states):
          self.state[i] = reset_state
      return self.state


def make_env(env_fn, seed, rank):
  """
  Utility function for multiprocessed env.
  
  :param env_id: (str) the environment ID
  :param num_env: (int) the number of environment you wish to have in subprocesses
  :param seed: (int) the inital seed for RNG
  """
  def _init():
    env = env_fn()
    env.seed(seed + rank)
    env = ReturnAndObsWrapper(env)
    return env

  set_global_seeds(seed)
  return _init


### BELOW is based on https://github.com/ShangtongZhang/DeepRL/blob/master/deep_rl/component/envs.py
### Added a fix for the VecEnv bug in infinite-horizon envs.


# adapted from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/envs.py
def make_env_by_id(env_id, seed, rank, episode_life=True):
  """Used for regular gym environments and Atari Envs"""
  def _init():
    if env_id.startswith("dm"):
      import dm_control2gym
      _, domain, task = env_id.split('-')
      env = dm_control2gym.make(domain_name=domain, task_name=task)
    else:
      env = gym.make(env_id)
    is_atari = hasattr(gym.envs, 'atari') and isinstance(env.unwrapped, gym.envs.atari.atari_env.AtariEnv)
    if is_atari:
      env = make_atari(env_id)
    env.seed(seed + rank)
    if is_atari:
      env = wrap_deepmind(env, episode_life=episode_life, clip_rewards=False, frame_stack=False, scale=False)
      obs_shape = env.observation_space.shape
      if len(obs_shape) == 3:
        env = TransposeImage(env)
      env = FrameStack(env, 4)
    env = ReturnAndObsWrapper(env)
    return env

  set_global_seeds(seed)
  return _init


class ReturnAndObsWrapper(gym.Wrapper):
  def __init__(self, env):
    gym.Wrapper.__init__(self, env)
    self.total_rewards = 0

  def step(self, action):
    obs, reward, done, info = self.env.step(action)
    info = AttrDict(info)
    self.total_rewards += reward
    if done:
      info.done_observation = obs
      info.terminal_state = True
      if info.get('TimeLimit.truncated'):
        info.terminal_state = False
      info.episodic_return = self.total_rewards
      self.total_rewards = 0
    else:
      info.terminal_state = False
      info.episodic_return = None
    return obs, reward, done, info

  def render(self, *args, **kwargs):
    return self.env.render(*args, **kwargs)

  def reset(self):
    return self.env.reset()

  def __getattr__(self, attr):
    return getattr(self.env, attr)


class FirstVisitDoneWrapper(gym.Wrapper):
  """A wrapper for sparse reward goal envs that makes them terminate
  upon achievement"""
  def step(self, action):
    obs, reward, done, info = self.env.step(action)
    if np.allclose(reward, 0.):
      done = True
      info['is_success'] = True
      if info.get('TimeLimit.truncated'):
        del info['TimeLimit.truncated']
    return obs, reward, done, info

  def reset(self):
    return self.env.reset()

  def __getattr__(self, attr):
    return getattr(self.env, attr)


class TransposeImage(gym.ObservationWrapper):
  def __init__(self, env=None):
    super(TransposeImage, self).__init__(env)
    obs_shape = self.observation_space.shape
    self.observation_space = gym.spaces.Box(self.observation_space.low[0, 0, 0],
                                            self.observation_space.high[0, 0, 0],
                                            [obs_shape[2], obs_shape[1], obs_shape[0]],
                                            dtype=self.observation_space.dtype)

  def observation(self, observation):
    return observation.transpose(2, 0, 1)


# The original LayzeFrames doesn't work well
class LazyFrames(object):
  def __init__(self, frames):
    """This object ensures that common frames between the observations are only stored once.
      It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
      buffers.
      This object should only be converted to numpy array before being passed to the model.
      You'd not believe how complex the previous solution was."""
    self._frames = frames

  def __array__(self, dtype=None):
    out = np.concatenate(self._frames, axis=0)
    if dtype is not None:
      out = out.astype(dtype)
    return out

  def __len__(self):
    return len(self.__array__())

  def __getitem__(self, i):
    return self.__array__()[i]


class FrameStack(gym.Wrapper):
  def __init__(self, env, k):
    """Stack k last frames.

    Returns lazy array, which is much more memory efficient.

    See Also
    --------
    baselines.common.atari_wrappers.LazyFrames
    """
    gym.Wrapper.__init__(self, env)
    self.k = k
    self.frames = deque([], maxlen=k)
    shp = env.observation_space.shape
    self.observation_space = spaces.Box(low=0, high=255, shape=(shp[0], shp[1], shp[2] * k), dtype=np.uint8)

  def reset(self):
    ob = self.env.reset()
    for _ in range(self.k):
      self.frames.append(ob)
    return self._get_ob()

  def step(self, action):
    ob, reward, done, info = self.env.step(action)
    self.frames.append(ob)
    return self._get_ob(), reward, done, info

  def _get_ob(self):
    assert len(self.frames) == self.k
    return LazyFrames(list(self.frames))
