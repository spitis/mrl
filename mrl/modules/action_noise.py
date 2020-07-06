import mrl
from mrl.utils.random_process import *

class ContinuousActionNoise(mrl.Module):
  def __init__(self, random_process_cls = GaussianProcess, *args, **kwargs):
    super().__init__('action_noise', required_agent_modules = ['env'], locals=locals())
    self._random_process_cls = random_process_cls
    self._args = args
    self._kwargs = kwargs

  def _setup(self):
    self.random_process = self._random_process_cls((self.env.num_envs, self.env.action_dim,), *self._args, **self._kwargs)
    self.varied_action_noise = self.config.get('varied_action_noise')

  def __call__(self, action):
    factor = 1
    if self.varied_action_noise:
      n_envs = self.env.num_envs
      factor = np.arange(0., 1. + (1./n_envs), 1./(n_envs-1)).reshape(n_envs, 1)
    
    return action + (self.random_process.sample() * self.env.max_action * factor)[:len(action)]

  def save(self, save_folder):
    self._save_props(['random_process'], save_folder)

  def load(self, save_folder):
    self._load_props(['random_process'], save_folder)