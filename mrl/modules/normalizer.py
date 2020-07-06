import mrl
import pickle
import os
import numpy as np
import torch

class RunningMeanStd(object):
  # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
  def __init__(self, epsilon=1e-4, shape=()):
    self.mean = np.zeros(shape, 'float64')
    self.var = np.ones(shape, 'float64')
    self.count = epsilon

  def update(self, x):
    batch_mean = np.mean(x, axis=0, keepdims=True)
    batch_var = np.var(x, axis=0, keepdims=True)
    batch_count = x.shape[0]
    self.update_from_moments(batch_mean, batch_var, batch_count)

  def update_from_moments(self, batch_mean, batch_var, batch_count):
    delta = batch_mean - self.mean
    tot_count = self.count + batch_count

    new_mean = self.mean + delta * batch_count / tot_count
    m_a = self.var * (self.count)
    m_b = batch_var * (batch_count)
    M2 = m_a + m_b + np.square(delta) * self.count * batch_count / (tot_count)
    new_var = M2 / (tot_count)

    self.mean = new_mean
    self.var = new_var
    self.count = tot_count


class Normalizer(mrl.Module):
  def __init__(self, normalizer):
    super().__init__('state_normalizer', required_agent_modules=[], locals=locals())
    self.normalizer = normalizer
    self.lazy_load = None

  def __call__(self, *args, **kwargs):
    if self.training:
      self.normalizer.read_only = False
    else:
      self.normalizer.read_only = True

    if self.lazy_load is not None:
      self.normalizer(*args, **kwargs)
      self.load(self.lazy_load)
      print("LOADED NORMALIZER")
      self.lazy_load = None

    return self.normalizer(*args, **kwargs)

  def save(self, save_folder):
    if self.normalizer.state_dict() is not None:
      with open(os.path.join(save_folder, 'normalizer.pickle'), 'wb') as f:
        pickle.dump(self.normalizer.state_dict(), f)

  def load(self, save_folder):
    if self.normalizer.state_dict() is not None:
      save_path = os.path.join(save_folder, 'normalizer.pickle')
      if os.path.exists(save_path):
        with open(save_path, 'rb') as f:
          self.normalizer.load_state_dict(pickle.load(f))
      else:
        print('WARNING: No saved normalizer state to load.')
    else:
      self.lazy_load = save_folder


# Below from https://github.com/ShangtongZhang/DeepRL/blob/master/deep_rl/utils/normalizer.py


class BaseNormalizer:
  def __init__(self, read_only=False):
    self.read_only = read_only

  def set_read_only(self):
    self.read_only = True

  def unset_read_only(self):
    self.read_only = False

  def state_dict(self):
    return None

  def load_state_dict(self, _):
    return


class MeanStdNormalizer(BaseNormalizer):
  def __init__(self, read_only=False, clip_before=200.0, clip_after=5.0, epsilon=1e-8):
    BaseNormalizer.__init__(self, read_only)
    self.read_only = read_only
    self.rms = None
    self.clip_before = clip_before
    self.clip_after = clip_after
    self.epsilon = epsilon

  def __call__(self, x, update=True):
    x = np.clip(np.asarray(x), -self.clip_before, self.clip_before)
    if self.rms is None:
      self.rms = RunningMeanStd(shape=(1, ) + x.shape[1:])
    if not self.read_only and update:
      self.rms.update(x)
    return np.clip((x - self.rms.mean) / np.sqrt(self.rms.var + self.epsilon), -self.clip_after, self.clip_after)

  def state_dict(self):
    if self.rms is not None:
      return {'mean': self.rms.mean, 'var': self.rms.var, 'count': self.rms.count}

  def load_state_dict(self, saved):
    self.rms.mean = saved['mean']
    self.rms.var = saved['var']
    self.rms.count = saved['count']


class RescaleNormalizer(BaseNormalizer):
  def __init__(self, coef=1.0):
    BaseNormalizer.__init__(self)
    self.coef = coef

  def __call__(self, x, *unused_args):
    if not isinstance(x, torch.Tensor):
      x = np.asarray(x)
    return self.coef * x


class ImageNormalizer(RescaleNormalizer):
  def __init__(self):
    RescaleNormalizer.__init__(self, 1.0 / 255)


class SignNormalizer(BaseNormalizer):
  def __call__(self, x, *unused_args):
    return np.sign(x)
