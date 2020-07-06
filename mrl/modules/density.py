"""
Density modules for estimating density of items in the replay buffer (e.g., states / achieved goals).
"""

import mrl
import numpy as np
from sklearn.neighbors import KernelDensity
from scipy.special import entr
from mrl.replays.online_her_buffer import OnlineHERBuffer
from mrl.utils.networks import MLP
import torch
import torch.nn.functional as F
import os
from mrl.utils.realnvp import RealNVP


class RawKernelDensity(mrl.Module):
  """
  A KDE-based density model for raw items in the replay buffer (e.g., states/goals).
  """
  def __init__(self, item, optimize_every=10, samples=10000, kernel='gaussian', bandwidth=0.1, normalize=True, 
    log_entropy=False, tag='', buffer_name='replay_buffer'):

    super().__init__('{}_kde{}'.format(item, tag), required_agent_modules=[buffer_name], locals=locals())

    self.step = 0
    self.item = item
    self.kde = KernelDensity(kernel=kernel, bandwidth=bandwidth)
    self.optimize_every = optimize_every
    self.samples = samples
    self.kernel = kernel
    self.bandwidth = bandwidth
    self.normalize = normalize
    self.kde_sample_mean = 0.
    self.kde_sample_std = 1.
    self.fitted_kde = None
    self.ready = False
    self.log_entropy = log_entropy
    self.buffer_name = buffer_name

  def _setup(self):
    assert isinstance(getattr(self, self.buffer_name), OnlineHERBuffer)

  def _optimize(self, force=False):
    buffer = getattr(self, self.buffer_name).buffer.BUFF['buffer_' + self.item]
    self.step +=1

    if force or (self.step % self.optimize_every == 0 and len(buffer)):
      self.ready = True
      sample_idxs = np.random.randint(len(buffer), size=self.samples)
      kde_samples = buffer.get_batch(sample_idxs)
      #og_kde_samples = kde_samples

      if self.normalize:
        self.kde_sample_mean = np.mean(kde_samples, axis=0, keepdims=True)
        self.kde_sample_std  = np.std(kde_samples, axis=0, keepdims=True) + 1e-4
        kde_samples = (kde_samples - self.kde_sample_mean) / self.kde_sample_std

      #if self.item == 'ag' and hasattr(self, 'ag_interest') and self.ag_interest.ready:
      #  ag_weights = self.ag_interest.evaluate_disinterest(og_kde_samples)
      #  self.fitted_kde = self.kde.fit(kde_samples, sample_weight=ag_weights.flatten())
      #else:
      self.fitted_kde = self.kde.fit(kde_samples)

      # Now also log the entropy
      if self.log_entropy and hasattr(self, 'logger') and self.step % 250 == 0:
        # Scoring samples is a bit expensive, so just use 1000 points
        num_samples = 1000
        s = self.fitted_kde.sample(num_samples)
        entropy = -self.fitted_kde.score(s)/num_samples + np.log(self.kde_sample_std).sum()
        self.logger.add_scalar('Explore/{}_entropy'.format(self.module_name), entropy, log_every=500)

  def evaluate_log_density(self, samples):
    assert self.ready, "ENSURE READY BEFORE EVALUATING LOG DENSITY"
    return self.fitted_kde.score_samples( (samples  - self.kde_sample_mean) / self.kde_sample_std )

  def evaluate_elementwise_entropy(self, samples, beta=0.):
    """ Given an array of samples, compute elementwise function of entropy of the form:

        elem_entropy = - (p(samples) + beta)*log(p(samples) + beta)

    Args:
      samples: 1-D array of size N
      beta: float, offset entropy calculation

    Returns:
      elem_entropy: 1-D array of size N, elementwise entropy with beta offset
    """
    assert self.ready, "ENSURE READY BEFORE EVALUATING ELEMENT-WISE ENTROPY"
    log_px = self.fitted_kde.score_samples( (samples  - self.kde_sample_mean) / self.kde_sample_std )
    px = np.exp(log_px)
    elem_entropy = entr(px + beta)
    return elem_entropy

  def save(self, save_folder):
    self._save_props(['kde', 'kde_sample_mean', 'kde_sample_std', 'fitted_kde', 'ready'], save_folder)

  def load(self, save_folder):
    self._load_props(['kde', 'kde_sample_mean', 'kde_sample_std', 'fitted_kde', 'ready'], save_folder)


class RawJointKernelDensity(mrl.Module):
  """
  A KDE-based density model for joint raw items in the replay buffer (e.g., behaviour and achieved goals).

  Args:
    item: a list of items in the replay buffer to build a joint density over
  """
  def __init__(self, items, optimize_every=10, samples=10000, kernel='gaussian', bandwidth=0.1, normalize=True, log_entropy=False, tag=''):
    super().__init__('{}_kde{}'.format("".join(items), tag), required_agent_modules=['replay_buffer'], locals=locals())

    self.step = 0
    self.items = items
    self.kde = KernelDensity(kernel=kernel, bandwidth=bandwidth)
    self.optimize_every = optimize_every
    self.samples = samples
    self.kernel = kernel
    self.bandwidth = bandwidth
    self.normalize = normalize
    self.kde_sample_mean = 0.
    self.kde_sample_std = 1.
    self.fitted_kde = None
    self.ready = False
    self.log_entropy = log_entropy

  def _setup(self):
    assert isinstance(self.replay_buffer, OnlineHERBuffer)

  def _optimize(self, force=False):
    buffers = []
    for item in self.items:
      buffers.append(self.replay_buffer.buffer.BUFF['buffer_' + item])
    
    self.step +=1

    if force or (self.step % self.optimize_every == 0 and len(buffers[0])):
      self.ready = True
      sample_idxs = np.random.randint(len(buffers[0]), size=self.samples)
      kde_samples = []
      for buffer in buffers:
        kde_samples.append(buffer.get_batch(sample_idxs))
      
      # Concatenate the items
      kde_samples = np.concatenate(kde_samples, axis=-1)

      if self.normalize:
        self.kde_sample_mean = np.mean(kde_samples, axis=0, keepdims=True)
        self.kde_sample_std  = np.std(kde_samples, axis=0, keepdims=True) + 1e-4
        kde_samples = (kde_samples - self.kde_sample_mean) / self.kde_sample_std
      
      self.fitted_kde = self.kde.fit(kde_samples)

      # Now also log the entropy
      if self.log_entropy and hasattr(self, 'logger') and self.step % 250 == 0:
        # Scoring samples is a bit expensive, so just use 1000 points
        num_samples = 1000
        s = self.fitted_kde.sample(num_samples)
        entropy = -self.fitted_kde.score(s)/num_samples + np.log(self.kde_sample_std).prod()
        self.logger.add_scalar('Explore/{}_entropy'.format(self.module_name), entropy, log_every=500)

  def evaluate_log_density(self, samples):
    assert self.ready, "ENSURE READY BEFORE EVALUATING LOG DENSITY"
    return self.fitted_kde.score_samples( (samples  - self.kde_sample_mean) / self.kde_sample_std )

  def evaluate_elementwise_entropy(self, samples, beta=0.):
    """ Given an array of samples, compute elementwise function of entropy of the form:

        elem_entropy = - (p(samples) + beta)*log(p(samples) + beta)

    Args:
      samples: 1-D array of size N
      beta: float, offset entropy calculation

    Returns:
      elem_entropy: 1-D array of size N, elementwise entropy with beta offset
    """
    assert self.ready, "ENSURE READY BEFORE EVALUATING ELEMENT-WISE ENTROPY"
    log_px = self.fitted_kde.score_samples( (samples  - self.kde_sample_mean) / self.kde_sample_std )
    px = np.exp(log_px)
    elem_entropy = entr(px + beta)
    return elem_entropy

  def save(self, save_folder):
    self._save_props(['kde', 'kde_sample_mean', 'kde_sample_std', 'fitted_kde', 'ready'], save_folder)

  def load(self, save_folder):
    self._load_props(['kde', 'kde_sample_mean', 'kde_sample_std', 'fitted_kde', 'ready'], save_folder)


class RandomNetworkDensity(mrl.Module):
  """
  A random network based ``density'' model for raw items in the replay buffer (e.g., states/goals). The ``density'' is in proportion
  to the error of the learning network.
  Based on https://arxiv.org/abs/1810.12894.
  """
  def __init__(self, item, optimize_every=1, batch_size=256, layers=(256, 256)):

    super().__init__('{}_rnd'.format(item), required_agent_modules=['replay_buffer'], locals=locals())

    self.step = 0
    self.item = item
    self.layers = layers
    self.optimize_every = optimize_every
    self.batch_size = batch_size
    self.tgt_net, self.prd_net, self.optimizer = None, None, None
    self.lazy_load = None

  def _setup(self):
    assert isinstance(self.replay_buffer, OnlineHERBuffer)

  def _init_from_sample(self, x):
    input_size = x.shape[-1]
    self.tgt_net = MLP(input_size, output_size = self.layers[-1], hidden_sizes = self.layers[:-1])
    self.prd_net = MLP(input_size, output_size = self.layers[-1], hidden_sizes = self.layers[:-1])
    if self.config.get('device'):
      self.tgt_net = self.tgt_net.to(self.config.device)
      self.prd_net = self.prd_net.to(self.config.device)
    self.optimizer = torch.optim.SGD(self.prd_net.parameters(), lr=0.1, weight_decay=1e-5)

  def evaluate_log_density(self, samples):
    """Not actually log density, just prediction error"""
    assert self.tgt_net is not None, "ENSURE READY BEFORE EVALUATING LOG DENSITY"

    samples = self.torch(samples)
    tgt = self.tgt_net(samples)
    prd = self.prd_net(samples)
    return self.numpy(-torch.mean((prd - tgt)**2, dim=-1, keepdim=True))

  @property
  def ready(self):
    return self.tgt_net is not None

  def _optimize(self, force=False):
    buffer = self.replay_buffer.buffer.BUFF['buffer_' + self.item]
    self.step +=1

    if force or (self.step % self.optimize_every == 0 and len(buffer)):
      sample_idxs = np.random.randint(len(buffer), size=self.batch_size)
      samples = buffer.get_batch(sample_idxs)

      # lazy load the networks if not yet loaded
      if self.tgt_net is None:
        self._init_from_sample(samples)
        if self.lazy_load is not None:
          self.load(self.lazy_load)
          self.lazy_load = None

      samples = self.torch(samples)
      tgt = self.tgt_net(samples)
      prd = self.prd_net(samples)
      loss = F.mse_loss(tgt, prd)

      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()

  def save(self, save_folder : str):
    path = os.path.join(save_folder, self.module_name + '.pt')
    if self.tgt_net is not None:
      torch.save({
        'tgt_state_dict': self.tgt_net.state_dict(),
        'prd_state_dict': self.prd_net.state_dict(),
        'opt_state_dict': self.optimizer.state_dict(),
      }, path)

  def load(self, save_folder : str):
    path = os.path.join(save_folder, self.module_name + '.pt')
    if self.tgt_net is None and os.path.exists(path):
      self.lazy_load = save_folder
    else:
      checkpoint = torch.load(path)
      self.tgt_net.load_state_dict(checkpoint['tgt_state_dict'])
      self.prd_net.load_state_dict(checkpoint['prd_state_dict'])
      self.optimizer.load_state_dict(checkpoint['opt_state_dict'])


class FlowDensity(mrl.Module):
  """
  Flow Density model (in this case Real NVP). Similar structure to random density above
  """
  def __init__(self, item, optimize_every=2, batch_size=1000, lr=1e-3, num_layer_pairs=3, normalize=True):

    super().__init__('{}_flow'.format(item), required_agent_modules=['replay_buffer'], locals=locals())

    self.step = 0
    self.item = item
    self.num_layer_pairs = num_layer_pairs
    self.optimize_every = optimize_every
    self.batch_size = batch_size
    self.lazy_load = None
    self.flow_model = None
    self.dev = None
    self.lr = lr
    self.sample_mean = 0.
    self.sample_std = 1.
    self.normalize= normalize

  def _setup(self):
    assert isinstance(self.replay_buffer, OnlineHERBuffer)

  def _init_from_sample(self, x):
    input_size = x.shape[-1]
    self.input_channel = input_size
    if self.config.get('device'):
      self.dev = self.config.device
    elif self.dev is None:
      self.dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Device=None is fine for default too based on network.py in realNVP
    self.flow_model = RealNVP(input_channel=self.input_channel, lr=self.lr, num_layer_pairs=self.num_layer_pairs, dev=self.dev)

  def evaluate_log_density(self, samples):
    assert self.ready, "ENSURE READY BEFORE EVALUATING LOG DENSITY"
    return self.flow_model.score_samples( (samples - self.sample_mean) / self.sample_std  )

  @property
  def ready(self):
    return self.flow_model is not None

  def _optimize(self, force=False):
    buffer = self.replay_buffer.buffer.BUFF['buffer_' + self.item]
    self.step +=1

    if force or (self.step % self.optimize_every == 0 and len(buffer)):
      sample_idxs = np.random.randint(len(buffer), size=self.batch_size)
      samples = buffer.get_batch(sample_idxs)
      if self.normalize:
        self.sample_mean = np.mean(samples, axis=0, keepdims=True)
        self.sample_std  = np.std(samples, axis=0, keepdims=True) + 1e-4
        samples = (samples - self.sample_mean) / self.sample_std

      # lazy load the model if not yet loaded
      if self.flow_model is None:
        self._init_from_sample(samples)
        if self.lazy_load is not None:
          self.load(self.lazy_load)
          self.lazy_load = None

      samples = self.torch(samples)
      #del self.flow_model
      #self.flow_model = RealNVP(input_channel=self.input_channel, lr=self.lr, num_layer_pairs=self.num_layer_pairs, dev=self.dev)
      self.flow_model.fit(samples, epochs=1)

  def save(self, save_folder : str):
    path = os.path.join(save_folder, self.module_name + '.pt')
    if self.flow_model is not None:
      torch.save({
        'flow_model': self.flow_model,
      }, path)

  def load(self, save_folder : str):
    path = os.path.join(save_folder, self.module_name + '.pt')
    if self.flow_model is None and os.path.exists(path):
      self.lazy_load = save_folder
    else:
      self.flow_model = torch.load(path)


"""
class DisagreementDensity(mrl.Module):
  #TODO
"""