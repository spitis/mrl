"""
Some of this is direct or modified from:
https://github.com/ShangtongZhang/DeepRL/tree/master/deep_rl/network
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, OneHotCategorical
import numpy as np

def layer_init(layer, w_scale=1.0):
  if hasattr(layer, 'weight') and len(layer.weight.shape) > 1:
    nn.init.orthogonal_(layer.weight.data)
    layer.weight.data.mul_(w_scale)
    nn.init.constant_(layer.bias.data, 0)
  return layer

class GELU(nn.Module):
  def forward(self, input):
    return F.gelu(input)


######################################################################################################
# Basic MLP, for non actor/critic things
######################################################################################################

class MLP(nn.Module):
  """Standard feedforward network.
  Args:
    input_size (int): number of input features
    output_size (int): number of output features
    hidden_sizes (int tuple): sizes of hidden layers

    norm: pre-activation module (e.g., nn.LayerNorm)
    activ: activation module (e.g., GELU, nn.ReLU)
    drop_prob: dropout probability to apply between layers (not applied to input)
  """
  def __init__(self, input_size, output_size=1, hidden_sizes=(256, 256), norm=nn.Identity, activ=nn.ReLU, drop_prob=0.):
    super().__init__()
    self.output_size = output_size

    layer_sizes = (input_size, ) + tuple(hidden_sizes) + (output_size, )
    if len(layer_sizes) == 2:
      layers = [nn.Linear(layer_sizes[0], layer_sizes[1], bias=False)]
    else:
      layers = []
      for dim_in, dim_out in zip(layer_sizes[:-1], layer_sizes[1:]):
        layers.append(nn.Linear(dim_in, dim_out))
        if norm not in [None, nn.Identity]:
          layers.append(norm(dim_out))
        layers.append(activ())
        if drop_prob > 0.:
          layers.append(nn.Dropout(p=drop_prob))
      layers = layers[:-(1 + (norm not in [None, nn.Identity]) + (drop_prob > 0))]
      layers = list(map(layer_init, layers))
    self.f = nn.Sequential(*layers)

  def forward(self, x):
    return self.f(x)
    
######################################################################################################
# Bodies. These are feature extractors; should be combined with a Head 
######################################################################################################


class NatureConvBody(nn.Module):
  def __init__(self, in_channels=4, feature_dim=512, norm = nn.Identity, activ = GELU):
    """Default in_channels is 4 because that is how many B/W Atari frames we stack"""
    super(NatureConvBody, self).__init__()
    self.feature_dim = feature_dim
    layers = [
        nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
        norm(32), activ(),
        nn.Conv2d(32, 64, kernel_size=4, stride=2),
        norm(64), activ(),
        nn.Conv2d(64, 64, kernel_size=3, stride=1),
        norm(64), activ(),
        nn.modules.Flatten(),
        nn.Linear(7 * 7 * 64, feature_dim),
        norm(feature_dim), activ(),
    ]
    layers = list(map(layer_init, layers))
    self.f = nn.Sequential(*layers)

  def forward(self, x):
    return self.f(x)


class FCBody(nn.Module):
  def __init__(self, input_size, layer_sizes=(256, 256), norm = nn.Identity, activ = GELU, use_layer_init = True):
    super(FCBody, self).__init__()
    self.feature_dim = layer_sizes[-1]

    layer_sizes = (input_size, ) + tuple(layer_sizes)
    layers = []
    for dim_in, dim_out in zip(layer_sizes[:-1], layer_sizes[1:]):
      layers += [
          nn.Linear(dim_in, dim_out),
          norm(dim_out), activ(),
      ]
    if use_layer_init:
      layers = list(map(layer_init, layers))
    self.f = nn.Sequential(*layers)

  def forward(self, x):
    return self.f(x)


######################################################################################################
# Main networks. Should be initialized with a body 
######################################################################################################


class Actor(nn.Module):
  """Returns batch of (action_dim,) vectors scaled to (-1, 1) (with tanh)"""
  def __init__(self, body : nn.Module, action_dim : int, max_action: float):
    super().__init__()
    self.body = body
    self.fc = layer_init(nn.Linear(self.body.feature_dim, action_dim), w_scale=0.1)
    self.max_action = max_action

  def forward(self, x):
    return self.max_action * torch.tanh(self.fc(self.body(x)))

class StochasticActor(nn.Module):
  """Used by SAC"""
  def __init__(self, body : nn.Module, action_dim : int, max_action: float, log_std_bounds = (-20, 2)):
    super().__init__()
    self.body = body
    self.fc = layer_init(nn.Linear(self.body.feature_dim, action_dim*2), w_scale=0.1)
    self.max_action = max_action
    self.min_log_std, self.max_log_std = log_std_bounds
    self.log2 = np.log(2)

  def forward(self, x):
    mu_and_log_std = self.fc(self.body(x))
    mu, log_std = torch.chunk(mu_and_log_std, 2, -1)

    action = mu
    logp_action = None
    if self.training:
      log_std = torch.clamp(log_std, self.min_log_std, self.max_log_std)
      std = torch.exp(log_std)
      action_distribution = Normal(mu, std)
      action = action_distribution.rsample()
      logp_action = action_distribution.log_prob(action).sum(axis=-1, keepdims=True)
      logp_action -= (2*(self.log2 - action - F.softplus(-2*action))).sum(axis=1, keepdims=True)

    action = torch.tanh(action)
    return self.max_action * action, logp_action

class Critic(nn.Module):
  """
  Can be used for DQN (output_dim = N) or DDPG (output_dim = 1),
  or a DPPG ensemble (output_dim = N), and so on.
  """
  def __init__(self, body : nn.Module, output_dim : int, use_layer_init : bool = True):
    super().__init__()
    self.body = body
    if use_layer_init:
      self.fc = layer_init(nn.Linear(self.body.feature_dim, output_dim), w_scale=0.1)
    else:
      self.fc = nn.Linear(self.body.feature_dim, output_dim)

  def forward(self, *x):
    return self.fc(self.body(torch.cat(x, -1)))

class DuelingNet(nn.Module):
  def __init__(self, action_dim, body):
    super(DuelingNet, self).__init__()
    self.fc_value = nn.Linear(body.feature_dim, 1)
    self.fc_advantage = nn.Linear(body.feature_dim, action_dim)
    self.body = body

  def forward(self, x):
    phi = self.body(x)
    value = self.fc_value(phi)
    advantange = self.fc_advantage(phi)
    q = value.expand_as(advantange) + (advantange - advantange.mean(1, keepdim=True).expand_as(advantange))
    return q


class CategoricalNet(nn.Module):
  """For C51"""

  def __init__(self, action_dim, num_atoms, body):
    super(CategoricalNet, self).__init__()
    self.fc_categorical = nn.Linear(body.feature_dim, action_dim * num_atoms)
    self.action_dim = action_dim
    self.num_atoms = num_atoms
    self.body = body

  def forward(self, x):
    phi = self.body(x)
    pre_prob = self.fc_categorical(phi).view((-1, self.action_dim, self.num_atoms))
    prob = F.softmax(pre_prob, dim=-1)
    log_prob = F.log_softmax(pre_prob, dim=-1)
    return prob, log_prob


class QuantileNet(nn.Module):
  """For QR DQN"""

  def __init__(self, action_dim, num_quantiles, body):
    super(QuantileNet, self).__init__()
    self.fc_quantiles = nn.Linear(body.feature_dim, action_dim * num_quantiles)
    self.action_dim = action_dim
    self.num_quantiles = num_quantiles
    self.body = body

  def forward(self, x):
    phi = self.body(x)
    quantiles = self.fc_quantiles(phi)
    quantiles = quantiles.view((-1, self.action_dim, self.num_quantiles))
    return quantiles

######################################################################################################
# MDN for condition distribution estimation
######################################################################################################

# From https://github.com/tonyduan/mdn/blob/master/mdn/models.py 
class MixtureDensityNetwork(nn.Module):
  """
  Mixture density network.
  [ Bishop, 1994 ]
  Parameters
  ----------
  dim_in: int; dimensionality of the covariates
  dim_out: int; dimensionality of the response variable
  n_components: int; number of components in the mixture model
  """
  def __init__(self, pi_body: nn.Module, normal_body: nn.Module, output_dim: int, n_components: int):
    super().__init__()
    self.pi_network = CategoricalNetwork(pi_body, n_components)
    self.normal_network = MixtureDiagNormalNetwork(normal_body, output_dim, n_components)

  def forward(self, x):
    return self.pi_network(x), self.normal_network(x)

  def loss(self, x, y):
    pi, normal = self.forward(x)
    loglik = normal.log_prob(y.unsqueeze(1).expand_as(normal.loc))
    loglik = torch.sum(loglik, dim=2)
    loss = -torch.logsumexp(torch.log(pi.probs) + loglik, dim=1)
    return loss

class MixtureDiagNormalNetwork(nn.Module):
  def __init__(self, body: nn.Module, output_dim: int, n_components: int):
    super().__init__()
    self.body = body
    self.n_components = n_components
    self.fc = layer_init(nn.Linear(self.body.feature_dim, 2 * output_dim * n_components))

  def forward(self, x):
    params = self.fc(self.body(x))
    mean, sd = torch.split(params, params.shape[1] // 2, dim=1)
    mean = torch.stack(mean.split(mean.shape[1] // self.n_components, 1))
    sd = torch.stack(sd.split(sd.shape[1] // self.n_components, 1))
    return Normal(mean.transpose(0, 1), torch.exp(sd).transpose(0, 1))

class CategoricalNetwork(nn.Module):
  def __init__(self, body: nn.Module, n_components: int):
    super().__init__()
    self.body = body
    self.fc = layer_init(nn.Linear(self.body.feature_dim, n_components))

  def forward(self, x):
    params = self.fc(self.body(x))
    return OneHotCategorical(logits=params)
