import numpy as np
import random
import gym
import torch
from types import LambdaType
from scipy.linalg import block_diag
import argparse
try:
  import tensorflow as tf
except:
  tf = None

def set_global_seeds(seed):
  """
    set the seed for python random, tensorflow, numpy and gym spaces

    :param seed: (int) the seed
    """
  if tf is not None:
    if hasattr(tf.random, 'set_seed'):
      tf.random.set_seed(seed)
    elif hasattr(tf.compat, 'v1'):
      tf.compat.v1.set_random_seed(seed)
    else:
      tf.set_random_seed(seed)
  np.random.seed(seed)
  random.seed(seed)
  # prng was removed in latest gym version
  if hasattr(gym.spaces, 'prng'):
    gym.spaces.prng.seed(seed)


class AttrDict(dict):
  """
    Behaves like a dictionary but additionally has attribute-style access
    for both read and write.
    e.g. x["key"] and x.key are the same,
    e.g. can iterate using:  for k, v in x.items().
    Can sublcass for specific data classes; must call AttrDict's __init__().
    """
  def __init__(self, *args, **kwargs):
    dict.__init__(self, *args, **kwargs)
    self.__dict__ = self

  def copy(self):
    """
        Provides a "deep" copy of all unbroken chains of types AttrDict, but
        shallow copies otherwise, (e.g. numpy arrays are NOT copied).
        """
    return type(self)(**{k: v.copy() if isinstance(v, AttrDict) else v for k, v in self.items()})


class AnnotatedAttrDict(AttrDict):
  """
  This is an AttrDict that accepts tuples of length 2 as values, where the
  second element is an annotation.
  """
  def __init__(self, *args, **kwargs):
    argdict = dict(*args, **kwargs)
    valuedict = {}
    annotationdict = {}
    for k, va in argdict.items():
      if hasattr(va, '__len__') and len(va) == 2 and type(va[1]) == str:
        v, a = va
        valuedict[k] = v
        annotationdict[k] = a
      else:
        valuedict[k] = va
    super().__init__(self, **valuedict)
    self.annotationdict = annotationdict

  def get_annotation(self, key):
    return self.annotationdict.get(key)


def soft_update(target, src, factor):
  with torch.no_grad():
    for target_param, param in zip(target.parameters(), src.parameters()):
      target_param.data.mul_(1.0 - factor)
      target_param.data.add_(factor * param.data)
 
def short_timestamp():
  """Returns string with timestamp"""
  import datetime
  return '{:%m%d%H%M%S}'.format(datetime.datetime.now())


def flatten_state(state, modalities=['observation', 'desired_goal']):
  #TODO: handle image modalities
  if isinstance(state, dict):
    return np.concatenate([state[m] for m in modalities], -1)
  return state


def add_config_args(argparser, config: AnnotatedAttrDict):
  """TODO: Make this add more types of args automatically?  """
  for k, v in config.items():
    try:
      if type(v) in (str, int, float):
        argparser.add_argument('--' + k, default=v, type=type(v), help=config.get_annotation(k))
      elif type(v) == bool:
        argparser.add_argument('--' + k, default=v, type=str2bool, help=config.get_annotation(k))
    except:
      pass
  return argparser


def str2bool(v):
  if isinstance(v, bool):
    return v
  if v.lower() in ('yes', 'true', 't', 'y', '1'):
    return True
  elif v.lower() in ('no', 'false', 'f', 'n', '0'):
    return False
  else:
    raise argparse.ArgumentTypeError('Boolean value expected.')


def merge_args_into_config(args, config: AttrDict):
  config.parent_folder = args.parent_folder

  other_args = {}
  for k, v in args.__dict__.items():
    if k in config:
      config[k] = v
    elif not isinstance(v, LambdaType):
      other_args[k] = v

  config.other_args = other_args

  return config


def make_agent_name(config, attr_list, prefix='agent'):
  agent_name = prefix
  attr_set = set()
  for attr in attr_list:
    s = shorten_attr(attr, attr_set)
    attr_set.add(s)
    if attr in config:
      agent_name += '_' + s + str(config[attr])
    elif attr in config.other_args:
      agent_name += '_' + s + '-' + str(config.other_args[attr])
    else:
      raise ValueError('Attribute {} not found in config!'.format(attr))
  return agent_name


def shorten_attr(attr, set, proposed_len=5):
  short = attr[:proposed_len]
  if short in set:
    return shorten_attr(attr, set, proposed_len + 1)
  return short


def softmax(X, theta=1.0, axis=-1):
  """
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats.
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """

  # make X at least 2d
  y = np.atleast_2d(X)

  # find axis
  if axis is None:
    axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

  # multiply y against the theta parameter,
  y = y * float(theta)

  # subtract the max for numerical stability
  y = y - np.max(y, axis=axis, keepdims=True)

  # exponentiate y
  y = np.exp(y)

  # take the sum along the specified axis
  ax_sum = np.sum(y, axis=axis, keepdims=True)

  # finally: divide elementwise
  p = y / ax_sum

  # flatten if X was 1D
  if len(X.shape) == 1: p = p.flatten()

  return p

def make_activ(activ_name):
  if activ_name.lower() == 'relu':
    return torch.nn.ReLU
  elif activ_name.lower() == 'gelu':
    from mrl.utils.networks import GELU
    return GELU
  elif activ_name.lower() == 'tanh':
    return torch.nn.Tanh
  else:
    raise NotImplementedError


def batch_block_diag(a, b):
  """ 
  This does what scipy.linalg.block_diag does but in batch mode and with only 2 array
  https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.linalg.block_diag.html
  """
  a_shape = a.shape
  b_shape = b.shape

  if len(a_shape) == 2:
    return block_diag(a, b)
  
  assert len(a_shape) == 3
  assert len(b_shape) == 3

  assert a_shape[0] == b_shape[0] # the batch dimension

  res = np.zeros((a_shape[0], a_shape[1] + b_shape[1], a_shape[2] + b_shape[2]))
  res[:,:a_shape[1], :a_shape[2]] = a
  res[:,a_shape[1]:, a_shape[2]:] = b

  return res

def batch_block_diag_many(*arrs):
  shapes = np.array([a.shape for a in arrs], dtype=np.int64) 

  if len(shapes[0]) == 2:
    return block_diag(*arrs)
  
  # shapes is now 2D: num_arrs x 3

  res = np.zeros( (shapes[0][0], shapes[:, 1].sum(), shapes[:,2].sum()) )

  r, c = 0, 0
  for i, (batch, rr, cc) in enumerate(shapes):
      res[:, r:r + rr, c:c + cc] = arrs[i]
      r += rr
      c += cc
  
  return res