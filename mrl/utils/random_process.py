"""
Just using Shantong's code for now, but really this should be implemented in 
Pytorch and done on GPU when it is available (see old code for a TF version)
"""

#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import numpy as np
from mrl.utils.schedule import *


class RandomProcess(object):
  def reset_states(self):
    pass


class GaussianProcess(RandomProcess):
  def __init__(self, size, std=ConstantSchedule(0.2)):
    self.size = size
    self.std = std

  def sample(self):
    return np.random.randn(*self.size) * self.std()


class OrnsteinUhlenbeckProcess(RandomProcess):
  def __init__(self, size, std=ConstantSchedule(0.2), theta=.15, dt=1e-2, x0=None, reset_every=50):
    self.theta = theta
    self.mu = 0
    self.std = std
    self.dt = dt
    self.x0 = x0
    self.size = size
    self.reset_states()
    self.steps = 0
    self.reset_every = reset_every

  def sample(self):
    self.steps += 1
    if self.steps % self.reset_every == 0:
      self.reset_states()
    x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.std() * np.sqrt(
        self.dt) * np.random.randn(*self.size)
    self.x_prev = x
    return x

  def reset_states(self):
    self.x_prev = self.x0 if self.x0 is not None else np.zeros(self.size)
