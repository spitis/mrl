"""
Success Prediction Module
"""

import mrl
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
import os
from mrl.replays.online_her_buffer import OnlineHERBuffer

class GoalSuccessPredictor(mrl.Module):
  """Predicts success using a learned discriminator"""

  def __init__(self, batch_size = 50, history_length = 200, optimize_every=250, log_every=5000):
    super().__init__(
      'success_predictor',
      required_agent_modules=[
        'env', 'replay_buffer', 'goal_discriminator'
      ],
      locals=locals())
    self.log_every = log_every
    self.batch_size = batch_size
    self.history_length = history_length
    self.optimize_every = optimize_every
    self.opt_steps = 0

  def _setup(self):
    super()._setup()
    assert isinstance(self.replay_buffer, OnlineHERBuffer)
    assert self.env.goal_env
    self.n_envs = self.env.num_envs
    self.optimizer = torch.optim.Adam(self.goal_discriminator.model.parameters())

  def _optimize(self):
    self.opt_steps += 1

    if len(self.replay_buffer.buffer.trajectories) > self.batch_size and self.opt_steps % self.optimize_every == 0:
      trajs = self.replay_buffer.buffer.sample_trajectories(self.batch_size, group_by_buffer=True, from_m_most_recent=self.history_length)
      successes = np.array([np.any(np.isclose(traj, 0.), axis=0) for traj in trajs[2]])

      start_states = np.array([t[0] for t in trajs[0]])
      behav_goals =  np.array([t[0] for t in trajs[7]])
      states = np.concatenate((start_states, behav_goals), -1)

      targets = self.torch(successes)
      inputs = self.torch(states)

      # outputs here have not been passed through sigmoid
      outputs = self.goal_discriminator(inputs)
      loss = F.binary_cross_entropy_with_logits(outputs, targets)

      if hasattr(self, 'logger'):
        self.logger.add_histogram('predictions', torch.sigmoid(outputs), self.log_every)
        self.logger.add_histogram('targets', targets, self.log_every)

      # optimize
      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()

  def __call__(self, *states_and_maybe_goals):
    """Input / output are numpy arrays"""
    states = np.concatenate(states_and_maybe_goals, -1)
    return self.numpy(torch.sigmoid(self.goal_discriminator(self.torch(states))))

  def save(self, save_folder : str):
    path = os.path.join(save_folder, self.module_name + '.pt')
    torch.save({
      'opt_state_dict': self.optimizer.state_dict()
    }, path)

  def load(self, save_folder : str):
    path = os.path.join(save_folder, self.module_name + '.pt')
    checkpoint = torch.load(path)
    self.optimizer.load_state_dict(checkpoint['opt_state_dict'])
