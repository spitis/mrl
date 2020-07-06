import mrl
import numpy as np
import torch, torch.nn.functional as F
import os

class GoalEnvReward(mrl.Module):
  def __init__(self):
    """Wraps environment's compute reward function"""
    super().__init__(
        'goal_reward', required_agent_modules=['env'], locals=locals())

  def _setup(self):
    assert self.env.goal_env, "Environment must be a goal environment!"
    assert hasattr(self.env, 'compute_reward'), "Environment must have compute reward defined!"

  def __call__(self, achieved_goals, goals, info):
    return self.env.compute_reward(achieved_goals, goals, info)


class NeighborReward(mrl.Module):
  def __init__(self, max_neighbor_distance = 1, optimize_every = 5, batch_size = 1000, temperature = 1.):
    """Wraps environment's compute reward function. Should probably only be used for first-visit achievment."""
    super().__init__(
        'goal_reward', required_agent_modules=['replay_buffer', 'neighbor_embedding_network'], locals=locals())
    
    self.step = 0
    self.optimize_every = optimize_every
    self.batch_size = batch_size
    self.temperature = temperature

    if max_neighbor_distance != 1: # this is the number of steps from which to count two goals as neighbors.
      raise NotImplementedError

  def _setup(self):
    assert self.env.goal_env, "Environment must be a goal environment!"
    assert hasattr(self.env, 'compute_reward'), "Environment must have compute reward defined!"

    self.optimizer = torch.optim.Adam(
      self.neighbor_embedding_network.model.parameters(),
      lr=self.config.critic_lr, # just using critic hparams for now
      weight_decay=self.config.critic_weight_decay)

  def _optimize(self):
    pag_buffer = self.replay_buffer.buffer.BUFF.buffer_previous_ag
    ag_buffer = self.replay_buffer.buffer.BUFF.buffer_ag
    self.step +=1

    if self.step % self.optimize_every == 0 and len(ag_buffer):
      sample_idxs = np.random.randint(len(ag_buffer), size=self.batch_size)

      ags = ag_buffer.get_batch(sample_idxs)
      pos = pag_buffer.get_batch(sample_idxs)

      # mix it up to keep it symmetric for now...
      temp = ags[:len(ags) //2].copy()
      ags[:len(ags) //2] = pos[:len(ags) //2]
      pos[:len(ags) //2] = temp

      # get random negative samples by a 1 index roll
      neg = np.roll(pos, 1, axis=0)

      # move to torch
      ags = self.torch(ags)
      pos = self.torch(pos)
      neg = self.torch(neg)

      # get embeddings
      embs = self.neighbor_embedding_network(torch.cat((ags, pos, neg), dim=0))
      ags, pos, neg = torch.chunk(embs, 3)

      pos_logits = -self.temperature * torch.norm(ags - pos, dim = 1)
      neg_logits = -self.temperature * torch.norm(ags - neg, dim = 1)

      # use soft targets
      loss = F.binary_cross_entropy_with_logits(torch.exp(pos_logits), torch.ones_like(pos_logits) * 0.99) +\
             F.binary_cross_entropy_with_logits(torch.exp(neg_logits), torch.ones_like(pos_logits) * 0.01)


      self.logger.add_tabular('intrinsic_reward_loss', self.numpy(loss))

      # optimize
      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()

  def __call__(self, achieved_goals, goals, info):
    """Should return 0 for ags, gs that are predicted to be neighbors, -1 otherwise, as a numpy array"""
    ags = achieved_goals.reshape(-1, achieved_goals.shape[-1])
    dgs = goals.reshape(-1, achieved_goals.shape[-1])

    ags = self.torch(ags)
    dgs = self.torch(dgs)

    # get embeddings
    embs = self.neighbor_embedding_network(torch.cat((ags, dgs), dim=0))
    ags, dgs = torch.chunk(embs, 2)

    # predict whether ags and dgs are transition neighbors
    preds = torch.exp(-self.temperature * torch.norm(ags - dgs, dim = 1))

    return -self.numpy(preds < 0.5).astype(np.float32)

  def save(self, save_folder : str):
    path = os.path.join(save_folder, self.module_name + '.pt')
    torch.save({
      'opt_state_dict': self.optimizer.state_dict()
    }, path)

  def load(self, save_folder : str):
    path = os.path.join(save_folder, self.module_name + '.pt')
    checkpoint = torch.load(path)
    self.optimizer.load_state_dict(checkpoint['opt_state_dict'])


  def load(self, save_folder):
    self._load_props(['random_process'], save_folder)