import mrl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from experiments.coda.coda_module import CodaBuffer, CodaOldBuffer
from mrl.utils.misc import batch_block_diag, batch_block_diag_many
import os


class RewardModel(mrl.Module):
  """
  A learned reward model for doing CoDA. 
  """
  def __init__(self, model, optimize_every=5, batch_size=2000):
    """
    Args:
      model: MLP that takes (s, a, ns) and regresses vs reward (1 dim)
    """
    super().__init__('reward_model', required_agent_modules=['replay_buffer'], locals=locals())

    self.step = 0
    self.optimize_every = optimize_every
    self.batch_size = batch_size
    self.optimizer = None
    self.model = model

  def _setup(self):
    self.model = self.model.to(self.config.device)
    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-4)

  def _optimize(self):
    config = self.config
    self.step += 1

    if self.step % self.optimize_every == 0 and len(self.replay_buffer):
      sample = self.replay_buffer.buffer.sample(self.batch_size)
      states, actions, rewards, next_states = sample[0], sample[1], sample[2], sample[3]

      sas = np.concatenate((states, actions, next_states), 1)
      sas = self.torch(sas)
      rewards = self.torch(rewards)

      pred = self.model(sas)
      loss = F.mse_loss(rewards, pred)

      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()

      return float(self.numpy(loss))

  def compute_reward(self, s, a, ns):      
    sas = np.concatenate((s, a, ns), 1)
    sas = self.torch(sas)
    return self.numpy(self.model(sas))

  def save(self, save_folder: str):
    path = os.path.join(save_folder, self.module_name + '.pt')
    torch.save({
        'model_state_dict': self.model.state_dict(),
        'opt_state_dict': self.optimizer.state_dict(),
    }, path)

  def load(self, save_folder: str):
    path = os.path.join(save_folder, self.module_name + '.pt')
    if os.path.exists(path):
      checkpoint = torch.load(path)
      self.model.load_state_dict(checkpoint['model_state_dict'])
      self.optimizer.load_state_dict(checkpoint['opt_state_dict'])


class PongClassifierRewardModel(mrl.Module):
  """
  A learned classifier for doing CoDA. 
  """
  def __init__(self, model, optimize_every=1, batch_size=256):
    """
    Args:
      model: MLP that takes (s, a, ns) and does cross entropy loss vs reward (3 dim)
    """
    super().__init__('reward_model', required_agent_modules=['replay_buffer'], locals=locals())

    self.step = 0
    self.optimize_every = optimize_every
    self.batch_size = batch_size
    self.optimizer = None
    self.model = model

  def _setup(self):
    self.model = self.model.to(self.config.device)
    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-4)

  def _optimize(self):
    config = self.config
    self.step += 1

    if self.step % self.optimize_every == 0 and len(self.replay_buffer):
      sample = self.replay_buffer.buffer.sample(self.batch_size)
      states, actions, rewards, next_states = sample[0], sample[1], sample[2], sample[3]

      sas = np.concatenate((states, actions, next_states), 1)
      sas = self.torch(sas)
      rewards = torch.squeeze(torch.round(self.torch(rewards)).to(torch.long) + 1, 1)

      pred = self.model(sas)

      loss = nn.CrossEntropyLoss()(pred, rewards)

      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()

      return float(self.numpy(loss))

  def compute_reward(self, s, a, ns):      
    sas = np.concatenate((s, a, ns), 1)
    sas = self.torch(sas)
    return np.argmax(self.numpy(self.model(sas)), -1)[:,None].astype(np.float32) - 1.

  def save(self, save_folder: str):
    path = os.path.join(save_folder, self.module_name + '.pt')
    torch.save({
        'model_state_dict': self.model.state_dict(),
        'opt_state_dict': self.optimizer.state_dict(),
    }, path)

  def load(self, save_folder: str):
    path = os.path.join(save_folder, self.module_name + '.pt')
    if os.path.exists(path):
      checkpoint = torch.load(path)
      self.model.load_state_dict(checkpoint['model_state_dict'])
      self.optimizer.load_state_dict(checkpoint['opt_state_dict'])


class CodaAttentionBasedMask(mrl.Module):
  """
  A learned attention-based mask for doing CoDA. 
  """
  def __init__(self, model, optimize_every=5, batch_size=2000):
    """
    Args:
      model: SimpleStackedAttention Model
    """
    super().__init__('coda_attention_model', required_agent_modules=['replay_buffer'], locals=locals())

    self.step = 0
    self.optimize_every = optimize_every
    self.batch_size = batch_size
    self.optimizer = None
    self.model = model

  def _setup(self):
    assert isinstance(self.replay_buffer, CodaBuffer) or isinstance(self.replay_buffer, CodaOldBuffer)
    self.model = self.model.to(self.config.device)
    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4, weight_decay=1e-5)

  def _optimize(self):
    config = self.config
    self.step += 1

    if self.step % self.optimize_every == 0 and len(self.replay_buffer) > config.min_experience_to_train_coda_attn:
      sample = self.replay_buffer.buffer.sample(self.batch_size)
      states, actions, next_states = sample[0], sample[1], sample[3]
      # Convert states to the correct format
      if not config.slot_based_state and config.slot_state_dims:
        state_slots = [states[:, s][:,None] for s in config.slot_state_dims]
        states = batch_block_diag_many(*state_slots)
        next_state_slots = [next_states[:, s][:,None] for s in config.slot_state_dims]
        next_states = batch_block_diag_many(*next_state_slots)

      if self.config.slot_action_dims is not None:
        action_slots = [actions[:, s][:,None] for s in config.slot_action_dims]
        actions = batch_block_diag_many(*action_slots)
      elif len(actions.shape) == 2:
        actions = actions[:, None]

      sa = batch_block_diag(states, actions)

      sa = self.torch(sa)
      next_states = self.torch(next_states)

      pred, mask = self.model.forward_with_mask(sa)
      pred = pred[:, :-actions.shape[1]]

      loss = F.mse_loss(next_states, pred)# + torch.sqrt(mask).mean()*1e-5

      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()

      return float(self.numpy(loss))

  def forward(self, states, actions):
    """1-step forward prediction; operates on raw states/actions"""
    # Convert states to the correct format
    config = self.config

    if not config.slot_based_state and config.slot_state_dims:
      state_slots = [states[:, s][:,None] for s in config.slot_state_dims]
      states = batch_block_diag_many(*state_slots)

    if self.config.slot_action_dims is not None:
      action_slots = [actions[:, s][:,None] for s in config.slot_action_dims]
      actions = batch_block_diag_many(*action_slots)
    elif len(actions.shape) == 2:
      actions = actions[:, None]

    sa = batch_block_diag(states, actions)

    sa = self.torch(sa)

    pred, _ = self.model.forward_with_mask(sa)
    
    next_states = self.numpy(pred)

    if not config.slot_based_state and config.slot_state_dims:
      next_states = next_states[:,self.replay_buffer.invert_batch_block_diag_mask[0], self.replay_buffer.invert_batch_block_diag_mask[1]]

    return next_states

  def get_mask(self, s, a, THRESH):
    """Note that states, actions are already reshaped to (batch, num_components, in_features) here."""
    sa = batch_block_diag(s, a)
    sa = self.torch(sa)

     # Mask here will be batch x num_components x num_components, but last column is garbage
    _, mask = self.model.forward_with_mask(sa)
    
    mask[:,:,-a.shape[1]:] = 0 # set action columns to 0
    mask =  self.numpy(mask)
    mask += np.eye(mask.shape[1])[None]
    
    mask = ((mask + mask.transpose(0,2,1)) / 2.)
    
    return (mask > THRESH).astype(np.float32)

  def save(self, save_folder: str):
    path = os.path.join(save_folder, self.module_name + '.pt')
    torch.save({
        'model_state_dict': self.model.state_dict(),
        'opt_state_dict': self.optimizer.state_dict(),
    }, path)

  def load(self, save_folder: str):
    path = os.path.join(save_folder, self.module_name + '.pt')
    if os.path.exists(path):
      checkpoint = torch.load(path)
      self.model.load_state_dict(checkpoint['model_state_dict'])
      self.optimizer.load_state_dict(checkpoint['opt_state_dict'])


class MBPOModel(mrl.Module):
  """
  An ensemble model based on MBPO (https://arxiv.org/abs/1906.08253)
  """
  def __init__(self, model_fn, optimize_every=5, batch_size=2000, ensemble_size=7):
    """
    Args:
      model_fn: Should be a constructor for an MLP that outputs [mean, std], so 2 x output_dims.
    """
    super().__init__('coda_attention_model', required_agent_modules=['replay_buffer'], locals=locals())

    self.step = 0
    self.optimize_every = optimize_every
    self.batch_size = batch_size
    self.optimizer = None
    self.model_fn = model_fn
    self.ensemble_size = ensemble_size

  def _setup(self):
    assert isinstance(self.replay_buffer, CodaBuffer) or isinstance(self.replay_buffer, CodaOldBuffer)

    self.models = [self.model_fn().to(self.config.device) for _ in range(self.ensemble_size)]
    params = sum([list(model.parameters()) for model in self.models], [])
    self.optimizer = torch.optim.Adam(params, lr=1e-3, weight_decay=5e-5)

  def _optimize(self):
    config = self.config
    self.step += 1

    if self.step % self.optimize_every == 0 and len(self.replay_buffer) > config.min_experience_to_train_coda_attn:
      sample = self.replay_buffer.buffer.sample(self.batch_size)
      states, actions, next_states = sample[0], sample[1], sample[3]

      sa = np.concatenate((states, actions), -1)
      sa = self.torch(sa)
      next_states = self.torch(next_states)

      res = torch.stack([m(sa) for m in self.models]) # ensemble_size x batch x 2*output
      preds, log_sigmasqs = torch.chunk(res, 2, 2)
      sigmasqs = F.softplus(log_sigmasqs) + 1e-6

      loss = ((preds - next_states[None])**2 / (2 * sigmasqs) + 0.5 * torch.log(sigmasqs)).mean()

      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()

      return float(self.numpy(loss))

  def forward(self, states, actions):
    """1-step forward prediction; operates on raw states/actions"""
    # Convert states to the correct format
    config = self.config

    sa = self.torch(np.concatenate((states, actions), -1))
    res = torch.stack([m(sa) for m in self.models]) # ensemble_size x batch x 2*output
    preds, log_sigmasqs = torch.chunk(res, 2, 2)
    sigmasqs = F.softplus(log_sigmasqs) + 1e-6
    sigmas = torch.sqrt(sigmasqs)
    next_state_dists = Normal(preds, sigmas)
    next_states = next_state_dists.rsample() # ensemble_size x batch x output
    next_states = [self.numpy(ns) for ns in torch.chunk(next_states, self.ensemble_size, 0)]
    next_states = np.concatenate(next_states).transpose((1, 0, 2))
    next_states = next_states[np.arange(len(next_states)), np.random.randint(self.ensemble_size, size=len(next_states))]
    return next_states

  def get_mask(self, s, a, THRESH):
    raise NotImplementedError

  def save(self, save_folder: str):
    pass

  def load(self, save_folder: str):
    pass


class SimpleMLP(nn.Module):
  """Standard feedforward NN with Relu activations.
  Layers should include both input and output dimensions."""
  def __init__(self, layers):
    super().__init__()
    layer_list = []
    for i, o in zip(layers[:-1], layers[1:]):
      layer_list += [nn.Linear(i, o), nn.ReLU()]
    layer_list = layer_list[:-1]
    self.f = nn.Sequential(*layer_list)

  def forward(self, x):
    return self.f(x)


class SimpleAttn(nn.Module):
  """Embeds input and applies a single softmax based attention on the output.
  Returns either the output, or the output together with the attention mask.
  """
  def __init__(self, embed_layers, attn_layers):
    super().__init__()
    assert (embed_layers[0] == attn_layers[0])

    attn_layers = tuple(attn_layers[:-1]) + (2 * attn_layers[-1], )

    self.embed = SimpleMLP(embed_layers)
    self.KQ = SimpleMLP(attn_layers)

  def forward(self, x):
    return self.forward_with_mask(x)[0]

  def forward_with_mask(self, x):
    embs = self.embed(x)
    K, Q = torch.chunk(self.KQ(x), chunks=2, dim=-1)
    A = F.softmax(Q.bmm(K.transpose(1, 2)), 2)

    output = A.bmm(embs)  # + embs
    return output, A


class MultipleSimpleAttn(nn.Module):
  def __init__(self, embed_layers, attn_layers, num_simple_attn=1):
    super().__init__()
    self.attns = nn.ModuleList([SimpleAttn(embed_layers, attn_layers) for i in range(num_simple_attn)])

  def forward(self, x):
    return self.forward_with_mask(x)[0]

  def forward_with_mask(self, x):
    outputs, masks = [], []
    for attn in self.attns:
      o, m = attn.forward_with_mask(x)
      outputs.append(o)
      masks.append(m)

    outputs = torch.stack(outputs)
    masks = torch.stack(masks)

    return torch.mean(outputs, dim=0), torch.mean(masks, dim=0)


class SimpleStackedAttn(nn.Module):
  """Stacks SimpleAttn blocks to make a `deep attention` model"""
  def __init__(self, in_features, out_features, num_attn_blocks=2, num_hidden_layers=2, num_hidden_units=512, num_heads=1):
    super().__init__()

    layers1 = (in_features,) + (num_hidden_units, ) * (num_hidden_layers - 1)
    layers2 = (num_hidden_units, ) * num_hidden_layers

    blocks = [MultipleSimpleAttn(layers1, layers1, num_simple_attn=num_heads)]
    for block in range(num_attn_blocks - 1):
      blocks.append(MultipleSimpleAttn(layers2, layers2, num_simple_attn=num_heads))

    output_projection = nn.Linear(num_hidden_units, out_features)
    self.f = nn.Sequential(*blocks, output_projection)

    self.in_features = in_features

  def forward(self, x):
    """
    Args:
      x with shape (batch, num_components, in_features)
    Returns:
      y with shape (batch, num_components, out_features)  
    """
    return self.f(x)

  def forward_with_mask(self, x):
    """
    Args:
      x with shape (batch, num_components, in_features)
    Returns:
      y with shape (batch, num_components, out_features)
      mask with shape (batch, num_components, num_components)
    """

    mask = torch.eye(x.size(1), device=x.device)[None].repeat(x.size(0), 1, 1)

    for module in self.f:
      if type(module) in [SimpleAttn, MultipleSimpleAttn]:
        x, m = module.forward_with_mask(x)
        mask = mask.bmm(m.transpose(1, 2))
      else:
        x = module(x)

    return x, mask
