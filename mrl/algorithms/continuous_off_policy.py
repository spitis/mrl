import mrl
from mrl.utils.misc import AttrDict, soft_update, flatten_state
from mrl.modules.model import PytorchModel

import numpy as np
import torch
import torch.nn.functional as F
import os

class ActorPolicy(mrl.Module):
  """Used for DDPG / TD3 and other deterministic policy variants"""
  def __init__(self):
    super().__init__(
        'policy',
        required_agent_modules=[
            'actor', 'action_noise', 'env', 'replay_buffer'
        ],
        locals=locals())
  
  def _setup(self):
    self.use_actor_target = self.config.get('use_actor_target')

  def __call__(self, state, greedy=False):
    action_scale = self.env.max_action

    # initial exploration and intrinsic curiosity
    res = None
    if self.training:
      if self.config.get('initial_explore') and len(self.replay_buffer) < self.config.initial_explore:
        res = np.array([self.env.action_space.sample() for _ in range(self.env.num_envs)])
      elif hasattr(self, 'ag_curiosity'):
        state = self.ag_curiosity.relabel_state(state)
        
    state = flatten_state(state, self.config.modalities + self.config.goal_modalities)  # flatten goal environments
    if hasattr(self, 'state_normalizer'):
      state = self.state_normalizer(state, update=self.training)

    if res is not None:
      return res

    state = self.torch(state)

    if self.use_actor_target:
      action = self.numpy(self.actor_target(state))
    else:
      action = self.numpy(self.actor(state))

    if self.training and not greedy:
      action = self.action_noise(action)
      if self.config.get('eexplore'):
        eexplore = self.config.eexplore
        if hasattr(self, 'ag_curiosity'):
          eexplore = self.ag_curiosity.go_explore * self.config.go_eexplore + eexplore
        mask = (np.random.random((action.shape[0], 1)) < eexplore).astype(np.float32)
        randoms = np.random.random(action.shape) * (2 * action_scale) - action_scale
        action = mask * randoms + (1 - mask) * action

    return np.clip(action, -action_scale, action_scale)


class StochasticActorPolicy(mrl.Module):
  """Used for SAC / learned action noise"""
  def __init__(self):
    super().__init__(
        'policy',
        required_agent_modules=[
            'actor', 'env', 'replay_buffer'
        ],
        locals=locals())

  def _setup(self):
    self.use_actor_target = self.config.get('use_actor_target')

  def __call__(self, state, greedy=False):
    action_scale = self.env.max_action

    # initial exploration and intrinsic curiosity
    res = None
    if self.training:
      if self.config.get('initial_explore') and len(self.replay_buffer) < self.config.initial_explore:
          res = np.array([self.env.action_space.sample() for _ in range(self.env.num_envs)])
      elif hasattr(self, 'ag_curiosity'):
        state = self.ag_curiosity.relabel_state(state)
      
    state = flatten_state(state, self.config.modalities + self.config.goal_modalities)  # flatten goal environments
    if hasattr(self, 'state_normalizer'):
      state = self.state_normalizer(state, update=self.training)
    
    if res is not None:
      return res

    state = self.torch(state)

    if self.use_actor_target:
      action, _ = self.actor_target(state)
    else:
      action, _ = self.actor(state)
    action = self.numpy(action)

    if self.training and not greedy and self.config.get('eexplore'):
      eexplore = self.config.eexplore
      if hasattr(self, 'ag_curiosity'):
        eexplore = self.ag_curiosity.go_explore * self.config.go_eexplore + eexplore
      mask = (np.random.random((action.shape[0], 1)) < eexplore).astype(np.float32)
      randoms = np.random.random(action.shape) * (2 * action_scale) - action_scale
      action = mask * randoms + (1 - mask) * action
    
    return np.clip(action, -action_scale, action_scale)


class BatchConstrainedPolicy(mrl.Module):
  """
  Used for continuous Batch Constrained Algos. Does NOT use a perturbation network. 

  Note: "actor" here must be an "action generator" (s -> {a_i}) not a traditional s -> a map. 
  """
  def __init__(self):
    super().__init__(
        'policy',
        required_agent_modules=[
            'critic', 'actor', 'env', 'replay_buffer'
        ],
        locals=locals())

  def __call__(self, state, greedy=False):
    action_scale = self.env.max_action

    # initial exploration and intrinsic curiosity
    res = None
    if self.training:
      if self.config.get('initial_explore') and len(self.replay_buffer) < self.config.initial_explore:
        res = np.array([self.env.action_space.sample() for _ in range(self.env.num_envs)])
      elif hasattr(self, 'ag_curiosity'):
        state = self.ag_curiosity.relabel_state(state)
        
    state = flatten_state(state, self.config.modalities + self.config.goal_modalities)  # flatten goal environments, # batch x state_dim
    if hasattr(self, 'state_normalizer'):
      state = self.state_normalizer(state, update=self.training)

    if res is not None:
      return res

    state = self.torch(state)

    action_proposals = self.actor(state) # batch x num_proposals x action_dim
    states = torch.repeat_interleave(state, action_proposals.shape[1], 0)
    q_values = self.critic(states, action_proposals.reshape(-1, action_proposals.shape[-1]))
    q_values = q_values.reshape(state.shape[0], action_proposals.shape[1]) # batch x num_proposals
    best_actions = torch.argmax(q_values, dim=-1, keepdims=True) # batch x 1
    action = action_proposals.gather(1, torch.tile(best_actions[:,:,None], (1, 1, action_proposals.shape[2]))).squeeze(1)
    action = self.numpy(action)
    
    if self.training and not greedy:
      action = self.action_noise(action)
      if self.config.get('eexplore'):
        eexplore = self.config.eexplore
        if hasattr(self, 'ag_curiosity'):
          eexplore = self.ag_curiosity.go_explore * self.config.go_eexplore + eexplore
        mask = (np.random.random((action.shape[0], 1)) < eexplore).astype(np.float32)
        randoms = np.random.random(action.shape) * (2 * action_scale) - action_scale
        action = mask * randoms + (1 - mask) * action

    return np.clip(action, -action_scale, action_scale)


class OffPolicyActorCritic(mrl.Module):
  """This is the standard DDPG"""

  def __init__(self):
    super().__init__(
        'algorithm',
        required_agent_modules=['actor','critic','replay_buffer', 'env'],
        locals=locals())

  def _setup(self):
    """Sets up actor/critic optimizers and creates target network modules"""

    self.targets_and_models = []

    # Actor setup
    actor_params = []
    self.actors = []
    for module in list(self.module_dict.values()):
      name = module.module_name
      if name.startswith('actor') and isinstance(module, PytorchModel):
        self.actors.append(module)
        actor_params += list(module.model.parameters())
        target = module.copy(name + '_target')
        target.model.load_state_dict(module.model.state_dict())
        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in target.model.parameters():
          p.requires_grad = False
        self.agent.set_module(name + '_target', target)
        self.targets_and_models.append((target.model, module.model))

    if actor_params:
      self.actor_opt = torch.optim.Adam(
          actor_params,
          lr=self.config.actor_lr,
          weight_decay=self.config.actor_weight_decay)
    else:
      self.actor_opt = AttrDict({'state_dict': lambda: []})
    
    self.actor_params = actor_params

    # Critic setup
    critic_params = []
    self.critics = []
    for module in list(self.module_dict.values()):
      name = module.module_name
      if name.startswith('critic') and isinstance(module, PytorchModel):
        self.critics.append(module)
        critic_params += list(module.model.parameters())
        target = module.copy(name + '_target')
        target.model.load_state_dict(module.model.state_dict())
        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in target.model.parameters():
          p.requires_grad = False
        self.agent.set_module(name + '_target', target)
        self.targets_and_models.append((target.model, module.model))

    self.critic_opt = torch.optim.Adam(
        critic_params,
        lr=self.config.critic_lr,
        weight_decay=self.config.critic_weight_decay)
    
    self.critic_params = critic_params

    self.action_scale = self.env.max_action

  def save(self, save_folder : str):
    path = os.path.join(save_folder, self.module_name + '.pt')
    torch.save({
      'actor_opt_state_dict': self.actor_opt.state_dict(),
      'critic_opt_state_dict': self.critic_opt.state_dict()
    }, path)

  def load(self, save_folder : str):
    path = os.path.join(save_folder, self.module_name + '.pt')
    checkpoint = torch.load(path)
    #self.actor_opt.load_state_dict(checkpoint['actor_opt_state_dict'])
    self.critic_opt.load_state_dict(checkpoint['critic_opt_state_dict'])

  def _optimize(self):
    if len(self.replay_buffer) > self.config.warm_up:
      states, actions, rewards, next_states, gammas = self.replay_buffer.sample(
          self.config.batch_size)

      self.optimize_from_batch(states, actions, rewards, next_states, gammas)
      
      if self.config.opt_steps % self.config.target_network_update_freq == 0:
        for target_model, model in self.targets_and_models:
          soft_update(target_model, model, self.config.target_network_update_frac)

  def optimize_from_batch(self, states, actions, rewards, next_states, gammas):
    raise NotImplementedError('Subclass this!')


class DDPG(OffPolicyActorCritic):

  def optimize_from_batch(self, states, actions, rewards, next_states, gammas):

    with torch.no_grad():
      q_next = self.critic_target(next_states, self.actor_target(next_states))
      target = (rewards + gammas * q_next)
      target = torch.clamp(target, *self.config.clip_target_range)

    if hasattr(self, 'logger') and self.config.opt_steps % 1000 == 0:
      self.logger.add_histogram('Optimize/Target_q', target)
    
    q = self.critic(states, actions)
    critic_loss = F.mse_loss(q, target)

    self.critic_opt.zero_grad()
    critic_loss.backward()
      
    # Grad clipping
    if self.config.grad_norm_clipping > 0.:	
      for p in self.critic_params:
        clip_coef = self.config.grad_norm_clipping / (1e-6 + torch.norm(p.grad.detach()))
        if clip_coef < 1:
          p.grad.detach().mul_(clip_coef)
    if self.config.grad_value_clipping > 0.:
      torch.nn.utils.clip_grad_value_(self.critic_params, self.config.grad_value_clipping)

    self.critic_opt.step()

    for p in self.critic_params:
      p.requires_grad = False

    a = self.actor(states)
    if self.config.get('policy_opt_noise'):
      noise = torch.randn_like(a) * (self.config.policy_opt_noise * self.action_scale)
      a = (a + noise).clamp(-self.action_scale, self.action_scale)
      
    actor_loss = -self.critic(states, a)[:,-1].mean()
    if self.config.action_l2_regularization:
      actor_loss += self.config.action_l2_regularization * F.mse_loss(a / self.action_scale, torch.zeros_like(a))

    self.actor_opt.zero_grad()
    actor_loss.backward()
      
    # Grad clipping
    if self.config.grad_norm_clipping > 0.:	
      for p in self.actor_params:
        clip_coef = self.config.grad_norm_clipping / (1e-6 + torch.norm(p.grad.detach()))
        if clip_coef < 1:
          p.grad.detach().mul_(clip_coef)
    if self.config.grad_value_clipping > 0.:
      torch.nn.utils.clip_grad_value_(self.actor_params, self.config.grad_value_clipping)
      
    self.actor_opt.step()

    for p in self.critic_params:
      p.requires_grad = True


class TD3(OffPolicyActorCritic):

  def optimize_from_batch(self, states, actions, rewards, next_states, gammas):
    config = self.config

    with torch.no_grad():
      a_next_max = self.actor(next_states)
      noise = torch.randn_like(a_next_max) * (config.td3_noise * self.action_scale)
      noise = noise.clamp(-config.td3_noise_clip * self.action_scale,
                          config.td3_noise_clip * self.action_scale)
      a_next_max = (a_next_max + noise).clamp(-self.action_scale, self.action_scale)

      q1, q2 = self.critic_target(next_states, a_next_max), self.critic2_target(
          next_states, a_next_max)
      
      target = (rewards + gammas * torch.min(q1, q2))
      target = torch.clamp(target, *self.config.clip_target_range)

    if hasattr(self, 'logger') and self.config.opt_steps % 1000 == 0:
      self.logger.add_histogram('Optimize/Target_q', target)

    q1, q2 = self.critic(states, actions), self.critic2(states, actions)
    critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)

    self.critic_opt.zero_grad()
    critic_loss.backward()
      
    # Grad clipping
    if self.config.grad_norm_clipping > 0.:	
      torch.nn.utils.clip_grad_norm_(self.critic_params, self.config.grad_norm_clipping)
    if self.config.grad_value_clipping > 0.:
      torch.nn.utils.clip_grad_value_(self.critic_params, self.config.grad_value_clipping)

    self.critic_opt.step()

    if config.opt_steps % config.td3_delay == 0:
      for p in self.critic_params:
        p.requires_grad = False

      a = self.actor(states)
      if self.config.get('policy_opt_noise'):
        noise = torch.randn_like(a) * (config.policy_opt_noise * self.action_scale)
        a = (a + noise).clamp(-self.action_scale, self.action_scale)
      actor_loss = -torch.min(self.critic(states, a)[:,-1], self.critic2(states, a)[:,-1]).mean()
      if self.config.action_l2_regularization:
        actor_loss += self.config.action_l2_regularization * F.mse_loss(a / self.action_scale, torch.zeros_like(a))
      if self.config.get('bc_loss'):
        bc_loss = self.config.bc_loss * F.mse_loss(a, actions)
        self.logger.add_scalar('Optimize/bc_loss', bc_loss)
        actor_loss += bc_loss

      self.actor_opt.zero_grad()
      actor_loss.backward()
      
      # Grad clipping
      if self.config.grad_norm_clipping > 0.:	
        torch.nn.utils.clip_grad_norm_(self.actor_params, self.config.grad_norm_clipping)
      if self.config.grad_value_clipping > 0.:
        torch.nn.utils.clip_grad_value_(self.actor_params, self.config.grad_value_clipping)

      self.actor_opt.step()

      for p in self.critic_params:
        p.requires_grad = True


class SAC(OffPolicyActorCritic):

  def optimize_from_batch(self, states, actions, rewards, next_states, gammas):
    config = self.config

    with torch.no_grad():
      # Target actions come from *current* policy
      a_next, logp_next = self.actor(next_states)
      q1 = self.critic_target(next_states, a_next)
      q2 = self.critic2_target(next_states, a_next)
      target = rewards + gammas * (torch.min(q1, q2) - config.entropy_coef * logp_next)
      target = torch.clamp(target, *self.config.clip_target_range)

    if hasattr(self, 'logger') and self.config.opt_steps % 1000 == 0:
      self.logger.add_histogram('Optimize/Target_q', target)

    q1, q2 = self.critic(states, actions), self.critic2(states, actions)
    critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)

    self.critic_opt.zero_grad()
    critic_loss.backward()
      
    # Grad clipping
    if self.config.grad_norm_clipping > 0.:	
      torch.nn.utils.clip_grad_norm_(self.critic_params, self.config.grad_norm_clipping)
    if self.config.grad_value_clipping > 0.:
      torch.nn.utils.clip_grad_value_(self.critic_params, self.config.grad_value_clipping)

    self.critic_opt.step()

    for p in self.critic_params:
      p.requires_grad = False

    a, logp = self.actor(states)
    q = torch.min(self.critic(states, a), self.critic2(states, a))

    actor_loss = (config.entropy_coef * logp - q).mean()

    if self.config.action_l2_regularization:
      actor_loss += self.config.action_l2_regularization * F.mse_loss(a / self.action_scale, torch.zeros_like(a))

    self.actor_opt.zero_grad()
    actor_loss.backward()
      
    # Grad clipping
    if self.config.grad_norm_clipping > 0.:	
      torch.nn.utils.clip_grad_norm_(self.actor_params, self.config.grad_norm_clipping)
    if self.config.grad_value_clipping > 0.:
      torch.nn.utils.clip_grad_value_(self.actor_params, self.config.grad_value_clipping)

    self.actor_opt.step()

    for p in self.critic_params:
      p.requires_grad = True



class BCDDPG(OffPolicyActorCritic):

  def optimize_from_batch(self, states, actions, rewards, next_states, gammas):

    with torch.no_grad():
      action_proposals = self.actor(next_states) # batch x num_proposals x action_dim
      _next_states = torch.repeat_interleave(next_states, action_proposals.shape[1], 0)
      q_values = self.critic_target(_next_states, action_proposals.reshape(-1, action_proposals.shape[-1]))
      q_values = q_values.reshape(next_states.shape[0], action_proposals.shape[1]) # batch x num_proposals
      q_next = q_values.max(-1, keepdims=True)[0]
      target = (rewards + gammas * q_next)
      target = torch.clamp(target, *self.config.clip_target_range)

    if hasattr(self, 'logger') and self.config.opt_steps % 1000 == 0:
      self.logger.add_histogram('Optimize/Target_q', target)
    
    q = self.critic(states, actions)
    critic_loss = F.mse_loss(q, target)

    self.critic_opt.zero_grad()
    critic_loss.backward()
      
    # Grad clipping
    if self.config.grad_norm_clipping > 0.:	
      for p in self.critic_params:
        clip_coef = self.config.grad_norm_clipping / (1e-6 + torch.norm(p.grad.detach()))
        if clip_coef < 1:
          p.grad.detach().mul_(clip_coef)
    if self.config.grad_value_clipping > 0.:
      torch.nn.utils.clip_grad_value_(self.critic_params, self.config.grad_value_clipping)

    self.critic_opt.step()

class SimpleConservativeSAC(OffPolicyActorCritic):

  def optimize_from_batch(self, states, actions, rewards, next_states, gammas):
    config = self.config

    with torch.no_grad():
      # Target actions come from *current* policy
      a_next, logp_next = self.actor(next_states)
      q1 = self.critic_target(next_states, a_next)
      q2 = self.critic2_target(next_states, a_next)
      target = rewards + gammas * (torch.min(q1, q2) - config.entropy_coef * logp_next)
      target = torch.clamp(target, *self.config.clip_target_range)

    if hasattr(self, 'logger') and self.config.opt_steps % 1000 == 0:
      self.logger.add_histogram('Optimize/Target_q', target)

    q1, q2 = self.critic(states, actions), self.critic2(states, actions)
    critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)

    # CQL Bit
    random_actions = (torch.rand((len(states), self.env.action_dim)) * 2 * self.env.max_action - self.env.max_action).to(states.device)
    random_q = self.critic(states, random_actions)
    random_actions = (torch.rand((len(states), self.env.action_dim)) * 2 * self.env.max_action - self.env.max_action).to(states.device)
    random_q += self.critic2(states, random_actions)
    random_q /= 2.

    #with torch.no_grad():
    #  a, logp = self.actor(states)
    #random_q = (self.critic(states, a) +  self.critic2(states, a)) / 2.

    min_q_loss = (F.l1_loss(random_q.mean(), torch.tensor(self.config.clip_target_range[0], dtype=torch.float32).cuda()) +\
                  F.l1_loss((q1 + q2).mean() / 2., torch.tensor(self.config.clip_target_range[1], dtype=torch.float32).cuda())) *\
                 self.config.min_q_weight

    if hasattr(self, 'logger') and self.config.opt_steps % 100 == 0:
      self.logger.add_scalar('Optimize/min_q_loss', min_q_loss)
      self.logger.add_scalar('Optimize/basic_critic_loss', critic_loss)

    critic_loss += min_q_loss

    self.critic_opt.zero_grad()
    critic_loss.backward()
      
    # Grad clipping
    if self.config.grad_norm_clipping > 0.:	
      torch.nn.utils.clip_grad_norm_(self.critic_params, self.config.grad_norm_clipping)
    if self.config.grad_value_clipping > 0.:
      torch.nn.utils.clip_grad_value_(self.critic_params, self.config.grad_value_clipping)

    self.critic_opt.step()

    for p in self.critic_params:
      p.requires_grad = False

    a, logp = self.actor(states)
    q = torch.min(self.critic(states, a), self.critic2(states, a))

    actor_loss = (config.entropy_coef * logp - q).mean()

    if self.config.action_l2_regularization:
      actor_loss += self.config.action_l2_regularization * F.mse_loss(a / self.action_scale, torch.zeros_like(a))

    self.actor_opt.zero_grad()
    actor_loss.backward()
      
    # Grad clipping
    if self.config.grad_norm_clipping > 0.:	
      torch.nn.utils.clip_grad_norm_(self.actor_params, self.config.grad_norm_clipping)
    if self.config.grad_value_clipping > 0.:
      torch.nn.utils.clip_grad_value_(self.actor_params, self.config.grad_value_clipping)

    self.actor_opt.step()

    for p in self.critic_params:
      p.requires_grad = True
