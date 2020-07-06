"""
This is a fixed horizon DDPG, based on https://arxiv.org/pdf/1907.04543.pdf.
"""
from mrl.algorithms.continuous_off_policy import *

class FHDDPG(OffPolicyActorCritic):

  def optimize_from_batch(self, states, actions, rewards, next_states, gammas): 
    config = self.config

    a_next_max = self.actor_target(next_states)
    noise = torch.randn_like(a_next_max) * (config.td3_noise * self.action_scale)
    noise = noise.clamp(-config.td3_noise_clip * self.action_scale,
                        config.td3_noise_clip * self.action_scale)
    a_next_max = (a_next_max + noise).clamp(-self.action_scale,
                                            self.action_scale)

    q_next = self.critic_target(next_states, a_next_max) # batch x n_horizons
    q_next = F.pad(q_next, (1, 0))[:, :-1]

    target = (rewards + gammas * q_next)
    target = torch.clamp(target, *self.config.clip_target_range).detach()

    if hasattr(self, 'logger') and self.config.opt_steps % 100 == 0:
      self.logger.add_histogram('Optimize/First_horizon_target', target[:,0])
      self.logger.add_histogram('Optimize/Last_horizon_target', target[:,-1])
    
    q = self.critic(states, actions)
    critic_loss = F.mse_loss(q, target)

    self.critic_opt.zero_grad()
    critic_loss.backward()
    self.critic_opt.step()

    a = self.actor(states) + noise
    actor_loss = -self.critic(states, a)[:, -5:].mean() # update actor using ensemble of last 5 horizons
    if self.config.action_l2_regularization:
      actor_loss += self.config.action_l2_regularization * F.mse_loss(a, torch.zeros_like(a))

    self.actor_opt.zero_grad()
    actor_loss.backward()
    self.actor_opt.step()