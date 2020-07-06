"""
This is a random ensemble hybrid of DDPG/TD3, roughly based on https://arxiv.org/pdf/1907.04543.pdf.
"""

from mrl.algorithms.continuous_off_policy import *

class RandomEnsembleDPG(OffPolicyActorCritic):
  def optimize_from_batch(self, states, actions, rewards, next_states,
                        gammas):
    config = self.config

    a_next_max = self.actor_target(next_states)
    noise = torch.randn_like(a_next_max) * (
        config.td3_noise * self.action_scale)
    noise = noise.clamp(-config.td3_noise_clip * self.action_scale,
                        config.td3_noise_clip * self.action_scale)
    a_next_max = (a_next_max + noise).clamp(-self.action_scale,
                                            self.action_scale)

    qs = []
    for critic in self.critics:
      qs.append(critic(next_states, a_next_max))

    qs = torch.cat(qs, dim=-1) # batch x num_qs

    sample = torch.distributions.dirichlet.Dirichlet(torch.ones(1, qs.size(-1))).sample().to(self.config.device)

    target = (rewards + gammas * (qs * sample).sum(-1, keepdim=True))
    target = torch.clamp(target, *self.config.clip_target_range).detach()

    if hasattr(self, 'logger') and self.config.opt_steps % 100 == 0:
      self.logger.add_histogram('Optimize/Target_q', target)

    qs = []
    for critic in self.critics:
      qs.append(critic(states, actions))
    qs = (torch.cat(qs, dim=-1) * sample).sum(-1, keepdim=True)
    
    critic_loss = F.mse_loss(qs, target)

    self.critic_opt.zero_grad()
    critic_loss.backward()
    self.critic_opt.step()

    if config.opt_steps % config.td3_delay == 0:
      a = self.actor(states)
      qs = []
      for critic in self.critics:
        qs.append(critic(states, a))
      qs = torch.cat(qs, dim=-1)
      
      actor_loss = -qs.mean()

      self.actor_opt.zero_grad()
    actor_loss.backward()
    self.actor_opt.step()