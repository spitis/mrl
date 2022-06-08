import mrl
from mrl.utils.misc import AttrDict
import numpy as np

class EpisodicEval(mrl.Module):
  def __init__(self):
    super().__init__('eval', required_agent_modules = ['eval_env', 'policy'], locals=locals())
  
  def __call__(self, num_episodes : int, *unused_args, any_success=False):
    """
    Runs num_steps steps in the environment and returns results.
    Results tracking is done here instead of in process_experience, since 
    experiences aren't "real" experiences; e.g. agent cannot learn from them.  
    """
    self.eval_mode()
    env = self.eval_env
    num_envs = env.num_envs
    
    episode_rewards, episode_steps = [], []
    discounted_episode_rewards = []
    is_successes = []
    record_success = False

    while len(episode_rewards) < num_episodes:
      state = env.reset()

      dones = np.zeros((num_envs,))
      steps = np.zeros((num_envs,))
      is_success = np.zeros((num_envs,))
      ep_rewards = [[] for _ in range(num_envs)]

      while not np.all(dones):
        action = self.policy(state)
        state, reward, dones_, infos = env.step(action)

        for i, (rew, done, info) in enumerate(zip(reward, dones_, infos)):
          if dones[i]:
            continue
          ep_rewards[i].append(rew)
          steps[i] += 1
          if done:
            dones[i] = 1. 
          if 'is_success' in info:
            record_success = True
            is_success[i] = max(info['is_success'], is_success[i]) if any_success else info['is_success'] 

      for ep_reward, step, is_succ in zip(ep_rewards, steps, is_success):
        if record_success:
          is_successes.append(is_succ)
        episode_rewards.append(sum(ep_reward))
        discounted_episode_rewards.append(discounted_sum(ep_reward, self.config.gamma))
        episode_steps.append(step)
    
    if hasattr(self, 'logger'):
      if len(is_successes):
        self.logger.add_scalar('Test/Success', np.mean(is_successes))
      self.logger.add_scalar('Test/Episode_rewards', np.mean(episode_rewards))
      self.logger.add_scalar('Test/Discounted_episode_rewards', np.mean(discounted_episode_rewards))
      self.logger.add_scalar('Test/Episode_steps', np.mean(episode_steps))

    return AttrDict({
      'rewards': episode_rewards,
      'steps': episode_steps
    })

def discounted_sum(lst, discount):
  sum = 0
  gamma = 1
  for i in lst:
    sum += gamma*i
    gamma *= discount
  return sum