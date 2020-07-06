"""
To benchmark implementation on multi-goal Fetch tasks.
"""

from mrl.import_all import *
from mrl.configs.make_continuous_agents import *

import time
import os
import gym
import numpy as np
import torch

config = best_slide_config()
config.alg = 'ddpg'

def main(args):
  if args.num_envs is None:
    import multiprocessing as mp
    args.num_envs = max(mp.cpu_count() - 1, 1)

  # hard code num_eval envs...
  args.num_eval_envs = args.num_envs

  merge_args_into_config(args, config)

  if config.gamma < 1.: config.clip_target_range = (np.round(-(1 / (1 - config.gamma)), 2), 0.)
  if config.gamma == 1: config.clip_target_range = (np.round(-args.env_max_step - 5, 2), 0.)

  config.agent_name = make_agent_name(config, ['env', 'her', 'seed', 'tb'], prefix=args.prefix)

  # 5. Setup / add basic modules to the config
  config.update(
      dict(trainer=StandardTrain(),
           evaluation=EpisodicEval(),
           policy=ActorPolicy(),
           logger=Logger(),
           state_normalizer=Normalizer(MeanStdNormalizer()),
           replay=OnlineHERBuffer(),
           action_noise=ContinuousActionNoise(GaussianProcess, std=ConstantSchedule(args.action_noise)),
           algorithm=DDPG()))

  torch.set_num_threads(min(4, args.num_envs))
  torch.set_num_interop_threads(min(4, args.num_envs))

  assert gym.envs.registry.env_specs.get(args.env) is not None
  env = lambda: gym.make(args.env)

  config.module_train_env = EnvModule(env, num_envs=config.num_envs, seed=config.seed)
  config.module_eval_env = EnvModule(env, num_envs=config.num_eval_envs, name='eval_env', seed=config.seed + 1138)

  e = config.module_eval_env
  config.actor = PytorchModel(
      'actor', lambda: Actor(FCBody(e.state_dim + e.goal_dim, args.layers, nn.LayerNorm), e.action_dim, e.max_action))
  config.critic = PytorchModel(
      'critic', lambda: Critic(FCBody(e.state_dim + e.goal_dim + e.action_dim, args.layers, nn.LayerNorm), 1))

  if e.goal_env:
    config.never_done = True  # NOTE: This is important in the standard Goal environments, which are never done

  agent  = mrl.config_to_agent(config)
  
  num_eps = max(args.num_eval_envs * 3, 10)
  res = np.mean(agent.eval(num_episodes=num_eps).rewards)
  agent.logger.log_color(f'Initial test reward ({num_eps} eps):', f'{res:.2f}')

  for epoch in range(int(args.max_steps // args.epoch_len)):
    t = time.time()
    agent.train(num_steps=args.epoch_len)

    # EVALUATE
    res = np.mean(agent.eval(num_episodes=num_eps).rewards)
    agent.logger.log_color(f'Test reward ({num_eps} eps):', f'{res:.2f}')
    agent.logger.log_color('Epoch time:', '{:.2f}'.format(time.time() - t), color='yellow')

    print(f"Saving agent at epoch {epoch}")
    agent.save('checkpoint')

# 3. Declare args for modules (also parent_folder is required!)
if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser(description="Train HER", formatter_class=lambda prog: argparse.RawTextHelpFormatter(prog, max_help_position=100, width=120))
  parser.add_argument('--parent_folder', default='./results', type=str, help='where to save progress')
  parser.add_argument('--prefix', type=str, default='her', help='Prefix for agent name (subfolder where it is saved)')
  parser.add_argument('--env', default="FetchReach-v1", type=str, help="gym environment")
  parser.add_argument('--max_steps', default=1000000, type=int, help="maximum number of training steps")
  parser.add_argument('--layers', nargs='+', default=(512, 512, 512), type=int, help='hidden layers for actor/critic')
  parser.add_argument('--tb', default='', type=str, help='a tag for the agent name / tensorboard')
  parser.add_argument('--epoch_len', default=5000, type=int, help='number of steps between evals')
  parser.add_argument('--num_envs', default=None, type=int, help='number of envs (defaults to procs - 1)')
  parser.add_argument('--env_max_step', default=50, type=int, help='max_steps_env_environment')

  parser = add_config_args(parser, config)
  args = parser.parse_args()

  import subprocess, sys
  args.launch_command = sys.argv[0] + ' ' + subprocess.list2cmdline(sys.argv[1:])

  main(args)
