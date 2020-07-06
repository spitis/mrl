"""
To benchmark implementation on standard Mujoco tasks.
"""

from mrl.import_all import *
from mrl.configs.make_continuous_agents import *

import time
import os
import gym
import numpy as np
import torch

def main(config, args):

  # use the old replay buffer, since it adds experience to buffer immediately and don't need HER
  del config.module_replay
  config.module_replay = OldReplayBuffer()

  torch.set_num_threads(min(4, args.num_envs))
  torch.set_num_interop_threads(min(4, args.num_envs))

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
  parser = argparse.ArgumentParser(description="Train Mujoco", formatter_class=lambda prog: argparse.RawTextHelpFormatter(prog, max_help_position=100, width=120))
  parser.add_argument('--parent_folder', default='./results', type=str, help='where to save progress')
  parser.add_argument('--prefix', type=str, default='mujoco', help='Prefix for agent name (subfolder where it is saved)')
  parser.add_argument('--env', default="HalfCheetah-v2", type=str, help="gym environment")
  parser.add_argument('--max_steps', default=5000000, type=int, help="maximum number of training steps")
  parser.add_argument('--alg', default='sac', type=str, help='algorithm to use (DDPG, TD3, SAC)')
  parser.add_argument('--layers', nargs='+', default=(512, 512, 512), type=int, help='hidden layers for actor/critic')
  parser.add_argument('--tb', default='', type=str, help='a tag for the agent name / tensorboard')
  parser.add_argument('--epoch_len', default=5000, type=int, help='number of steps between evals')
  parser.add_argument('--num_envs', default=None, type=int, help='number of envs (defaults to procs - 1)')

  args, unknown = parser.parse_known_args()

  if args.alg.lower() == 'ddpg':
    make = make_ddpg_agent
    config = spinning_up_ddpg_config()
  elif args.alg.lower() == 'td3':
    make = make_td3_agent
    config = spinning_up_td3_config()
  elif args.alg.lower() == 'sac':
    make = make_sac_agent
    config = spinning_up_sac_config()
  config.batch_size = 500
  config.replay_size = int(2.5e6)
  config.warm_up = 10000
  config.initial_explore = 16000

  parser = add_config_args(parser, config)
  args = parser.parse_args()

  args.layer_norm = True # helps a lot with Humanoid
  args.num_eval_envs = args.num_envs

  import subprocess, sys
  args.launch_command = sys.argv[0] + ' ' + subprocess.list2cmdline(sys.argv[1:])

  config = make(base_config=config, args=args, agent_name_attrs=['env', 'alg', 'seed', 'tb'])

  main(config, args)
