"""

This script converts policies collected with collect_policies.py to a replay buffer. 

E.g.:
PYTHONPATH=./ python experiments/batch_rl/policy_to_buffer.py --parent_folder {YOUR FOLDER}/batchrl/ --env Ant-v3 --num_envs 10 --load_folder {CHECKPOINT_DIRECTORY}
PYTHONPATH=./ python experiments/batch_rl/policy_to_buffer.py --parent_folder {YOUR FOLDER}/batchrl/ --env HalfCheetah-v3 --num_envs 10 --load_folder {CHECKPOINT_DIRECTORY}
...

"""

from mrl.import_all import *
from mrl.configs.make_continuous_agents import *
from experiments.mega.make_env import make_env

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

  # LOAD THE AGENT
  assert os.path.exists(args.load_folder)
  with open(os.path.join(args.load_folder, 'config.pickle'), 'rb') as f:
    agent.config = pickle.load(f)
  for module in agent.module_dict.values():
    # DONT LOAD THE REPLAY BUFFER
    if module.module_name is not 'replay_buffer':
      print("Loading module {}".format(module.module_name))
      module.load(args.load_folder)
  
  res = agent.eval(num_episodes=5).rewards
  agent.logger.log_color('Initial test reward ({} eps): {:.2f}'.format(len(res), np.mean(res)))

  for epoch in range(int(args.max_steps // args.epoch_len)):
    t = time.time()
    if args.collect_policy_type == 'exploratory':
      agent.train(num_steps=args.epoch_len, dont_optimize=True)
    else:
      agent.eval_mode()
      agent.train(num_steps=args.epoch_len, dont_optimize=True, dont_train=True)
    
    # EVALUATE
    res = agent.eval(num_episodes=5).rewards
    agent.logger.log_color('Test reward ({} eps): {:.2f}'.format(len(res), np.mean(res)))
    res = int(np.mean(res))
    agent.logger.log_color('Epoch time:', '{:.2f}'.format(time.time() - t), color='yellow')

  print("Saving agent at epoch {}".format(epoch))
  avg_perf = (np.mean(agent.replay_buffer.buffer.items['reward'].data))
  agent.logger.log_color('TOTAL PERFORMANCE: {:.2f}'.format(avg_perf))
  agent.save('checkpoint')

# 3. Declare args for modules (also parent_folder is required!)
if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser(description="Train DDPG", formatter_class=lambda prog: argparse.RawTextHelpFormatter(prog, max_help_position=100, width=120))
  parser.add_argument('--parent_folder', default='/tmp/batchrl', type=str, help='where to save progress')
  parser.add_argument('--load_folder', default='/tmp/batchrl', type=str, help='checkpoint to load')
  parser.add_argument('--prefix', type=str, default='batchrl', help='Prefix for agent name (subfolder where it is saved)')
  parser.add_argument('--env', default="HalfCheetah-v3", type=str, help="gym environment")
  parser.add_argument('--max_steps', default=1000000, type=int, help="maximum number of training steps")
  parser.add_argument('--alg', default='sac', type=str, help='algorithm to use (DDPG or TD3)')
  parser.add_argument('--layers', nargs='+', default=(512, 512, 512), type=int, help='hidden layers for actor/critic')
  parser.add_argument('--tb', default='collected', type=str, help='a tag for the agent name / tensorboard')
  parser.add_argument('--epoch_len', default=5000, type=int, help='number of steps between evals')
  parser.add_argument('--num_envs', default=None, type=int, help='number of envs (defaults to procs - 1)')
  parser.add_argument('--collect_policy_type', default='exploratory', type=str, help='type of policy to collect (exploratory or greedy)')

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
  config.save_replay_buf = True

  parser = add_config_args(parser, config)
  args = parser.parse_args()

  args.layer_norm = True # helps a lot with Humanoid
  args.num_eval_envs = args.num_envs

  import subprocess, sys
  args.launch_command = sys.argv[0] + ' ' + subprocess.list2cmdline(sys.argv[1:])

  config = make(base_config=config, args=args, agent_name_attrs=['env', 'alg', 'seed', 'tb'])

  main(args)
