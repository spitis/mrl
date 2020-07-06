import os, sys

import numpy as np, torch
from tqdm import tqdm
from experiments.coda.pong.pong_env import CustomPong

from mrl.import_all import *
from mrl.configs.make_continuous_agents import *

def blockPrint():
    sys.stdout = open(os.devnull, 'w')

def enablePrint():
    sys.stdout = sys.__stdout__

def make_env():
  return CustomPong()

def train_batchrl_agent(dataset, agent_tag, num_steps=120000, results_folder='/tmp/pong_results', seed=0, num_eval_eps=10):
  print("Training off-policy agent in batch mode on the dataset...")
  blockPrint()

  config = make_td3_agent(make_td3_agent(), args=AttrDict(
    parent_folder=results_folder,
    env=make_env,
    max_steps=int(1e6),
    replay_size=int(1.6e6),
    alg='td3',
    layers=(128, 128),
    tb=agent_tag,
    actor_lr=3e-4,
    critic_lr=3e-4,
    epoch_len=2500,
    batch_size=1000,
    clip_target_range=(-50, 50),
    num_envs=10,
    num_eval_envs=10,
    optimize_every=1,
    gamma=0.98,
    seed=seed
  ), agent_name_attrs=['alg', 'seed', 'tb'])

  del config.module_state_normalizer
  del config.module_replay
  config.module_replay = OldReplayBuffer()
  config.never_done = True
  config.min_experience_to_train_coda_attn = 0
  agent  = mrl.config_to_agent(config)

  agent.replay_buffer.buffer.add_batch(*dataset)

  enablePrint()

  res = [np.mean(agent.eval(num_eval_eps).rewards)]
  for epoch in tqdm(range(num_steps // 1000)):
    for _ in range(1000):
      agent.train_mode()
      agent.config.env_steps +=1 
      agent.algorithm._optimize()
      agent.eval_mode()
    res += [np.mean(agent.eval(num_eval_eps).rewards)]

  agent.save() 

  print("Done training agent!")
  print("Average score over final 10 epochs: {}".format(np.mean(res[-10:])))