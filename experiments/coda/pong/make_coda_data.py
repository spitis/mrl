import os, sys

import numpy as np, torch
from tqdm import tqdm

from mrl.import_all import *
from mrl.configs.make_continuous_agents import *
from experiments.coda.coda_module import CodaOldBuffer
from experiments.coda.sandy_module import CodaAttentionBasedMask, SimpleStackedAttn, PongClassifierRewardModel, SimpleMLP
from experiments.coda.pong.pong_env import CustomPong

def blockPrint():
    sys.stdout = open(os.devnull, 'w')

def enablePrint():
    sys.stdout = sys.__stdout__

def make_env():
  return CustomPong()

def ground_truth_compute_reward(s, a, ns):
  p0x = ns[:,0]
  p1x = ns[:,4]
  bx  = ns[:,8]
  
  absbx = bx * 2.1
  absp0x = p0x * 0.3 - 1.3
  absp1x = p1x * 0.3 + 1.3

  r = np.zeros_like(p0x)
  
  r[absbx >= absp1x] = 1.
  r[absbx <= absp0x] = -1.
  return r.reshape(-1, 1)

def make_coda_data(real_dataset, coda_data, seed=0, amt_real_data=None, reward_fn=None):
  dataset = real_dataset
  if coda_data == 0:
    print("Did not make any Coda data!")
    return real_dataset

  assert len(dataset[0]) >= 10000, "Need at least 10K real samples!"

  print("Making a dummy agent... ", end='')

  blockPrint()

  config = make_sac_agent(spinning_up_sac_config(), args=AttrDict(
    parent_folder='/tmp/make_coda_data',
    env=make_env,
    alg='sac',
    layers=(128,128),
    tb='',
    replay_size=500000,
    seed=seed
  ), agent_name_attrs=['alg', 'seed', 'tb'])
  del config.module_state_normalizer
  del config.module_replay
  config.module_replay = CodaOldBuffer(
    max_coda_ratio=0.5,
    num_coda_source_transitions=5000, 
    num_coda_source_pairs=1000, 
    coda_samples_per_pair=2,
    coda_buffer_size=1250000,
    add_bad_dcs=False)

  config.slot_state_dims = [[0,1,2,3],[4,5,6,7],[8,9,10,11]]
  config.slot_action_dims = [[0,1]]
  model = SimpleStackedAttn(14, 12, num_attn_blocks=2, num_hidden_layers=3, num_hidden_units=256, num_heads=1)
  config.mask_module = CodaAttentionBasedMask(model=model, optimize_every=1, batch_size=256)

  config.never_done = True
  config.min_experience_to_train_coda_attn = 0
  agent  = mrl.config_to_agent(config)

  reward_module = PongClassifierRewardModel(SimpleMLP((12+2+12, 128, 3)), optimize_every=1, batch_size=512)
  agent.set_module('reward_module', reward_module)

  enablePrint()
  print("OK!")

  """Add the real dataset"""
  if amt_real_data is None:
    agent.replay_buffer.buffer.add_batch(*dataset)
  else:
    agent.replay_buffer.buffer.add_batch(*[x[:amt_real_data] for x in dataset])

  """Train the attention model on real data only"""
  attn_losses, rew_losses = [], []
  print("Training attention model...")
  for i in tqdm(range(2000)):
    attn_losses.append(agent.coda_attention_model._optimize())
    rew_losses.append(agent.reward_module._optimize())

  """Make sure it trained OK"""
  assert np.mean(attn_losses[-10:]) < 0.005
  assert np.mean(rew_losses[-10:]) < 0.1

  """Add the extra (MBPO) data before applying CoDA"""
  if amt_real_data is not None:
    agent.replay_buffer.buffer.add_batch(*[x[amt_real_data:] for x in dataset])

  """Now let's make some Coda data"""
  print("Making Coda data...")
  from functools import partial
  agent.get_coda_mask = partial(agent.coda_attention_model.get_mask, THRESH=0.02)
  if reward_fn is None:
    agent.replay_buffer.reward_fn = agent.reward_module.compute_reward
  else:
    agent.replay_buffer.reward_fn = reward_fn

  # Do one step to see how much Coda data is made
  agent.replay_buffer._optimize()
  delta = len(agent.replay_buffer.coda_buffer)
  
  for i in tqdm(range(int(coda_data + 15000) // delta)):
    agent.replay_buffer._optimize()


  s, a, r, ns, d = dataset

  CODASIZE = min(coda_data, len(agent.replay_buffer.coda_buffer))
  coda_s  = agent.replay_buffer.coda_buffer.items['state'].data[:CODASIZE]
  coda_a  = agent.replay_buffer.coda_buffer.items['action'].data[:CODASIZE]
  coda_ns = agent.replay_buffer.coda_buffer.items['next_state'].data[:CODASIZE]
  coda_d  = agent.replay_buffer.coda_buffer.items['done'].data[:CODASIZE]

  # Manually relabel reward since coda buffer doesn't do it until you sample
  # In 20 sections just to make sure it doesn't blow the memory.
  coda_r = []
  for x, y, z in zip(np.array_split(coda_s, 20), np.array_split(coda_a, 20), np.array_split(coda_ns, 20)):
    coda_r.append(agent.replay_buffer.reward_fn(x, y, z))
  coda_r = np.concatenate(coda_r, 0)

  dataset = (
    np.concatenate((s,  coda_s)),
    np.concatenate((a,  coda_a)), 
    np.concatenate((r,  coda_r)), 
    np.concatenate((ns, coda_ns)), 
    np.concatenate((d,  coda_d))
  )

  print(f'Successfully made {CODASIZE} coda samples!')
  return dataset