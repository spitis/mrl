import os, sys

import numpy as np, torch
from tqdm import tqdm

from mrl.import_all import *
from mrl.configs.make_continuous_agents import *
from experiments.coda.coda_module import CodaOldBuffer
from experiments.coda.sandy_module import *
from experiments.coda.pong.pong_env import CustomPong

TARGET_LOSS_LEVEL = -4.25


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

def make_dyna_data(real_dataset, dyna_data, num_step_rollouts=5, seed=0):
  dataset = real_dataset
  if dyna_data == 0:
    print("Did not make any Dyna data!")
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
    replay_size=1100000,
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
  model_fn = lambda: SimpleMLP((12+2, 200, 200, 200, 200, 24))
  config.mask_module = MBPOModel(model_fn=model_fn, optimize_every=1, batch_size=256)

  config.never_done = True
  config.min_experience_to_train_coda_attn = 0
  agent  = mrl.config_to_agent(config)

  reward_module = PongClassifierRewardModel(SimpleMLP((12+2+12, 128, 3)), optimize_every=1, batch_size=512)
  agent.set_module('reward_module', reward_module)

  enablePrint()
  print("OK!")

  """Add the real dataset"""
  agent.replay_buffer.buffer.add_batch(*dataset)

  """Train the attention model"""
  attn_losses, rew_losses = [], []
  print("Training attention model...")
  for i in tqdm(range(2000)):
    attn_losses.append(agent.coda_attention_model._optimize())
    rew_losses.append(agent.reward_module._optimize())

  """Now train it to a decent loss..."""
  assert np.mean(rew_losses[-10:]) < 0.1
  i = 0
  while True:
    try:
      assert np.mean(attn_losses[-10:]) < TARGET_LOSS_LEVEL
      break
    except:
      i += 1
      if i > 1000:
        assert False, "failed to train the model... =("
      print(".", end='')
      for _ in range(1000):
        attn_losses.append(agent.coda_attention_model._optimize())
  print("Done training the model! ")

  """Now let's make some Dyna data"""
  print("Making Dyna data...")
  model_fn = agent.coda_attention_model.forward
  reward_fn = agent.reward_module.compute_reward

  while len(agent.replay_buffer.coda_buffer) < dyna_data:
    states = agent.replay_buffer.buffer.sample(1000)[0]
    batch = sample_dyna_batch(states, agent.env.action_space, num_step_rollouts, model_fn, reward_fn)
    agent.replay_buffer.coda_buffer.add_batch(*batch)

  s, a, r, ns, d = dataset

  DYNASIZE = min(dyna_data, len(agent.replay_buffer.coda_buffer))
  coda_s  = agent.replay_buffer.coda_buffer.items['state'].data[:DYNASIZE]
  coda_a  = agent.replay_buffer.coda_buffer.items['action'].data[:DYNASIZE]
  coda_r  = agent.replay_buffer.coda_buffer.items['reward'].data[:DYNASIZE]
  coda_ns = agent.replay_buffer.coda_buffer.items['next_state'].data[:DYNASIZE]
  coda_d  = agent.replay_buffer.coda_buffer.items['done'].data[:DYNASIZE]

  dataset = (
    np.concatenate((s,  coda_s)),
    np.concatenate((a,  coda_a)), 
    np.concatenate((r,  coda_r)), 
    np.concatenate((ns, coda_ns)), 
    np.concatenate((d,  coda_d))
  )

  print(f'Successfully made {DYNASIZE} dyna samples!')
  return dataset, reward_fn

def get_random_actions(states, action_space):
  res = []
  for s in states:
    res.append(action_space.sample())
  return np.array(res)

def sample_dyna_batch(init_states, action_space, num_steps, model_fn, reward_fn):
  s = []
  a = []
  r = []
  ns = []
  d = []

  states = init_states
  for i in range(num_steps):
    actions = get_random_actions(states, action_space)
    next_states = model_fn(states, actions)
    rewards = reward_fn(states, actions, next_states)
    dones = np.zeros_like(rewards)

    s.append(states)
    a.append(actions)
    r.append(rewards)
    ns.append(next_states)
    d.append(dones)

    states = next_states

  return list(map(np.concatenate, (s, a, r, ns, d)))