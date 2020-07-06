import numpy as np
import torch
import gym
import time
import os
import json
from copy import deepcopy
from functools import partial

from mrl.import_all import *
from mrl.configs.make_continuous_agents import *
from mrl.utils.misc import batch_block_diag

from envs.customfetch.custom_fetch import DisentangledFetchPushEnv,\
  DisentangledFetchSlideEnv, DisentangledFetchPickAndPlaceEnv, SlideNEnv, PushNEnv
from gym.wrappers import TimeLimit

from experiments.coda.coda_generic import get_true_abstract_mask_spriteworld, batch_get_heuristic_mask_fetchpush
from experiments.coda.coda_module import CodaBuffer
from experiments.coda.sandy_module import CodaAttentionBasedMask, SimpleStackedAttn

config = best_slide_config()
config.alg = 'ddpg'
import multiprocessing as mp


def make_disentangled_fetch_env(envstr):
  if 'push' in envstr.lower():
    env = lambda: TimeLimit(DisentangledFetchPushEnv(), 50)
    eval_env = env
    dummy_env = gym.make('FetchPush-v1')
  elif 'pick' in envstr.lower():
    env = lambda: TimeLimit(DisentangledFetchPickAndPlaceEnv(), 50)
    eval_env = env
    dummy_env = gym.make('FetchPickAndPlace-v1')
  elif 'slide' in envstr.lower():
    env = lambda: TimeLimit(DisentangledFetchSlideEnv(), 50)
    eval_env = lambda: TimeLimit(DisentangledFetchSlideEnv(), 50)
    dummy_env = DisentangledFetchSlideEnv()

  I = np.concatenate((np.zeros(10), np.ones(12))).astype(np.int64)
  J = np.arange(22, dtype=np.int64)
  state_dims = (I, J)
  return env, eval_env, dummy_env, state_dims


def main(args):

  if args.num_envs is None:
    import multiprocessing as mp
    args.num_envs = max(mp.cpu_count() - 1, 1)

  # hard code num_eval envs...
  args.num_eval_envs = args.num_envs

  merge_args_into_config(args, config)
  config.min_experience_to_train_coda_attn = args.min_experience_to_train_coda_attn
  if config.gamma < 1.: config.clip_target_range = (np.round(-(1 / (1 - config.gamma)), 2), 0.)
  if config.gamma == 1: config.clip_target_range = (np.round(-args.env_max_step - 5, 2), 0.)

  config.agent_name = make_agent_name(config, [
      'envstr', 'alg', 'her', 'relabel_type', 'seed', 'tb', 'max_coda_ratio', 'coda_every', 'coda_source_pairs',
      'batch_size', 'optimize_every'
  ],
                                      prefix=args.prefix)

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

  if gym.envs.registry.env_specs.get(args.envstr) is not None:
    args.env = args.envstr
    dummy_env_config = None
    reward_fn = None
  elif 'disentangled' in args.envstr.lower():
    args.env, args.eval_env, dummy_env, state_dims = make_disentangled_fetch_env(args.envstr)
    config.slot_state_dims = [np.arange(10)] + [10 + 12 * i + np.arange(12) for i in range(1)]
    config.slot_action_dims = None
    dummy_env_config = None
    reward_fn = lambda s, a, ns, g: dummy_env.compute_reward(ns[:, 10:13], g, None)[:, None]

  elif 'slide' in args.envstr.lower():
    n = int(args.envstr.lower().replace('slide', ''))
    args.env = lambda: SlideNEnv(n=n, distance_threshold=args.train_dt)
    args.eval_env = lambda: SlideNEnv(n=n)
    dummy_env_config = None
    dummy_env = SlideNEnv(n=n)
    reward_fn = lambda s, a, ns, g: dummy_env.compute_reward(ns[:, dummy_env.ag_dims], g, None)[:, None]
    config.slot_state_dims = dummy_env.disentangled_idxs
    config.slot_action_dims = None
    config.slot_goal_dims = dummy_env.disentangled_goal_idxs

    if args.relabel_type == 'online_attn':
      model = SimpleStackedAttn(10 + 12 * n + 4,
                                10 + 12 * n,
                                num_attn_blocks=2,
                                num_hidden_layers=2,
                                num_hidden_units=128,
                                num_heads=5)
      config.mask_module = CodaAttentionBasedMask(model=model, optimize_every=2, batch_size=512)
  elif 'push' in args.envstr.lower():
    n = int(args.envstr.lower().replace('push', ''))
    args.env = lambda: PushNEnv(n=n, distance_threshold=args.train_dt)
    args.eval_env = lambda: PushNEnv(n=n)
    dummy_env_config = None
    dummy_env = PushNEnv(n=n)
    reward_fn = lambda s, a, ns, g: dummy_env.compute_reward(ns[:, dummy_env.ag_dims], g, None)[:, None]
    config.slot_state_dims = dummy_env.disentangled_idxs
    config.slot_goal_dims = dummy_env.disentangled_goal_idxs
  else:
    raise NotImplementedError

  if type(args.env) is str:
    env = lambda: gym.make(args.env)
  else:
    env = args.env

  config.module_train_env = EnvModule(env, num_envs=config.num_envs, seed=config.seed)
  config.module_eval_env = EnvModule(env, num_envs=config.num_eval_envs, name='eval_env', seed=config.seed + 1138)

  e = config.module_eval_env
  if config.slot_based_state and hasattr(config, 'slot_state_dims'):
    e.state_dim = len(config.slot_state_dims[0])
  config.actor = PytorchModel(
      'actor', lambda: Actor(FCBody(e.state_dim + e.goal_dim, args.layers, nn.LayerNorm), e.action_dim, e.max_action))
  config.critic = PytorchModel(
      'critic', lambda: Critic(FCBody(e.state_dim + e.goal_dim + e.action_dim, args.layers, nn.LayerNorm), 1))

  if e.goal_env:
    config.never_done = True  # NOTE: This is important in the standard Goal environments, which are never done

  # add Coda buffer if using Coda
  if args.relabel_type is not None:
    del config.replay
    config.module_replay = CodaBuffer(max_coda_ratio=args.max_coda_ratio,
                                      make_coda_data_every=args.coda_every,
                                      num_coda_source_transitions=20000,
                                      num_coda_source_pairs=args.coda_source_pairs,
                                      coda_samples_per_pair=args.coda_samples_per_pair,
                                      coda_buffer_size=args.coda_buffer_size,
                                      add_bad_dcs=args.add_bad_dcs,
                                      coda_on_goal_components=args.coda_on_goal_components,
                                      spriteworld_config=dummy_env_config,
                                      reward_fn=reward_fn,
                                      num_procs=min(args.num_envs, 4))

  agent = mrl.config_to_agent(config)

  # set up get_coda_mask function
  if args.relabel_type is not None:
    if args.relabel_type == 'ground_truth':
      agent.get_coda_mask = get_true_abstract_mask_spriteworld
    elif args.relabel_type == 'push_heuristic':
      agent.get_coda_mask = batch_get_heuristic_mask_fetchpush
    elif args.relabel_type == 'online_attn':
      agent.get_coda_mask = partial(agent.coda_attention_model.get_mask, THRESH=args.thresh)
    else:
      raise NotImplementedError()

  if args.checkpoint_dir is not None:
    # If a checkpoint has been initialized load it.
    if os.path.exists(os.path.join(args.checkpoint_dir, 'INITIALIZED')):
      agent.load_from_checkpoint(args.checkpoint_dir)

  if args.visualize_trained_agent:
    print("Loading agent at epoch {}".format(0))
    agent.load('checkpoint')

    agent.eval_mode()
    env = agent.eval_env
    state = env.reset()

    for _ in range(1000000):
      env.render()
      time.sleep(0.02)
      action = agent.policy(state)
      state, reward, done, info = env.step(action)
      env.render()
      print(reward[0])

  else:

    if args.save_embeddings:
      s1_buff = agent.replay_buffer.buffer.BUFF.buffer_state
      s2_buff = agent.replay_buffer.buffer.BUFF.buffer_next_state
      s1_coda = agent.replay_buffer.coda_buffer.items['state']
      s2_coda = agent.replay_buffer.coda_buffer.items['next_state']

    num_eps = max(args.num_eval_envs * 3, 10)
    res = np.mean(agent.eval(num_episodes=num_eps).rewards)
    agent.logger.log_color(f'Initial test reward ({num_eps} eps):', '{:.2f}'.format(res))

    for epoch in range(int(args.max_steps // args.epoch_len)):
      t = time.time()
      agent.train(num_steps=args.epoch_len)

      # VIZUALIZE GOALS
      if args.save_embeddings:
        idxs = np.random.choice(len(s1_buff), size=min(len(s1_buff), 1000), replace=False)
        last_idxs = np.arange(max(0, len(s1_buff) - args.epoch_len), len(s1_buff))

        rands1 = s1_buff.get_batch(idxs)
        rands1 = np.concatenate((rands1[:, 0, :3], rands1[:, 1, 10:13]), 1)
        agent.logger.add_embedding('rand_s1s', rands1)

        rands1 = s2_buff.get_batch(idxs)
        rands1 = np.concatenate((rands1[:, 0, :3], rands1[:, 1, 10:13]), 1)
        agent.logger.add_embedding('rand_s2s', rands1)

        rands1 = s1_coda.get_batch(idxs)
        rands1 = np.concatenate((rands1[:, 0, :3], rands1[:, 1, 10:13]), 1)
        agent.logger.add_embedding('rand_coda1s', rands1)

        rands1 = s2_coda.get_batch(idxs)
        rands1 = np.concatenate((rands1[:, 0, :3], rands1[:, 1, 10:13]), 1)
        agent.logger.add_embedding('rand_coda2s', rands1)

        rands1 = s1_buff.get_batch(last_idxs)
        rands1 = np.concatenate((rands1[:, 0, :3], rands1[:, 1, 10:13]), 1)
        agent.logger.add_embedding('last_s1s', rands1)

      # EVALUATE
      res = np.mean(agent.eval(num_episodes=num_eps).rewards)
      agent.logger.log_color(f'Test reward ({num_eps} eps):', '{:.2f}'.format(res))
      agent.logger.log_color('Epoch time:', '{:.2f}'.format(time.time() - t), color='yellow')

      if args.relabel_type is not None:
        agent.logger.log_color('Coda buffer size:', len(agent.replay_buffer.coda_buffer))

      print("Saving agent at epoch {}".format(epoch))
      agent.save('checkpoint')

      # Also save to checkpoint if a checkpoint_dir is specified.
      if args.checkpoint_dir is not None:
        agent.save_checkpoint(args.checkpoint_dir)

      # Quit if env_steps > max_steps (since epoch counter starts anew once we reload from checkpoint)
      if agent.config.env_steps > args.max_steps:
        break


# 3. Declare args for modules (also parent_folder is required!)
if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser(
      description="Train CODA",
      formatter_class=lambda prog: argparse.RawTextHelpFormatter(prog, max_help_position=100, width=120))
  parser.add_argument('--parent_folder', default='/tmp/test_coda2', type=str, help='where to save progress')
  parser.add_argument('--prefix', type=str, default='coda', help='Prefix for agent name (subfolder where it is saved)')
  parser.add_argument('--envstr', default="place_sparse_1", type=str, help="env string [see code]")
  parser.add_argument('--max_steps', default=2500000, type=int, help="maximum number of training steps")
  parser.add_argument('--layers', nargs='+', default=(512, 512, 512), type=int, help='hidden layers for actor/critic')
  parser.add_argument('--tb', default='', type=str, help='a tag for the agent name / tensorboard')
  parser.add_argument('--epoch_len', default=5000, type=int, help='number of steps between evals')
  parser.add_argument('--num_envs', default=None, type=int, help='number of envs (defaults to procs - 1)')
  parser.add_argument('--visualize_trained_agent', action='store_true', help="visualize the trained agent")
  parser.add_argument('--train_dt', default=0.05, type=float, help='training distance threshold')

  # Coda Args
  parser.add_argument("--relabel_type", default=None, type=str, help='type of relabeling to do')
  parser.add_argument("--attn_mech_dir",
                      default=None,
                      type=str,
                      help='directory from which to load attention mechanism')
  parser.add_argument("--thresh", default=0.02, type=float, help='Threshold on attention mask')
  parser.add_argument("--max_coda_ratio", default=0.5, type=float, help='Max proportion of coda:real data')
  parser.add_argument("--coda_samples_per_pair", default=2, type=int)
  parser.add_argument("--coda_every", default=500, type=int)
  parser.add_argument("--coda_source_pairs", default=1000, type=int)
  parser.add_argument("--min_experience_to_train_coda_attn", default=25000, type=int)
  parser.add_argument("--coda_buffer_size", default=None, type=int, help="Size of Coda Buffer (== replay_size if None)")
  parser.add_argument("--add_bad_dcs", action='store_true', help="add entangled samples to the coda buffer?")
  parser.add_argument("--coda_on_goal_components",
                      action='store_true',
                      help="relabel the goals themselves when doing coda")
  parser.add_argument('--save_embeddings', action='store_true', help='save ag embeddings during training?')

  # CHECKPOINT
  parser.add_argument('--checkpoint_dir', default=None, help='checkpoint directory, if any')

  parser = add_config_args(parser, config)
  args = parser.parse_args()

  import subprocess, sys
  args.launch_command = sys.argv[0] + ' ' + subprocess.list2cmdline(sys.argv[1:])

  main(args)