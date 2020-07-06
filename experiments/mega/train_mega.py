# 1. Imports
from mrl.import_all import *
from mrl.modules.train import debug_vectorized_experience
from experiments.mega.make_env import make_env
import time
import os
import gym
import numpy as np
import torch.nn as nn

# 2. Get default config and update any defaults (this automatically updates the argparse defaults)
config = protoge_config()
# config.batch_size = 2000

# 3. Make changes to the argparse below

def main(args):

  # 4. Update the config with args, and make the agent name. 
  if args.num_envs is None:
    import multiprocessing as mp
    args.num_envs = max(mp.cpu_count() - 1, 1)

  merge_args_into_config(args, config)
  
  if config.gamma < 1.: config.clip_target_range = (np.round(-(1 / (1-config.gamma)), 2), 0.)
  if config.gamma == 1: config.clip_target_range = (np.round(- args.env_max_step - 5, 2), 0.)

  if args.sparse_reward_shaping:
    config.clip_target_range = (-np.inf, np.inf)

  config.agent_name = make_agent_name(config, ['env','alg','her','layers','seed','tb','ag_curiosity','eexplore','first_visit_succ', 'dg_score_multiplier','alpha'], prefix=args.prefix)

  # 5. Setup / add basic modules to the config
  config.update(
      dict(
          trainer=StandardTrain(),
          evaluation=EpisodicEval(),
          policy=ActorPolicy(),
          logger=Logger(),
          state_normalizer=Normalizer(MeanStdNormalizer()),
          replay=OnlineHERBuffer(),
      ))

  config.prioritized_mode = args.prioritized_mode
  if config.prioritized_mode == 'mep':
    config.prioritized_replay = EntropyPrioritizedOnlineHERBuffer()

  if not args.no_ag_kde:
    config.ag_kde = RawKernelDensity('ag', optimize_every=1, samples=10000, kernel=args.kde_kernel, bandwidth = args.bandwidth, log_entropy=True)
  if args.ag_curiosity is not None:
    config.dg_kde = RawKernelDensity('dg', optimize_every=500, samples=10000, kernel='tophat', bandwidth = 0.2)
    config.ag_kde_tophat = RawKernelDensity('ag', optimize_every=100, samples=10000, kernel='tophat', bandwidth = 0.2, tag='_tophat')
    if args.transition_to_dg:
      config.alpha_curiosity = CuriosityAlphaMixtureModule()
    if 'rnd' in args.ag_curiosity:
      config.ag_rnd = RandomNetworkDensity('ag')
    if 'flow' in args.ag_curiosity:
      config.ag_flow = FlowDensity('ag')

    use_qcutoff = not args.no_cutoff

    if args.ag_curiosity == 'minq':
      config.ag_curiosity = QAchievedGoalCuriosity(max_steps = args.env_max_step, num_sampled_ags=args.num_sampled_ags, use_qcutoff=use_qcutoff, keep_dg_percent=args.keep_dg_percent)
    elif args.ag_curiosity == 'randq':
      config.ag_curiosity = QAchievedGoalCuriosity(max_steps = args.env_max_step, randomize=True, num_sampled_ags=args.num_sampled_ags, use_qcutoff=use_qcutoff, keep_dg_percent=args.keep_dg_percent)
    elif args.ag_curiosity == 'minkde':
      config.ag_curiosity = DensityAchievedGoalCuriosity(max_steps = args.env_max_step, num_sampled_ags=args.num_sampled_ags, use_qcutoff=use_qcutoff, keep_dg_percent=args.keep_dg_percent)
    elif args.ag_curiosity == 'minrnd':
      config.ag_curiosity = DensityAchievedGoalCuriosity('ag_rnd', max_steps = args.env_max_step, num_sampled_ags=args.num_sampled_ags, use_qcutoff=use_qcutoff, keep_dg_percent=args.keep_dg_percent)
    elif args.ag_curiosity == 'minflow':
      config.ag_curiosity = DensityAchievedGoalCuriosity('ag_flow', max_steps = args.env_max_step, num_sampled_ags=args.num_sampled_ags, use_qcutoff=use_qcutoff, keep_dg_percent=args.keep_dg_percent)
    elif args.ag_curiosity == 'randkde':
      config.ag_curiosity = DensityAchievedGoalCuriosity(alpha = args.alpha, max_steps = args.env_max_step, randomize=True, num_sampled_ags=args.num_sampled_ags, use_qcutoff=use_qcutoff, keep_dg_percent=args.keep_dg_percent)
    elif args.ag_curiosity == 'randrnd':
      config.ag_curiosity = DensityAchievedGoalCuriosity('ag_rnd', alpha = args.alpha, max_steps = args.env_max_step, num_sampled_ags=args.num_sampled_ags, use_qcutoff=use_qcutoff, keep_dg_percent=args.keep_dg_percent)
    elif args.ag_curiosity == 'randflow':
      config.ag_curiosity = DensityAchievedGoalCuriosity('ag_flow', alpha = args.alpha, max_steps = args.env_max_step, num_sampled_ags=args.num_sampled_ags, use_qcutoff=use_qcutoff, keep_dg_percent=args.keep_dg_percent)
    elif args.ag_curiosity == 'goaldisc':
      config.success_predictor = GoalSuccessPredictor(batch_size=args.succ_bs, history_length=args.succ_hl, optimize_every=args.succ_oe)
      config.ag_curiosity = SuccessAchievedGoalCuriosity(max_steps=args.env_max_step, use_qcutoff=use_qcutoff, keep_dg_percent=args.keep_dg_percent)
    elif args.ag_curiosity == 'entropygainscore':
      config.bg_kde = RawKernelDensity('bg', optimize_every=args.env_max_step, samples=10000, kernel=args.kde_kernel, bandwidth = args.bandwidth, log_entropy=True)
      config.bgag_kde = RawJointKernelDensity(['bg','ag'], optimize_every=args.env_max_step, samples=10000, kernel=args.kde_kernel, bandwidth = args.bandwidth, log_entropy=True)
      config.ag_curiosity = EntropyGainScoringGoalCuriosity(max_steps=args.env_max_step, use_qcutoff=use_qcutoff, keep_dg_percent=args.keep_dg_percent)
    else:
      raise NotImplementedError

  if args.noise_type.lower() == 'gaussian': noise_type = GaussianProcess
  if args.noise_type.lower() == 'ou': noise_type = OrnsteinUhlenbeckProcess
  config.action_noise = ContinuousActionNoise(noise_type, std=ConstantSchedule(args.action_noise))

  if args.alg.lower() == 'ddpg': 
    config.algorithm = DDPG()
  elif args.alg.lower() == 'td3':
    config.algorithm = TD3()
    config.target_network_update_freq *= 2
  elif args.alg.lower() == 'dqn': 
    config.algorithm = DQN()
    config.policy = QValuePolicy()
    config.qvalue_lr = config.critic_lr
    config.qvalue_weight_decay = config.actor_weight_decay
    config.double_q = True
    config.random_action_prob = LinearSchedule(1.0, config.eexplore, 1e5)
  else:
    raise NotImplementedError

  # 6. Setup / add the environments and networks (which depend on the environment) to the config
  env, eval_env = make_env(args)
  if args.first_visit_done:
    env1, eval_env1 = env, eval_env
    env = lambda: FirstVisitDoneWrapper(env1())
    eval_env = lambda: FirstVisitDoneWrapper(eval_env1())
  if args.first_visit_succ:
    config.first_visit_succ = True

  config.train_env = EnvModule(env, num_envs=args.num_envs, seed=args.seed)
  config.eval_env = EnvModule(eval_env, num_envs=args.num_eval_envs, name='eval_env', seed=args.seed + 1138)

  e = config.eval_env
  if args.alg.lower() == 'dqn':
    config.qvalue = PytorchModel('qvalue', lambda: Critic(FCBody(e.state_dim + e.goal_dim, args.layers, nn.LayerNorm, make_activ(config.activ)), e.action_dim))
  else:
    config.actor = PytorchModel('actor',
                                lambda: Actor(FCBody(e.state_dim + e.goal_dim, args.layers, nn.LayerNorm, make_activ(config.activ)), e.action_dim, e.max_action))
    config.critic = PytorchModel('critic',
                                lambda: Critic(FCBody(e.state_dim + e.goal_dim + e.action_dim, args.layers, nn.LayerNorm, make_activ(config.activ)), 1))
    if args.alg.lower() == 'td3':
      config.critic2 = PytorchModel('critic2',
        lambda: Critic(FCBody(e.state_dim + e.goal_dim + e.action_dim, args.layers, nn.LayerNorm, make_activ(config.activ)), 1))

  if args.ag_curiosity == 'goaldisc':
    config.goal_discriminator = PytorchModel('goal_discriminator', lambda: Critic(FCBody(e.state_dim + e.goal_dim, args.layers, nn.LayerNorm, make_activ(config.activ)), 1))

  if args.reward_module == 'env':
    config.goal_reward = GoalEnvReward()
  elif args.reward_module == 'intrinsic':
    config.goal_reward = NeighborReward()
    config.neighbor_embedding_network = PytorchModel('neighbor_embedding_network',
                                                     lambda: FCBody(e.goal_dim, (256, 256)))
  else:
    raise ValueError('Unsupported reward module: {}'.format(args.reward_module))

  if config.eval_env.goal_env:
    if not (args.first_visit_done or args.first_visit_succ):
      config.never_done = True  # NOTE: This is important in the standard Goal environments, which are never done


  # 7. Make the agent and run the training loop.
  agent = mrl.config_to_agent(config)

  if args.visualize_trained_agent:
    print("Loading agent at epoch {}".format(0))
    agent.load('checkpoint')
    
    if args.intrinsic_visualization:
      agent.eval_mode()
      agent.train(10000, render=True, dont_optimize=True)

    else:
      agent.eval_mode()
      env = agent.eval_env

      for _ in range(10000):
        print("NEW EPISODE")
        state = env.reset()
        env.render()
        done = False
        while not done:
          time.sleep(0.02)
          action = agent.policy(state)
          state, reward, done, info = env.step(action)
          env.render()
          print(reward[0])
  else:
    ag_buffer = agent.replay_buffer.buffer.BUFF.buffer_ag
    bg_buffer = agent.replay_buffer.buffer.BUFF.buffer_bg

    # EVALUATE
    res = np.mean(agent.eval(num_episodes=30).rewards)
    agent.logger.log_color('Initial test reward (30 eps):', '{:.2f}'.format(res))

    for epoch in range(int(args.max_steps // args.epoch_len)):
      t = time.time()
      agent.train(num_steps=args.epoch_len)

      # VIZUALIZE GOALS
      if args.save_embeddings:
        sample_idxs = np.random.choice(len(ag_buffer), size=min(len(ag_buffer), args.epoch_len), replace=False)
        last_idxs = np.arange(max(0, len(ag_buffer)-args.epoch_len), len(ag_buffer))
        agent.logger.add_embedding('rand_ags', ag_buffer.get_batch(sample_idxs))
        agent.logger.add_embedding('last_ags', ag_buffer.get_batch(last_idxs))
        agent.logger.add_embedding('last_bgs', bg_buffer.get_batch(last_idxs))

      # EVALUATE
      res = np.mean(agent.eval(num_episodes=30).rewards)
      agent.logger.log_color('Test reward (30 eps):', '{:.2f}'.format(res))
      agent.logger.log_color('Epoch time:', '{:.2f}'.format(time.time() - t), color='yellow')

      print("Saving agent at epoch {}".format(epoch))
      agent.save('checkpoint')


# 3. Declare args for modules (also parent_folder is required!)
if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser(description="Train DDPG", formatter_class=lambda prog: argparse.RawTextHelpFormatter(prog, max_help_position=100, width=120))
  parser.add_argument('--parent_folder', default='/tmp/test_mega', type=str, help='where to save progress')
  parser.add_argument('--prefix', type=str, default='proto', help='Prefix for agent name (subfolder where it is saved)')
  parser.add_argument('--env', default="FetchPush-v1", type=str, help="gym environment")
  parser.add_argument('--max_steps', default=5000000, type=int, help="maximum number of training steps")
  parser.add_argument('--alg', default='DDPG', type=str, help='algorithm to use (DDPG or TD3)')
  parser.add_argument(
      '--layers', nargs='+', default=(512,512,512), type=int, help='sizes of layers for actor/critic networks')
  parser.add_argument('--noise_type', default='Gaussian', type=str, help='type of action noise (Gaussian or OU)')
  parser.add_argument('--tb', default='', type=str, help='a tag for the agent name / tensorboard')
  parser.add_argument('--epoch_len', default=5000, type=int, help='number of steps between evals')
  parser.add_argument('--num_envs', default=None, type=int, help='number of envs')

  # Make env args
  parser.add_argument('--eval_env', default='', type=str, help='evaluation environment')
  parser.add_argument('--test_with_internal', default=True, type=bool, help='test with internal reward fn')
  parser.add_argument('--reward_mode', default=0, type=int, help='reward mode')
  parser.add_argument('--env_max_step', default=50, type=int, help='max_steps_env_environment')
  parser.add_argument('--per_dim_threshold', default='0.', type=str, help='per_dim_threshold')
  parser.add_argument('--hard', action='store_true', help='hard mode: all goals are high up in the air')
  parser.add_argument('--pp_in_air_percentage', default=0.5, type=float, help='sets in air percentage for fetch pick place')
  parser.add_argument('--pp_min_air', default=0.2, type=float, help='sets the minimum height in the air for fetch pick place when in hard mode')
  parser.add_argument('--pp_max_air', default=0.45, type=float, help='sets the maximum height in the air for fetch pick place')
  parser.add_argument('--train_dt', default=0., type=float, help='training distance threshold')
  parser.add_argument('--slow_factor', default=1., type=float, help='slow factor for moat environment; lower is slower. ')

  # Other args
  parser.add_argument('--first_visit_succ', action='store_true', help='Episodes are successful on first visit (soft termination).')
  parser.add_argument('--first_visit_done', action='store_true', help='Episode terminates upon goal achievement (hard termination).')
  parser.add_argument('--ag_curiosity', default=None, help='the AG Curiosity model to use: {minq, randq, minkde}')
  parser.add_argument('--bandwidth', default=0.1, type=float, help='bandwidth for KDE curiosity')
  parser.add_argument('--kde_kernel', default='gaussian', type=str, help='kernel for KDE curiosity')
  parser.add_argument('--num_sampled_ags', default=100, type=int, help='number of ag candidates sampled for curiosity')
  parser.add_argument('--alpha', default=-1.0, type=float, help='Skewing parameter on the empirical achieved goal distribution. Default: -1.0')
  parser.add_argument('--reward_module', default='env', type=str, help='Reward to use (env or intrinsic)')
  parser.add_argument('--save_embeddings', action='store_true', help='save ag embeddings during training?')
  parser.add_argument('--succ_bs', default=100, type=int, help='success predictor batch size')
  parser.add_argument('--succ_hl', default=200, type=int, help='success predictor history length')
  parser.add_argument('--succ_oe', default=250, type=int, help='success predictor optimize every')
  parser.add_argument('--ag_pred_ehl', default=5, type=int, help='achieved goal predictor number of timesteps from end to consider in episode')
  parser.add_argument('--transition_to_dg', action='store_true', help='transition to the dg distribution?')
  parser.add_argument('--no_cutoff', action='store_true', help="don't use the q cutoff for curiosity")
  parser.add_argument('--visualize_trained_agent', action='store_true', help="visualize the trained agent")
  parser.add_argument('--intrinsic_visualization', action='store_true', help="if visualized agent should act intrinsically; requires saved replay buffer!")
  parser.add_argument('--keep_dg_percent', default=-1e-1, type=float, help='Percentage of time to keep desired goals')
  parser.add_argument('--prioritized_mode', default='none', type=str, help='Modes for prioritized replay: none, mep (default: none)')
  parser.add_argument('--no_ag_kde', action='store_true', help="don't track ag kde")

  parser = add_config_args(parser, config)
  args = parser.parse_args()

  import subprocess, sys
  args.launch_command = sys.argv[0] + ' ' + subprocess.list2cmdline(sys.argv[1:])

  main(args)
