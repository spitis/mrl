# 1. Imports
from mrl.import_all import *
from experiments.mega.make_env import make_env
import time
import numpy as np
import torch, torch.nn as nn

# 2. Get default config and update any defaults (this automatically updates the argparse defaults)
config = best_slide_config()

# 3. Make changes to the argparse below
def main(args):

  # 4. Update the config with args, and make the agent name. 
  if args.num_envs is None:
    import multiprocessing as mp
    args.num_envs = max(mp.cpu_count() - 1, 1)
  merge_args_into_config(args, config)

  torch.set_num_threads(min(8, args.num_envs))
  torch.set_num_interop_threads(min(8, args.num_envs))
  
  if config.gamma < 1.: config.clip_target_range = (np.round(-(1 / (1-config.gamma)), 2), 0.)
  if config.gamma == 1: config.clip_target_range = (np.round(- args.env_max_step - 5, 2), 0.)
  if args.sparse_reward_shaping or 'sac' in args.alg.lower():
    config.clip_target_range = (-np.inf, np.inf)

  config.agent_name = make_agent_name(config, ['env','alg','tb', 'seed'], prefix=args.prefix)

  # 6. Setup environments & add them to config, so modules can refer to them if need be
  env, eval_env = make_env(args)
  if args.first_visit_done: 
    env1, eval_env1 = env, eval_env
    env = lambda: FirstVisitDoneWrapper(env1()) # Terminates the training episode on "done"
    eval_env = lambda: FirstVisitDoneWrapper(eval_env1())
  if args.first_visit_succ:
    config.first_visit_succ = True  # Continues the training episode on "done", but counts it as if "done" (gamma = 0)
  if 'dictpush' in args.env.lower():
    config.modalities = ['gripper', 'object', 'relative']
    if 'reach' in args.env.lower():
      config.goal_modalities = ['gripper_goal', 'object_goal']
    else:
      config.goal_modalities = ['desired_goal']
    config.achieved_goal = GoalEnvAchieved()
  config.train_env = EnvModule(env, num_envs=args.num_envs, seed=args.seed, modalities=config.modalities, goal_modalities=config.goal_modalities)
  config.eval_env = EnvModule(eval_env, num_envs=args.num_eval_envs, name='eval_env', seed=args.seed + 1138, modalities=config.modalities, goal_modalities=config.goal_modalities)

  # 7. Setup / add modules to the config

  # Base Modules
  config.update(
      dict(
          trainer=StandardTrain(),
          evaluation=EpisodicEval(),
          policy=ActorPolicy(),
          logger=Logger(),
          state_normalizer=Normalizer(MeanStdNormalizer()),
          replay=OnlineHERBuffer(),
      ))

  # Goal Selection Modules
  if args.ag_curiosity is not None:
    config.ag_kde = RawKernelDensity('ag', optimize_every=4, samples=2000, kernel=args.kde_kernel, bandwidth = args.bandwidth, log_entropy=True)
    config.dg_kde = RawKernelDensity('dg', optimize_every=500, samples=5000, kernel='tophat', bandwidth = 0.2)
    config.ag_kde_tophat = RawKernelDensity('ag', optimize_every=100, samples=5000, kernel='tophat', bandwidth = 0.2, tag='_tophat')
    if args.transition_to_dg:
      config.alpha_curiosity = CuriosityAlphaMixtureModule()

    use_qcutoff = not args.no_cutoff

    if args.ag_curiosity == 'minkde':
      config.ag_curiosity = DensityAchievedGoalCuriosity(max_steps = args.env_max_step, num_sampled_ags=args.num_sampled_ags, use_qcutoff=use_qcutoff, keep_dg_percent=args.keep_dg_percent)
    else:
      raise NotImplementedError

  # Action Noise Modules
  if args.noise_type.lower() == 'gaussian': noise_type = GaussianProcess
  if args.noise_type.lower() == 'ou': noise_type = OrnsteinUhlenbeckProcess
  config.action_noise = ContinuousActionNoise(noise_type, std=ConstantSchedule(args.action_noise))

  # Algorithm Modules
  if args.alg.lower() == 'ddpg': 
    config.algorithm = DDPG()
  elif args.alg.lower() == 'td3':
    config.algorithm = TD3()
    config.target_network_update_freq *= 2
  elif args.alg.lower() == 'sac':
    config.algorithm = SAC()
  elif args.alg.lower() == 'dqn': 
    config.algorithm = DQN()
    config.policy = QValuePolicy()
    config.qvalue_lr = config.critic_lr
    config.qvalue_weight_decay = config.actor_weight_decay
    config.double_q = True
    config.random_action_prob = LinearSchedule(1.0, config.eexplore, 1e5)
  else:
    raise NotImplementedError

  # 7. Actor/Critic Networks
  e = config.eval_env
  if args.alg.lower() == 'dqn':
    config.qvalue = PytorchModel('qvalue', lambda: Critic(FCBody(e.state_dim + e.goal_dim, args.layers, nn.Identity, make_activ(config.activ)), e.action_dim))
  else:
    config.actor = PytorchModel('actor',
                                lambda: Actor(FCBody(e.state_dim + e.goal_dim, args.layers, nn.Identity, make_activ(config.activ)), e.action_dim, e.max_action))
    config.critic = PytorchModel('critic',
                                lambda: Critic(FCBody(e.state_dim + e.goal_dim + e.action_dim, args.layers, nn.Identity, make_activ(config.activ)), 1))
    if args.alg.lower() in ['td3', 'sac']:
      config.critic2 = PytorchModel('critic2',
        lambda: Critic(FCBody(e.state_dim + e.goal_dim + e.action_dim, args.layers, nn.Identity, make_activ(config.activ)), 1))
    if args.alg.lower() == 'sac':
      del config.actor
      config.actor = PytorchModel('actor', lambda: StochasticActor(FCBody(e.state_dim + e.goal_dim, args.layers, nn.Identity, make_activ(config.activ)), 
                                       e.action_dim, e.max_action, log_std_bounds = (-20, 2)))
      del config.policy
      config.policy = StochasticActorPolicy()
  
  # 8. Reward modules
  if args.reward_module == 'env':
    config.goal_reward = GoalEnvReward()
  elif args.reward_module == 'intrinsic':
    config.goal_reward = NeighborReward()
    config.neighbor_embedding_network = PytorchModel('neighbor_embedding_network', lambda: FCBody(e.goal_dim, (256, 256)))
  else:
    raise ValueError('Unsupported reward module: {}'.format(args.reward_module))

  if config.eval_env.goal_env:
    if not (args.first_visit_done or args.first_visit_succ):
      config.never_done = True  # NOTE: This is important in the standard Goal environments, which are never done

  # 9. Make the agent
  agent = mrl.config_to_agent(config)

  if args.checkpoint_dir is not None:
    # If a checkpoint has been initialized load it.
    if os.path.exists(os.path.join(args.checkpoint_dir, 'INITIALIZED')):
      agent.load_from_checkpoint(args.checkpoint_dir)

  # 10.A Vizualize a trained agent
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

  # 10.B Or run the training loop
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
      
      # Also save to checkpoint if a checkpoint_dir is specified.
      if args.checkpoint_dir is not None:
        agent.save_checkpoint(args.checkpoint_dir)


# 3. Declare args for modules (also parent_folder is required!)
if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser(description="Train DDPG", formatter_class=lambda prog: argparse.RawTextHelpFormatter(prog, max_help_position=100, width=120))
  parser.add_argument('--parent_folder', default='/tmp/test_mega', type=str, help='where to save progress')
  parser.add_argument('--prefix', type=str, default='mrl', help='Prefix for agent name (subfolder where it is saved)')
  parser.add_argument('--env', default="FetchReach-v1", type=str, help="gym environment")
  parser.add_argument('--max_steps', default=1000000, type=int, help="maximum number of training steps")
  parser.add_argument('--alg', default='DDPG', type=str, help='algorithm to use (DDPG or TD3)')
  parser.add_argument(
      '--layers', nargs='+', default=(512,512), type=int, help='sizes of layers for actor/critic networks')
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

  # Checkpoint directory for slurm preemptions
  parser.add_argument('--checkpoint_dir', default=None, help='checkpoint directory, if any')

  parser = add_config_args(parser, config)
  args = parser.parse_args()

  import subprocess, sys
  args.launch_command = sys.argv[0] + ' ' + subprocess.list2cmdline(sys.argv[1:])

  main(args)
