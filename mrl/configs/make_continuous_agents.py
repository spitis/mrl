from mrl.import_all import *
from argparse import Namespace
import gym
import time


def make_ddpg_agent(base_config=default_ddpg_config,
                    args=Namespace(env='InvertedPendulum-v2',
                                   tb='',
                                   parent_folder='/tmp/mrl',
                                   layers=(256, 256),
                                   num_envs=None),
                    agent_name_attrs=['env', 'seed', 'tb'],
                    **kwargs):

  if callable(base_config):
    base_config = base_config()
  config = base_config

  if hasattr(args, 'num_envs') and args.num_envs is None:
    import multiprocessing as mp
    args.num_envs = max(mp.cpu_count() - 1, 1)

  if not hasattr(args, 'prefix'):
    args.prefix = 'ddpg'
  if not args.tb:
    args.tb = str(time.time())

  merge_args_into_config(args, config)

  config.agent_name = make_agent_name(config, agent_name_attrs, prefix=args.prefix)

  
  base_modules = {
      k: v
      for k, v in dict(module_train=StandardTrain(),
                       module_eval=EpisodicEval(),
                       module_policy=ActorPolicy(),
                       module_logger=Logger(),
                       module_state_normalizer=Normalizer(MeanStdNormalizer()),
                       module_replay=OnlineHERBuffer(),
                       module_action_noise=ContinuousActionNoise(GaussianProcess,
                                                                 std=ConstantSchedule(config.action_noise)),
                       module_algorithm=DDPG()).items() if not k in config
  }

  config.update(base_modules)

  if type(args.env) is str:
    env = lambda: gym.make(args.env)
    eval_env = env
  else:
    env = args.env
    eval_env = env
  
  if hasattr(args, 'eval_env') and args.eval_env is not None:
    if type(args.eval_env) is str:
      eval_env = lambda: gym.make(args.eval_env)
    else:
      eval_env = args.eval_env


    
  config.module_train_env = EnvModule(env, num_envs=config.num_envs, seed=config.seed)
  config.module_eval_env = EnvModule(eval_env, num_envs=config.num_eval_envs, name='eval_env', seed=config.seed + 1138)

  layer_norm = nn.LayerNorm if (hasattr(args, 'layer_norm') and args.layer_norm) else nn.Identity

  e = config.module_eval_env
  config.module_actor = PytorchModel(
      'actor', lambda: Actor(FCBody(e.state_dim + e.goal_dim, args.layers, layer_norm, make_activ(config.activ)), e.action_dim, e.max_action))
  config.module_critic = PytorchModel(
      'critic', lambda: Critic(FCBody(e.state_dim + e.goal_dim + e.action_dim, args.layers, layer_norm, make_activ(config.activ)), 1))

  if e.goal_env:
    config.never_done = True # important for standard Gym goal environments, which are never done

  return config


def make_td3_agent(base_config=spinning_up_td3_config,
                   args=Namespace(env='InvertedPendulum-v2',
                                  tb='',
                                  prefix='td3',
                                  parent_folder='/tmp/mrl',
                                  layers=(256, 256),
                                  num_envs=None),
                   agent_name_attrs=['env', 'seed', 'tb'],
                   **kwargs):
  
  config = make_ddpg_agent(base_config, args, agent_name_attrs, **kwargs)
  del config.module_algorithm
  config.module_algorithm = TD3()

  layer_norm = nn.LayerNorm if (hasattr(args, 'layer_norm') and args.layer_norm) else nn.Identity
  
  e = config.module_eval_env
  config.module_critic2 = PytorchModel('critic2',
      lambda: Critic(FCBody(e.state_dim + e.goal_dim + e.action_dim, args.layers, layer_norm, make_activ(config.activ), False), 1, False))

  return config


def make_sac_agent(base_config=spinning_up_sac_config,
                   args=Namespace(env='InvertedPendulum-v2',
                                  tb='',
                                  prefix='sac',
                                  parent_folder='/tmp/mrl',
                                  layers=(256, 256),
                                  num_envs=None),
                   agent_name_attrs=['env', 'seed', 'tb'],
                   **kwargs):
  
  config = make_ddpg_agent(base_config, args, agent_name_attrs, **kwargs)
  e = config.module_eval_env
  layer_norm = nn.LayerNorm if (hasattr(args, 'layer_norm') and args.layer_norm) else nn.Identity
  
  del config.module_actor
  del config.module_action_noise
  del config.module_policy
  config.module_policy = StochasticActorPolicy()
  del config.module_algorithm
  config.module_algorithm = SAC()

  config.module_actor = PytorchModel(
      'actor', lambda: StochasticActor(FCBody(e.state_dim + e.goal_dim, args.layers, layer_norm, make_activ(config.activ)), 
        e.action_dim, e.max_action, log_std_bounds = (-20, 2)))

  config.module_critic2 = PytorchModel('critic2',
      lambda: Critic(FCBody(e.state_dim + e.goal_dim + e.action_dim, args.layers, layer_norm, make_activ(config.activ), False), 1, False))

  return config