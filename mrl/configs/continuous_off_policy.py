import torch
import numpy as np
from mrl.utils.misc import AnnotatedAttrDict

default_ddpg_config = lambda: AnnotatedAttrDict(
    device=('cuda' if torch.cuda.is_available() else 'cpu', 'torch device (cpu or gpu)'),
    gamma=(0.99, 'discount factor'),
    actor_lr=(1e-3, 'actor learning rate'),
    critic_lr=(1e-3, 'critic learning rate'),
    actor_weight_decay=(0., 'weight decay to apply to actor'),
    action_l2_regularization=(1e-2, 'l2 penalty for action norm'),
    critic_weight_decay=(0., 'weight decay to apply to critic'),
    optimize_every=(2, 'how often optimize is called, in terms of environment steps'),
    batch_size=(2000, 'batch size for training the actors/critics'),
    warm_up=(10000, 'minimum steps in replay buffer needed to optimize'),  
    initial_explore=(10000, 'steps that actor acts randomly for at beginning of training'), 
    grad_norm_clipping=(-1., 'gradient norm clipping'),
    grad_value_clipping=(-1., 'gradient value clipping'),
    target_network_update_frac=(0.005, 'polyak averaging coefficient for target networks'),
    target_network_update_freq=(1, 'how often to update target networks; NOTE: TD3 uses this too!'),
    clip_target_range=((-np.inf, np.inf), 'q/value targets are clipped to this range'),
    td3_noise=(0.1, 'noise added to next step actions in td3'),
    td3_noise_clip=(0.3, 'amount to which next step noise in td3 is clipped'),
    td3_delay=(2, 'how often the actor is trained, in terms of critic training steps, in td3'),
    entropy_coef=(0.2, 'Entropy regularization coefficient for SAC'),
    policy_opt_noise=(0., 'how much policy noise to add to actor optimization'),
    action_noise=(0.1, 'maximum std of action noise'),
    eexplore=(0., 'how often to do completely random exploration (overrides action noise)'),
    go_eexplore=(0.1, 'epsilon exploration bonus from each point of go explore, when using intrinsic curiosity'),
    go_reset_percent=(0., 'probability to reset episode early for each point of go explore, when using intrinsic curiosity'),
    overshoot_goal_percent=(0., 'if using instrinsic FIRST VISIT goals, should goal be overshot on success?'),
    direct_overshoots=(False, 'if using overshooting, should it be directed in a straight line?'),
    dg_score_multiplier=(1., 'if using instrinsic goals, score multiplier for goal candidates that are in DG distribution'),
    cutoff_success_threshold=(0.3, 0.7), # thresholds for decreasing/increasing the cutoff
    initial_cutoff=(-3, 'initial (and minimum) cutoff for intrinsic goal curiosity'),
    activ=('gelu', 'activation to use for hidden layers in networks'),
    curiosity_beta=(-3., 'beta to use for curiosity_alpha module'),
    sigma_l2_regularization=(0., 'l2 regularization on sigma critics log variance'),

    # Below are args to other modules (maybe should live in those modules?)
    seed=(0, 'random seed'),
    replay_size=(int(1e6), 'maximum size of replay buffer'),
    save_replay_buf=(False, 'save replay buffer checkpoint during training?'),
    num_envs=(12, 'number of parallel envs to run'),
    num_eval_envs=(10, 'number of parallel eval envs to run'),
    log_every=(5000, 'how often to log things'),
    varied_action_noise=(False, 'if true, action noise for each env in vecenv is interpolated between 0 and action noise'),
    use_actor_target=(False, 'if true, use actor target network to act in the environment'),
    her=('futureactual_2_2', 'strategy to use for hindsight experience replay'),
    prioritized_mode=('none', 'buffer prioritization strategy'),
    future_warm_up=(25000, 'minimum steps in replay buffer needed to stop doing ONLY future sampling'),  
    sparse_reward_shaping=(0., 'coefficient of euclidean distance reward shaping in sparse goal envs'),
    n_step_returns=(1, 'if using n-step returns, how many steps?'),
    slot_based_state=(False, 'if state is organized by slot; i.e., [batch_size, num_slots, slot_feats]'),
    modalities=(['observation'], 'keys the agent accesses in dictionary env for observations'),
    goal_modalities=(['desired_goal'], 'keys the agent accesses in dictionary env for goals')
)

def protoge_config():
  config = default_ddpg_config()
  config.gamma = 0.98
  config.actor_lr = 1e-3
  config.critic_lr = 1e-3
  config.actor_weight_decay = 0.
  config.action_l2_regularization = 1e-1
  config.target_network_update_freq = 40
  config.target_network_update_frac = 0.05
  config.optimize_every = 1
  config.batch_size = 2000
  config.warm_up = 2500
  config.initial_explore = 5000
  config.replay_size = int(1e6)
  config.clip_target_range = (-50.,0.)
  config.action_noise = 0.1
  config.eexplore = 0.1
  config.go_eexplore = 0.1
  config.go_reset_percent = 0.
  config.her = 'rfaab_1_4_3_1_1'
  config.grad_value_clipping = 5.
  return config

def best_slide_config():
  config = protoge_config()
  config.eexplore = 0.2
  config.grad_value_clipping = -1
  config.her = 'futureactual_2_2'
  config.replay_size = int(2.5e6)
  config.initial_explore = 10000
  config.warm_up = 5000
  config.action_l2_regularization = 1e-2
  config.optimize_every = 2
  config.target_network_update_freq = 10
  config.activ = 'relu'
  return config

def protoge_td3_config():
  config = default_ddpg_config()
  config.gamma = 0.99
  config.actor_lr = 1e-3
  config.critic_lr = 1e-3
  config.actor_weight_decay = 0.
  config.target_network_update_freq = 40
  config.target_network_update_frac = 0.05
  config.optimize_every = 2
  config.batch_size = 1000
  config.warm_up = 2500
  config.replay_size = int(1e6)
  config.action_noise = 0.1
  config.eexplore = 0.1
  config.grad_value_clipping = 5.
  return config

def spinning_up_td3_config():
  config = default_ddpg_config()
  config.gamma = 0.99
  config.replay_size = int(1e6)
  config.target_network_update_frac = 0.005
  config.target_network_update_freq = 2
  config.actor_lr = 1e-3
  config.critic_lr = 1e-3
  config.action_noise = 0.1
  config.td3_noise = 0.2
  config.td3_noise_clip = 0.5
  config.td3_delay = 2
  config.batch_size = 100
  config.warm_up = 1000
  config.optimize_every = 1
  config.action_l2_regularization = 0
  config.activ = 'relu'

  # hidden sizes = (400, 300)
  # no layer norm

  return config

def spinning_up_sac_config():
  config = default_ddpg_config()
  config.gamma = 0.99
  config.replay_size = int(1e6)
  config.target_network_update_frac = 0.005
  config.target_network_update_freq = 1
  config.actor_lr = 1e-3
  config.critic_lr = 1e-3
  config.action_noise = 0.1
  config.td3_noise = 0.2
  config.td3_noise_clip = 0.5
  config.td3_delay = 2
  config.batch_size = 100
  config.warm_up = 1000
  config.optimize_every = 1
  config.action_l2_regularization = 0
  config.activ = 'relu'

  # hidden sizes = (256, 256)
  # no layer norm

  return config

def spinning_up_ddpg_config():
  config = spinning_up_td3_config()
  config.target_network_update_freq = 1
  config.activ = 'relu'
  return config

def td3_config():
  config = spinning_up_td3_config()
  config.actor_lr = 3e-4
  config.critic_lr = 3e-4
  config.batch_size = 256

  # hidden sizes = (256, 256)
  # no layer norm

  return config

