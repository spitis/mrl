import torch
import numpy as np
from mrl.utils.misc import AnnotatedAttrDict
from mrl.utils.schedule import LinearSchedule

default_dqn_config = lambda: AnnotatedAttrDict(
    device=('cuda' if torch.cuda.is_available() else 'cpu', 'torch device (cpu or gpu)'),
    gamma=(0.99, 'discount factor'),
    qvalue_lr=(1e-3, 'Q-value learning rate'),
    qvalue_weight_decay=(0., 'weight decay to apply to qvalue'),
    optimize_every=(2, 'how often optimize is called, in terms of environment steps'),
    batch_size=(1000, 'batch size for training the Q-values'),
    warm_up=(10000, 'minimum steps in replay buffer needed to optimize'),  
    initial_explore=(10000, 'whether to act randomly during warmup'), 
    grad_norm_clipping=(-1, 'gradient norm clipping (implemented as backward hook)'),
    grad_value_clipping=(-1, 'gradient value clipping'),
    random_action_prob=(LinearSchedule(1.0, 0.1, 1e4), 'Epsilon decay schedule'),
    target_network_update_frac=(0.005, 'polyak averaging coefficient for target networks'),
    target_network_update_freq=(2, 'how often to update target networks; NOTE: TD3 uses this too!'),
    clip_target_range=((-np.inf, np.inf), 'q/value targets are clipped to this range'),
    go_eexplore=(0.1, 'epsilon exploration bonus from each point of go explore, when using intrinsic curiosity'),
    go_reset_percent=(0.025, 'probability to reset epsiode early for each point of go explore, when using intrinsic curiosity'),
    overshoot_goal_percent=(False, 'if using instrinsic goals, should goal be overshot on success?'),
    direct_overshoots=(False, 'if using overshooting, should it be directed in a straight line?'),
    dg_score_multiplier=(1., 'if using instrinsic goals, score multiplier for goal candidates that are in DG distribution'),
    cutoff_success_threshold=(0.3, 0.7), # thresholds for decreasing/increasing the cutoff
    initial_cutoff=(-3, 'initial (and minimum) cutoff for intrinsic goal curiosity'),
    double_q=(False, 'Use Double DQN or not. Default: False'),
    activ=('gelu', 'activation to use for hidden layers in networks'),
    curiosity_beta=(-3., 'beta to use for curiosity_alpha module'),

    # Below are args to other modules (maybe should live in those modules?)
    seed=(0, 'random seed'),
    replay_size=(int(1e6), 'maximum size of replay buffer'),
    num_envs=(12, 'number of parallel envs to run'),
    log_every=(5000, 'how often to log things'),
    use_qvalue_target=(False, 'if true, use target network to act in the environment'),
    her=('futureactual_2_2', 'strategy to use for hindsight experience replay'),
    prioritized_mode=('none', 'buffer prioritization strategy'),
    future_warm_up=(20000, 'minimum steps in replay buffer needed to stop doing ONLY future sampling'),  
    sparse_reward_shaping=(0., 'coefficient of euclidean distance reward shaping in sparse goal envs'),
    n_step_returns=(1, 'if using n-step returns, how many steps?'),
    slot_based_state=(False, 'if state is organized by slot; i.e., [batch_size, num_slots, slot_feats]'),
    modalities=(['observation'], 'keys the agent accesses in dictionary env for observations'),
    goal_modalities=(['desired_goal'], 'keys the agent accesses in dictionary env for goals')
)

def dqn_config():
  config = default_dqn_config()
  config.gamma = 0.98
  config.qvalue_lr = 1e-4
  config.qvalue_weight_decay = 0.
  config.target_network_update_freq = 40
  config.target_network_update_frac = 0.05
  config.optimize_every = 2
  config.batch_size = 1000
  config.warm_up = 2500
  config.random_action_prob = LinearSchedule(1.0, 0.1, 1e4)
  config.replay_size = int(1e6)
  return config
