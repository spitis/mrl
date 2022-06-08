import gym, numpy as np

from envs.customfetch.custom_fetch import FetchHookSweepAllEnv, PushEnv, SlideEnv, PickPlaceEnv, GoalType, StackEnv, PushLeft, PushRight, SlideNEnv, DictPush, DictPushAndReach, FetchHookSweepAllEnv
from envs.customfetch.custom_hand import HandBlockEnv, HandPenEnv, HandEggEnv, HandReachFullEnv
from envs.customfetch.epsilon_wrapper import EpsilonWrapper
from envs.sibrivalry.toy_maze import PointMaze2D, SimpleMazeEnv
from envs.sibrivalry.ant_maze import AntMazeEnv
from gym.wrappers import TimeLimit
from envs.goalgan.ant_maze import AntMazeEnv as GGAntMaze
try:
  import envs.spritelu
  from envs.spritelu.spriteworld.configs.protoge.moat import make_moat_env 
except:
  pass
RotationDict = {'z':'Z', 'parallel':'Parallel','xyz':'XYZ','':''}
HandObjectDict = {'block':'Block', 'pen':'Pen', 'egg':'Egg'}

def make_env(args):
  if ',' in args.per_dim_threshold:
    per_dim_threshold = np.array([float(t) for t in args.per_dim_threshold.split(',')])
  else:
    per_dim_threshold = float(args.per_dim_threshold)
  
  if gym.envs.registry.env_specs.get(args.env) is not None:
    env_fn = lambda: gym.make(args.env)
    eval_env_fn = env_fn
  elif 'dictpushandreach' in args.env.lower():
    env_fn = lambda: TimeLimit(DictPushAndReach(), 50)
    eval_env_fn = lambda: TimeLimit(DictPushAndReach(), 50)
  elif 'dictpush' in args.env.lower():
    env_fn = lambda: TimeLimit(DictPush(), 50)
    eval_env_fn = lambda: TimeLimit(DictPush(), 50)
  elif 'pointmaze' in args.env.lower():
    env_fn = lambda: PointMaze2D()
    eval_env_fn = lambda: PointMaze2D(test=True)
  elif 'simplemaze' in args.env.lower():
    env_fn = lambda: SimpleMazeEnv()
    eval_env_fn = lambda: SimpleMazeEnv(test=True)
  elif 'moat' in args.env.lower():
    env_fn = lambda: make_moat_env(slow_factor=args.slow_factor)
    eval_env_fn = lambda: make_moat_env(slow_factor=args.slow_factor)
  elif 'antmaze' in args.env.lower():
    if 'hiro' in args.env.lower():
      env_fn = lambda: AntMazeEnv(variant='AntMaze-HIRO', eval=False)
      eval_env_fn = lambda: AntMazeEnv(variant='AntMaze-HIRO', eval=True)
    elif 'gg' in args.env.lower():
      env_fn = lambda: GGAntMaze(eval=False)
      eval_env_fn = lambda: GGAntMaze(eval=True)
    else:
      env_fn = lambda: AntMazeEnv(variant='AntMaze-SR', eval=False)
      eval_env_fn = lambda: AntMazeEnv(variant='AntMaze-SR', eval=True)
  elif 'antpush' in args.env.lower():
    env_fn = lambda: AntMazeEnv(variant='AntPush', eval=False)
    eval_env_fn = lambda: AntMazeEnv(variant='AntPush', eval=True)
  elif 'antfall' in args.env.lower():
    env_fn = lambda: AntMazeEnv(variant='AntFall', eval=False)
    eval_env_fn = lambda: AntMazeEnv(variant='AntFall', eval=True)
  elif ('pen_' in args.env.lower()) or ('block_' in args.env.lower()) or ('egg_' in args.env.lower()):
    # The environment name is of the form: {block,pen,egg}_{full,rotate-{z,parallel,xyz}}_{dist_thres}_{rot_thres}
    env_type, mode, dt, rt = args.env.split('_')

    if mode == "full":
      target_pos = 'random'
      target_rot = 'xyz'
      mode_str = "Full"
    else:
      assert("rotate" in mode)
      mode_str = "Rotate"
      target_pos = 'ignore'
      _, target_rot = mode.split('-')
      assert (target_rot in ['z', 'parallel', 'xyz', ''])
      mode_str = "Rotate{}".format(RotationDict[target_rot]) # Some hand env don't specify the rotation type
      # if target_rot == '', set this to 'xyz'
      if target_rot == '':
        target_rot = 'xyz'

    if env_type=='block':
      HandEnv = HandBlockEnv
    elif env_type=='pen':
      HandEnv = HandPenEnv
    elif env_type=='egg':
      HandEnv = HandEggEnv
    else:
      raise ValueError

    max_step = max(args.env_max_step, 100)
    env_fn = lambda: HandEnv(max_step=max_step, distance_threshold=float(dt), rotation_threshold=float(rt),\
                                              target_position=target_pos, target_rotation=target_rot)
    assert(env_type in HandObjectDict.keys())
    gym_env_str = "HandManipulate{}{}-v0".format(HandObjectDict[env_type], mode_str)
    if args.eval_env and args.eval_env.lower() != 'none':
      eval_env_fn = lambda: gym.make(args.eval_env)
    else:
      eval_env_fn = lambda: gym.make(gym_env_str)

  elif ('handreach_' in args.env.lower()):
    env_type, dt = args.env.split('_')
    max_step = max(args.env_max_step, 50)
    env_fn = lambda: HandReachFullEnv(max_step=max_step, distance_threshold=float(dt))
    eval_env_fn = lambda: gym.make('HandReach-v0')
  elif args.env.lower()=='pushright_pushleft':
      env_fn = lambda: PushRight()
      eval_env_fn = lambda: PushLeft()
  elif args.env.lower()=='pushright_pushright':
      env_fn = lambda: PushRight()
      eval_env_fn = lambda: PushRight()
  elif args.env.lower()=='pushleft_pushright':
      env_fn = lambda: PushLeft()
      eval_env_fn = lambda: PushRight()
  elif args.env.lower()=='pushleft_pushleft':
      env_fn = lambda: PushLeft()
      eval_env_fn = lambda: PushLeft()
  elif args.env.lower()=='sweep2':
      env_fn = lambda: FetchHookSweepAllEnv(place_two=True)
      eval_env_fn = lambda: FetchHookSweepAllEnv(place_two=True)
  elif args.env.lower()=='sweep':
      env_fn = lambda: FetchHookSweepAllEnv(smaller_state=True, place_two=True, place_random=False)
      eval_env_fn = lambda: FetchHookSweepAllEnv(smaller_state=True, place_two=True, place_random=False)
  else:
    env, external, internal = args.env.split('_')
    if external.lower() == 'all':
      external = GoalType.ALL
    elif external.lower() == 'objgrip':
      external = GoalType.OBJ_GRIP
    elif external.lower() == 'objspeed':
      external = GoalType.OBJSPEED
    elif external.lower() == 'objspeedrot':
      external = GoalType.OBJSPEED2
    elif external.lower() == 'obj':
      external = GoalType.OBJ
    elif external.lower() == 'grip':
      external = GoalType.GRIP
    else:
      raise ValueError
    
    if internal.lower() == 'all':
      raise ValueError
    elif internal.lower() == 'objgrip':
      internal = GoalType.OBJ_GRIP
    elif internal.lower() == 'obj':
      internal = GoalType.OBJ
    elif internal.lower() == 'grip':
      internal = GoalType.GRIP
    else:
      raise ValueError

    n_blocks = 0
    range_min = None # For pickplace
    range_max = None # For pickplace
    if env.lower() == 'push':
      Env = PushEnv
    elif env.lower() == 'slide':
      Env = SlideEnv
    elif env.lower() == 'pickplace':
      Env = PickPlaceEnv
      n_blocks = args.pp_in_air_percentage # THIS IS THE "IN_AIR_PERCENTAGE"
      range_min = args.pp_min_air # THIS IS THE MINIMUM_AIR
      range_max = args.pp_max_air # THIS IS THE MINIMUM_AIR
    elif 'stack' in env.lower():
      Env = StackEnv
      n_blocks = int(env.lower().replace('stack',''))
    elif 'slide' in env.lower():
      Env = SlideNEnv
      n_blocks = int(env.lower().replace('slide',''))
    else:
      raise ValueError("Invalid environment")

    
    env_fn = lambda: Env(max_step=args.env_max_step, internal_goal = internal, external_goal = external, mode=args.reward_mode, 
                        per_dim_threshold=per_dim_threshold, hard=args.hard, distance_threshold=args.train_dt, n = n_blocks,
                        range_min=range_min, range_max=range_max)

    eval_env_fn = lambda: Env(max_step=50, internal_goal = internal, 
        external_goal = external, mode=args.reward_mode, compute_reward_with_internal=args.test_with_internal,
        hard=args.hard, n = n_blocks, range_min=range_min, range_max=range_max)
  return env_fn, eval_env_fn
