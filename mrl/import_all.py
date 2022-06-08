# Core
import mrl
from mrl import Agent, Module, config_to_agent

# Utils
from mrl.utils.networks import *
from mrl.utils.misc import AttrDict, add_config_args, merge_args_into_config, make_agent_name, make_activ, str2bool

# Replays
from mrl.replays.online_her_buffer import OnlineHERBuffer
from mrl.replays.prioritized_replay import EntropyPrioritizedOnlineHERBuffer
from mrl.replays.old_replay_buffer import OldReplayBuffer

# Configs
from mrl.configs.continuous_off_policy import *
from mrl.configs.discrete_off_policy import *

# Modules
from mrl.modules.action_noise import *
from mrl.modules.curiosity import *
from mrl.modules.density import *
from mrl.modules.env import EnvModule, FirstVisitDoneWrapper
from mrl.modules.eval import EpisodicEval
from mrl.modules.goal_modules import *
from mrl.modules.logging import Logger, colorize
from mrl.modules.model import PytorchModel
from mrl.modules.normalizer import *
from mrl.modules.success_prediction import GoalSuccessPredictor
from mrl.modules.train import StandardTrain

# Algorithms
from mrl.algorithms.continuous_off_policy import *
from mrl.algorithms.discrete_off_policy import QValuePolicy, DQN
from mrl.algorithms.fixed_horizon_DDPG import FHDDPG
from mrl.algorithms.random_ensemble_DPG import RandomEnsembleDPG


# Private imports if available
try:
  from private import *
except:
  print(colorize("Warning: Failed to import private modules.", color='red', bold=True))
  pass