import numpy as np
import torch
import pickle
import os
import glob
import shutil
from typing import Iterable, Optional, List, Union, Callable
from mrl.utils.misc import short_timestamp
from mrl.utils.misc import AttrDict


class Agent():
  """
  The base agent class. Important: Agents should almost always be generated from a config_dict
  using mrl.util.config_to_agent(config_dict). See configs folder for default configs / examples.
  
  Agent is a flat collection of mrl.Module, which may include:
    - environments (train/eval)
    - replay buffer(s)
    - new task function
    - action function  (exploratory + greedy) 
    - loss function
    - intrinsic curiosity module 
    - value / policy networks and other models (e.g. goal generation)
    - planner (e.g., MCTS)
    - logger
    - anything else you want (image tagger, human interface, etc.)

  Agent has some lifecycle methods (process_experience, optimize, save, load) that call the 
  corresponding lifecycle hooks on modules that declare them.

  Modules have a reference to the Agent so that they can access each other via the Agent. Actually,
  modules use __getattr__ to access the agent directly (via self.*), so they are effectively agent
  methods that are defined in separate files / have their own initialize/save/load functions.

  Modules are registered and saved/restored individually. This lets you swap out / tweak individual
  agent methods without subclassing the agent. Individual saves let you swap out saved modules via
  the filesystem (good for, e.g., BatchRL), avoid pickling problems from non-picklable modules.
  """

  def __init__(
      self,
      module_list: Iterable,  # list of mrl.Modules (possibly nested)
      config: AttrDict):  # hyperparameters and module settings

    self.config = config
    parent_folder = config.parent_folder
    assert parent_folder, "Setting the agent's parent folder is required!"
    self.agent_name = config.get('agent_name') or 'agent_' + short_timestamp()
    self.agent_folder = os.path.join(parent_folder, self.agent_name)
    load_agent = False
    if os.path.exists(self.agent_folder):
      print('Detected existing agent! Loading agent from checkpoint...')
      load_agent = True
    else:
      os.makedirs(self.agent_folder, exist_ok=True)

    self._process_experience_registry = []  # set of modules which define _process_experience
    self._optimize_registry = []            # set of modules which define _optimize
    self.config.env_steps = 0
    self.config.opt_steps = 0

    module_list = flatten_modules(module_list)
    self.module_dict = AttrDict()
    for module in module_list:
      assert module.module_name
      setattr(self, module.module_name, module)
      self.module_dict[module.module_name] = module
    for module in module_list:
      self._register_module(module)

    self.training = True

    if load_agent:
      self.load()
      print('Successfully loaded saved agent!')
    else:
      self.save()

  def train_mode(self):
    """Set agent to train mode; exploration / use dropout / etc. As in Pytorch."""
    self.training = True

  def eval_mode(self):
    """Set agent to eval mode; act deterministically / don't use dropout / etc."""
    self.training = False

  def process_experience(self, experience: AttrDict):
    """Calls the _process_experience function of each relevant module
    (typically, these will include a replay buffer and one or more logging modules)"""
    self.config.env_steps += self.env.num_envs if hasattr(self, 'env') else 1

    for module in self._process_experience_registry:
      module._process_experience(experience)

  def optimize(self):
    """Calls the _optimize function of each relevant module
    (typically, this will be the main algorithm; but may include others)"""
    self.config.opt_steps += 1
    for module in self._optimize_registry:
      module._optimize()

  def _register_module(self, module):
    """
    Provides module with a reference to agent so that modules can interact; e.g., 
    allows agent's policy to reference the value function.

    Then, calls each module's _setup and verify methods to _setup the module and
    verify that agent has all required modules.
    """
    self.module_dict[module.module_name] = module

    module.agent = self
    module.verify_agent_compatibility()
    module._setup()
    module.new_task()
    if hasattr(module, '_process_experience'):
      self._process_experience_registry.append(module)
    if hasattr(module, '_optimize'):
      self._optimize_registry.append(module)

  def set_module(self, module_name, module):
    """
    Sets a module (can be used to switch environments / policies)
    """
    setattr(self, module_name, module)
    self._register_module(module)

  def save(self, subfolder: Optional[str] = None):
    """
    The state of all stateful modules is saved to the agent's folder.
    The agent itself is NOT saved, and should be (1) rebuilt, and (2) restored using self.load().
    Subfolder can be used to save various checkpoints of same agent.
    """
    save_folder = self.agent_folder
    subfolder = subfolder or 'checkpoint'
    save_folder = os.path.join(save_folder, subfolder)

    if not os.path.exists(save_folder):
      os.makedirs(save_folder)

    for module in self.module_dict.values():
      module.save(save_folder)
    
    with open(os.path.join(save_folder, 'config.pickle'), 'wb') as f:
      pickle.dump(self.config, f)

  def load(self, subfolder: Optional[str] = None):
    """
    Restores state of stateful modules from the agent's folder[/subfolder].
    """
    save_folder = self.agent_folder
    subfolder = subfolder or 'checkpoint'
    save_folder = os.path.join(save_folder, subfolder)

    assert os.path.exists(save_folder), "load path does not exist!"
    
    with open(os.path.join(save_folder, 'config.pickle'), 'rb') as f:
      self.config = pickle.load(f)

    for module in self.module_dict.values():
      print("Loading module {}".format(module.module_name))
      module.load(save_folder)

  def save_checkpoint(self, checkpoint_dir):
    """
    Saves agent together with its buffer regardless of save buffer.
    Keeps 2 saves in the in folder in case the job is killed and last
    checkpoint is corrupted.

    NOTE: You should call agent.save to save to the main folder BEFORE calling this.
    """
    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)

    with open(os.path.join(checkpoint_dir, 'INITIALIZED'), 'w') as f:
      f.write('INITIALIZED')

    subfolder1 = os.path.join(checkpoint_dir, '1')
    subfolder2 = os.path.join(checkpoint_dir, '2')

    os.makedirs(os.path.join(subfolder1, 'checkpoint'), exist_ok=True)
    os.makedirs(os.path.join(subfolder2, 'checkpoint'), exist_ok=True)

    done1 = os.path.join(subfolder1, 'DONE')
    done2 = os.path.join(subfolder2, 'DONE')

    if not os.path.exists(done1):
      savedir = subfolder1
      done_file = done1
    elif not os.path.exists(done2):
      savedir = subfolder2
      done_file = done2
    else:
      modtime1 = os.path.getmtime(done1)
      modtime2 = os.path.getmtime(done2)
      if modtime1 < modtime2:
        savedir = subfolder1
        done_file = done1
      else:
        savedir = subfolder2
        done_file = done2

      os.remove(done_file)

    savedir_checkpoint = os.path.join(savedir, 'checkpoint')
    # First save all modules, including replay buffer
    old_save_replay_buf = self.config.save_replay_buf
    self.config.save_replay_buf = True
    for module in self.module_dict.values():
      module.save(savedir_checkpoint)
    self.config.save_replay_buf = old_save_replay_buf
    
    # Now save the config also
    with open(os.path.join(savedir_checkpoint, 'config.pickle'), 'wb') as f:
      pickle.dump(self.config, f)
    
    # Now copy over the config and results files from the agent_folder
    files_and_folders = glob.glob(os.path.join(self.agent_folder, '*'))
    for file_or_folder in files_and_folders:
      if os.path.isfile(file_or_folder):
        shutil.copy(file_or_folder, savedir)

    # Finally, print the DONE file.
    with open(done_file, 'w') as f:
      f.write('DONE')

  def load_from_checkpoint(self, checkpoint_dir):
    """
    This loads an agent from a checkpoint_dir to which it was saved using the `save_checkpoint` method.
    """
    subfolder1 = os.path.join(checkpoint_dir, '1')
    subfolder2 = os.path.join(checkpoint_dir, '2')
    done1 = os.path.join(subfolder1, 'DONE')
    done2 = os.path.join(subfolder2, 'DONE')

    if not os.path.exists(done1):
      assert os.path.exists(done2)
      savedir = subfolder2
    elif not os.path.exists(done2):
      savedir = subfolder1
    else:
      modtime1 = os.path.getmtime(done1)
      modtime2 = os.path.getmtime(done2)
      if modtime1 > modtime2:
        savedir = subfolder1
      else:
        savedir = subfolder2

    savedir_checkpoint = os.path.join(savedir, 'checkpoint')

    # First load the agent
    with open(os.path.join(savedir_checkpoint, 'config.pickle'), 'rb') as f:
      self.config = pickle.load(f)

    for module in self.module_dict.values():
      print("Loading module {}".format(module.module_name))
      module.load(savedir_checkpoint)

    # Then copy over the config and results file to the agent_folder
    files_and_folders = glob.glob(os.path.join(savedir, '*'))
    for file_or_folder in files_and_folders:
      if os.path.isfile(file_or_folder):
        shutil.copy(file_or_folder, self.agent_folder)


  def torch(self, x, type=torch.float):
    if isinstance(x, torch.Tensor): return x
    elif type == torch.float:
      return torch.FloatTensor(x).to(self.config.device)
    elif type == torch.long:
      return torch.LongTensor(x).to(self.config.device)
    elif type == torch.bool:
      return torch.BoolTensor(x).to(self.config.device)

  def numpy(self, x):
    return x.cpu().detach().numpy()

class Module(object):
  """
  This is the base class / module for Agent modules. Each module must inherit from it to be used
  in an Agent. 

  So that modules can be saved independently, you should access other modules only through the
  self.agent attribute and not create any new references to other modules.See the Agent.save and 
  Agent.load methods. 
  
  Note that __getattr__ passes through to the agent, so that a call to self.* is the same as a
  call to self.agent.* whenever * is not defined.  
  """

  def __init__(self, module_name: str, 
                     required_agent_modules: Optional[List[str]] = None, 
                     locals: Optional[dict] = None):
    self.module_name = module_name  # Required
    self.config_spec = locals # Optionally used to log arguments to each module (Called by Logger)
    if required_agent_modules is not None:
      self.required_agent_modules = required_agent_modules
    else:
      self.required_agent_modules = []

  def __getattr__(self, name):
    """Attribute access passes through to agent when local attribute does not exist"""
    return getattr(self.agent, name)

  def verify_agent_compatibility(self):
    """Called by agent to verify that module has everything it needs"""
    assert self.module_name is not None
    assert hasattr(self, 'agent')
    for module in self.required_agent_modules:
      assert hasattr(self.agent, module), 'Agent is missing module {}'.format(module)

  def _setup(self):
    """Called after self.agent is set to do any required _setup with other modules"""
    pass

  def new_task(self):
    """Called during _setup, and also by trainig loop if there is a task switch."""
    pass

  def save(self, save_folder: str):
    """Saves module state (note: reference to agent not available). Only some modules 
    have state that is worth saving (e.g., replays or models)"""
    pass

  def load(self, save_folder: str):
    """Restores individual module state"""
    pass

  def _save_props(self, prop_names : List[str], save_folder: str):
    """Convenience method for saving module attributes"""
    prop_dict = {prop: self.__dict__[prop] for prop in prop_names}
    with open(os.path.join(save_folder, "{}_props.pickle".format(self.module_name)), 'wb') as f:
      pickle.dump(prop_dict, f)

  def _load_props(self, prop_names : List[str], save_folder: str):
    """Convenience method for loading module attributes"""
    with open(os.path.join(save_folder, "{}_props.pickle".format(self.module_name)), 'rb') as f:
      prop_dict = pickle.load(f)
    for k, v in prop_dict.items():
      self.__dict__[k] = v

  def __str__(self):
    return self.__class__.__name__


def config_to_agent(config_dict: dict):
  module_list = []
  config = AttrDict()
  for k, v in config_dict.items():
    if is_module_or_or_module_list(v):
      module_list += flatten_modules(v)
    else:
      config[k] = v

  return Agent(module_list, config)


def is_module_or_or_module_list(item):
  if isinstance(item, Module):
    return True
  elif isinstance(item, Iterable):
    if isinstance(item, str):
      return False
    return is_module_or_or_module_list(next(iter(item)))
  else:
    return False


def flatten_modules(module_list: Union[Iterable, Module]):
  res = []
  if isinstance(module_list, Module):
    return [module_list]
  for module in module_list:
    res += flatten_modules(module)
  return res


class FunctionModule(Module):
  """
  Used to wrap functions in an mrl.Module.
  """

  def __init__(self, function: Callable, name: Optional[str] = None):
    super().__init__(name or function.__name__, required_agent_modules=[])
    self.function = function

  def __call__(self, *args, **kwargs):
    return self.function(*args, **kwargs)