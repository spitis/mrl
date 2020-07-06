import mrl
import numpy as np

# temporary fix for dumb tensorflow / tensorboard error. https://github.com/pytorch/pytorch/issues/30966
try:
  import tensorflow as tf
  import tensorboard as tb
  tf.io.gfile = tb.compat.tensorflow_stub.io.gfile
except:
  import tensorboard as tb

from torch.utils.tensorboard import SummaryWriter
from tabulate import tabulate
from collections import defaultdict
import json
import os
import time
import csv

class Logger(mrl.Module):
  """
  Logger that processes vectorized experiences and 
  records results to the console. 
  """
  def __init__(self, average_every=100):
    super().__init__('logger', required_agent_modules=['env'], locals=locals())
    self.average_every = average_every
    self.writer = None

  def _setup(self):
    # rewards and steps are always tracked
    self.rewards_per_env = np.zeros((self.env.num_envs, ))
    self.steps_per_env = np.zeros((self.env.num_envs, ))
    self.episode_rewards = []
    self.episode_steps = []
    self.steps = 0
    self.episodes = 0
    self.tabular = defaultdict(list)
    self.last_log_step = defaultdict(int)
    self.log_every_n_steps = self.config.log_every
    self.save_config()

  def lazy_init_writer(self):
    if self.writer is None:
      self.writer = SummaryWriter(self.agent_folder)

  def update_csv(self, tag, value, step):
    fields = ['wall_time', 'step', tag]
    path = os.path.join(self.agent_folder, self.agent_name + '__' + tag.replace('/', '__') + '.csv')
    if not os.path.exists(path):
      with open(path, 'w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(fields)
    with open(path, 'a') as f:
      writer = csv.writer(f)
      writer.writerow([time.time(), step, value])

  def add_scalar(self, tag, value, log_every=1000, step=None):
    """Adds scalar to tensorboard"""
    self.lazy_init_writer()
    if step is None:
      step = self.config.env_steps
    if step - self.last_log_step[tag] >= log_every:
      self.last_log_step[tag] = step
      self.writer.add_scalar(tag, value, step)
      self.update_csv(tag, value, step)

  def add_histogram(self, tag, values, log_every=1000, step=None, **kwargs):
    """Adds histogram to tensorboard"""
    self.lazy_init_writer()
    if isinstance(values, list):
      values = np.array(values, dtype=np.float32)
    elif isinstance(values, np.ndarray):
      values = values.astype(np.float32)
    if step is None:
      step = self.config.env_steps
    if step - self.last_log_step[tag] >= log_every:
      self.last_log_step[tag] = step
      self.writer.add_histogram(tag, values, step, **kwargs)

  def add_embedding(self, tag, values, log_every=1000, step=None, **kwargs):
    """Adds embedding data to tensorboard"""
    self.lazy_init_writer()
    if isinstance(values, list):
      values = np.array(values, dtype=np.float32)
    elif isinstance(values, np.ndarray):
      values = values.astype(np.float32)
    assert len(values.shape) == 2
    if step is None:
      step = self.config.env_steps
    if step - self.last_log_step[tag] >= log_every:
      self.last_log_step[tag] = step
      self.writer.add_embedding(mat=values, tag=tag, global_step=step, **kwargs)

  def add_tabular(self, tag, value):
    """Adds scalar to console logger"""
    self.tabular[tag].append(value)

  def log_color(self, tag, value='', color='cyan'):
    print(colorize(tag, color=color, bold=True), value)

  def save_config(self):
    config_json = convert_json({**self.config, **record_attrs(self.module_dict.values())})
    config_json['agent_name'] = self.agent_name
    output = json.dumps(config_json, separators=(',', ':\t'), indent=4, sort_keys=True)
    print(colorize('\nAgent folder:', color='magenta', bold=True))
    print(self.agent_folder)
    print(colorize('\nSaving config:', color='cyan', bold=True))
    print(output)
    with open(os.path.join(self.agent_folder, "config.json"), 'w') as out:
      out.write(output)

  def flush_console(self):
    table = [('Environment steps', self.steps), ('Total episodes', self.episodes),
             ('Avg rewards (last {})'.format(self.average_every), np.mean(self.episode_rewards[-self.average_every:])),
             ('Avg episode len (last {})'.format(self.average_every), np.mean(self.episode_steps[-self.average_every:]))
             ]
    for k, v in self.tabular.items():
      table.append(('Avg ' + k + ' (last {})'.format(self.average_every), np.mean(v[-self.average_every:])))
    table = tabulate(table, headers=['Tag', 'Value'], tablefmt="psql", floatfmt="8.1f")

    print(table)

  def _process_experience(self, experience):
    rewards, dones = experience.reward, experience.trajectory_over

    self.rewards_per_env += rewards
    self.steps_per_env += 1

    if np.any(dones):
      self.episode_rewards += list(self.rewards_per_env[dones])
      self.episode_steps += list(self.steps_per_env[dones])
      self.rewards_per_env[dones] = 0
      self.steps_per_env[dones] = 0
      self.episodes += np.sum(dones)

    self.steps += self.env.num_envs
    if self.steps % self.log_every_n_steps < self.env.num_envs:
      self.flush_console()
      self.add_scalar('Train/Episode_rewards', np.mean(self.episode_rewards[-30:]))
      self.add_scalar('Train/Episode_steps', np.mean(self.episode_steps[-30:]))

  def save(self, save_folder):
    self._save_props([
        'episode_rewards', 'episode_steps',
        'steps', 'episodes', 'tabular', 'last_log_step'
    ], save_folder)

  def load(self, save_folder):
    self._load_props([
        'episode_rewards', 'episode_steps',
        'steps', 'episodes', 'tabular', 'last_log_step'
    ], save_folder)


color2num = dict(gray=30, red=31, green=32, yellow=33, blue=34, magenta=35, cyan=36, white=37, crimson=38)


def colorize(string, color, bold=False, highlight=False):
  """
    Colorize a string.
    This function was originally written by John Schulman.
    """
  attr = []
  num = color2num[color]
  if highlight: num += 10
  attr.append(str(num))
  if bold: attr.append('1')
  return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)


def convert_json(obj, dict_to_str=False):
  """ Convert obj to a version which can be serialized with JSON. """
  if is_json_serializable(obj):
    return obj
  else:
    if isinstance(obj, dict) and not dict_to_str:
      return {convert_json(k): convert_json(v) for k, v in obj.items()}

    elif isinstance(obj, tuple):
      return tuple([convert_json(x) for x in obj])

    elif isinstance(obj, list):
      return [convert_json(x) for x in obj]

    elif hasattr(obj, '__name__') and not ('lambda' in obj.__name__):
      return convert_json(obj.__name__)

    elif hasattr(obj, '__dict__') and obj.__dict__:
      obj_dict = {convert_json(k): convert_json(v) for k, v in obj.__dict__.items()}
      return {str(obj): obj_dict}

    return str(obj)


def is_json_serializable(v):
  try:
    json.dumps(v)
    return True
  except:
    return False


def record_attrs(module_list):
  res = {}
  for module in module_list:
    res['module_' + module.module_name] = convert_json(strip_config_spec(module.config_spec), dict_to_str=True)
  return res


def strip_config_spec(config_spec):
  if '__class__' in config_spec:
    del config_spec['__class__']
  return config_spec