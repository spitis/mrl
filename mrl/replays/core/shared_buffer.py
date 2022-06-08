import numpy as np
import multiprocessing as mp
from mrl.replays.core.replay_buffer import RingBuffer
from mrl.utils.misc import AttrDict
from multiprocessing import RawArray, RawValue
from collections import OrderedDict
from numpy.random import default_rng

BUFF = None

def worker_init(buffer : AttrDict):
  global BUFF
  BUFF = buffer

def future_samples(idxs):
  """Assumes there is an 'ag' field, and samples n transitions, pairing each with a future ag
  from its trajectory."""
  global BUFF
  rng = default_rng()

  # get original transitions
  transition = []
  for buf in BUFF.items:
    item = BUFF[buf].get_batch(idxs)
    transition.append(item)

  # add random future goals
  tlefts = BUFF.buffer_tleft.get_batch(idxs)
  idxs = idxs + (rng.uniform(size=len(idxs)) * tlefts).round().astype(np.int64)
  ags = BUFF.buffer_ag.get_batch(idxs)
  transition.append(ags)
  return transition


def actual_samples(idxs):
  """Assumes there is an 'dg' field, and samples n transitions, pairing each with a random dg"""
  global BUFF

  # get original transitions
  transition = []
  for buf in BUFF.items:
    item = BUFF[buf].get_batch(idxs)
    transition.append(item)

  # add random actual goals
  idxs = np.random.choice(len(BUFF.buffer_tidx), len(idxs))
  dgs = BUFF.buffer_dg.get_batch(idxs)
  transition.append(dgs)

  return transition


def achieved_samples(idxs):
  """Assumes there is an 'ag' field, and samples n transitions, pairing each with a random ag"""
  global BUFF

  # get original transitions
  transition = []
  for buf in BUFF.items:
    item = BUFF[buf].get_batch(idxs)
    transition.append(item)

  # add random achieved goals
  idxs = np.random.choice(len(BUFF.buffer_tidx), len(idxs))
  ags = BUFF.buffer_ag.get_batch(idxs)
  transition.append(ags)

  return transition


def behavioral_samples(idxs):
  """Assumes there is an 'bg' field, and samples n transitions, pairing each with a random bg"""
  global BUFF

  # get original transitions
  transition = []
  for buf in BUFF.items:
    item = BUFF[buf].get_batch(idxs)
    transition.append(item)

  # add random achieved goals
  idxs = np.random.choice(len(BUFF.buffer_tidx), len(idxs))
  bgs = BUFF.buffer_bg.get_batch(idxs)
  transition.append(bgs)

  return transition


def n_step_samples(args):
  """Samples s_t and s_{t+n}, along with the discounted reward in between.
  Assumes buffers include: state, action, reward, next_state, done
  Because some sampled states will not have enough future states, this will
  sometimes return less than num_samples samples.
  """
  state_idxs, n_steps, gamma = args

  global BUFF

  tlefts = BUFF.buffer_tleft.get_batch(state_idxs)

  # prune idxs for which there are not enough future transitions
  good_state_idxs = state_idxs[tlefts >= n_steps - 1]
  potentially_bad_idxs = state_idxs[tlefts < n_steps - 1] # 0 tleft corresp to 1 step
  potentially_bad_delt = tlefts[tlefts < n_steps - 1]
  also_good_state_idxs = potentially_bad_idxs[np.isclose(BUFF.buffer_done.get_batch(potentially_bad_idxs + potentially_bad_delt), 1).reshape(-1)]

  state_idxs = np.concatenate((good_state_idxs, also_good_state_idxs), axis=0)
  t_idxs = BUFF.buffer_tidx.get_batch(state_idxs)

  # start building the transition, of state, action, n_step reward, n_step next_state, done
  transition = [BUFF.buffer_state.get_batch(state_idxs), BUFF.buffer_action.get_batch(state_idxs)]
  rewards = np.zeros_like(state_idxs, dtype=np.float32).reshape(-1, 1)

  for i in range(n_steps):
    query = state_idxs + i
    r_delta = (gamma ** i) * BUFF.buffer_reward.get_batch(query)
    diff_traj = t_idxs != BUFF.buffer_tidx.get_batch(query)
    r_delta[diff_traj] *= 0.
    rewards += r_delta

  transition.append(rewards)
  transition.append(BUFF.buffer_next_state.get_batch(query)) # n_step state

  dones = BUFF.buffer_done.get_batch(query)
  dones += diff_traj.astype(np.float32).reshape(-1, 1)
  transition.append(dones)

  return transition



# def stacked_samples(args):
#   """Samples the state and next state as a sequence of states."""
#   num_samples, num_previous_states = args

#   global BUFF
#   base_idxs = np.random.choice(len(BUFF.buffer_tidx), n)
#   idxs = [base_idxs] + [base_idxs - (i + 1) for i in range(num_previous_states)]

#   # get original transitions
#   transition = []
#   for buf in BUFF.items:
#     if buf == 'buffer_state':
#       item =
#     item = BUFF[buf].get_batch(idxs)
#     transition.append(item)

#   # add random future goals
#   tlefts = BUFF.buffer_tleft.get_batch(idxs)
#   idxs = idxs + np.round(np.random.uniform(size=n) * tlefts).astype(np.int64)
#   ags = BUFF.buffer_ag.get_batch(idxs)
#   transition.append(ags)
#   return transition

class SharedMemoryTrajectoryBuffer():
  """
  An alternative replay buffer object that stores contiguous trajectories in a shared memory buffer.
  This should allow for expensive queries (like online hindsight relabeling) to be executed in parallel.

  NOTE: even hindsight relabeling is only faster with Multiprocessing when using minibatches of 5000+.
  E.g.:
    - for sample size 10K:, sample_actual = 1.36ms vs 2.34 (MP vs single proc); future = 1.69ms vs 2.31ms
    - for sample size 500:, sample_actual = 420us vs 220us (MP vs single proc); future = 390us vs 160us

  So probably should just leave it on n_cpu = 1, unless there is something really expensive to do, like
  computing statistics of whole trajectories, or searching forward through the buffer.
  """

  def __init__(self, limit, item_shape, n_cpu=1):
    """
    The replay buffer object. Stores everything in float32.

    :param limit: (int) the max number of transitions to store
    :param item_shape: a list of tuples of (str) item name and (tuple) the shape for item
      Ex: [("observations", env.observation_space.shape),\
          ("actions",env.action_space.shape),\
          ("rewards", (1,)),\
          ("dones", (1,))]
    """
    self.limit = limit

    global BUFF
    BUFF = AttrDict()
    self.BUFF = BUFF # a global object that has shared RawArray-based RingBuffers.

    BUFF.items = []

    # item buffers
    for name, shape in item_shape:
      BUFF.items.append('buffer_' + name)
      BUFF['raw_' + name] = RawArray('f', int(np.prod((limit, ) + shape)))
      BUFF['np_' + name] =\
        np.frombuffer(BUFF['raw_' + name], dtype=np.float32).reshape((limit, ) + shape)
      BUFF['buffer_' + name] = RingBuffer(limit, shape=shape, data=BUFF['np_' + name])

    # special buffers
    BUFF.raw_tidx = RawArray('d', limit)
    BUFF.np_tidx = np.frombuffer(BUFF.raw_tidx, dtype=np.int64)
    BUFF.buffer_tidx = RingBuffer(limit, shape=(), dtype=np.int64, data=BUFF.np_tidx)

    BUFF.raw_tleft = RawArray('d', limit)
    BUFF.np_tleft = np.frombuffer(BUFF.raw_tleft, dtype=np.int64)
    BUFF.buffer_tleft = RingBuffer(limit, shape=(), dtype=np.int64, data=BUFF.np_tleft)

    if 'buffer_bg' in BUFF: # is this a successful trajectory?
      BUFF.raw_success = RawArray('f', limit)
      BUFF.np_success = np.frombuffer(BUFF.raw_success, dtype=np.float32)
      BUFF.buffer_success = RingBuffer(limit, shape=(), dtype=np.float32, data=BUFF.np_success)

    self.trajectories = OrderedDict() # a centralized dict of trajectory_id --> trajectory_idxs
    self.total_trajectory_len = 0
    self.current_trajectory = 0

    self.pool = None
    self.n_cpu = n_cpu
    if n_cpu > 1:
      self.pool = mp.Pool(n_cpu, initializer=worker_init, initargs=(BUFF,))

  def add_trajectory(self, *items):
    """
    Append a trajectory of transitions to the buffer.

    :param items: a list of batched transition values to append to the replay buffer,
        in the item order that we initialized the ReplayBuffer with.
    """
    trajectory_len = len(items[0])

    for buffer_name, batched_values in zip(self.BUFF.items, items):
      self.BUFF[buffer_name].append_batch(batched_values)

    # add to special buffers
    self.BUFF.buffer_tleft.append_batch(np.arange(trajectory_len, dtype=np.int64)[::-1])
    idxs = self.BUFF.buffer_tidx.append_batch(np.ones((trajectory_len,), dtype=np.int64) * self.current_trajectory)

    # Keep track of successes, if has behavioral goals.
    if 'buffer_bg' in BUFF:
      for buffer_name, batched_values in zip(self.BUFF.items, items):
        if buffer_name == 'buffer_reward':
          success = np.any(np.isclose(batched_values, 0.))
          if success:
            self.BUFF.buffer_success.append_batch(np.ones((trajectory_len,), dtype=np.float32))
          else:
            self.BUFF.buffer_success.append_batch(np.zeros((trajectory_len,), dtype=np.float32))

    self.trajectories[self.current_trajectory] = idxs
    self.total_trajectory_len += trajectory_len

    # remove trajectories until all remaining trajectories fit in the buffer.
    while self.total_trajectory_len > self.limit:
      _, idxs = self.trajectories.popitem(last=False)
      self.total_trajectory_len -= len(idxs)
    self.current_trajectory += 1

  def sample_trajectories(self, n, group_by_buffer=False, from_m_most_recent=None):
    """
    Samples n full trajectories (optionally from 'from_m_most_recent' trajectories)
    """

    if from_m_most_recent is not None and (len(self.trajectories) - n >= 0):
      min_idx = max(self.current_trajectory - len(self.trajectories), self.current_trajectory - from_m_most_recent)
      idxs = np.random.randint(min_idx, self.current_trajectory, n)
    else:
      idxs = np.random.randint(self.current_trajectory - len(self.trajectories), self.current_trajectory, n)
    queries = [self.trajectories[i] for i in idxs]
    splits = np.cumsum([len(q) for q in queries[:-1]])
    query = np.concatenate(queries)
    transition = []
    for buf in BUFF.items:
      transition.append(np.split(BUFF[buf].get_batch(query), splits))

    if group_by_buffer:
      return transition

    return list(zip(*transition))

  def sample(self, batch_size, batch_idxs=None):
    """
    sample a random batch from the buffer

    :param batch_size: (int) the number of element to sample for the batch
    :return: (list) the sampled batch
    """
    if self.size == 0:
      return []

    if batch_idxs is None:
      batch_idxs = np.random.randint(self.size, size=batch_size)

    transition = []
    for buf in BUFF.items:
      item = BUFF[buf].get_batch(batch_idxs)
      transition.append(item)

    return transition

  def sample_slices(self, batch_size, slice_size):
    """Tries to sample slices of length slice_size randomly. Slices must be
    from same trajectory, which may not happen even when oversampled, so it's possible
    (but unlikely, at least for small slice_size) to get a small batch size than requested.
    """
    if self.size == 0:
      return [[] for _ in range(slice_size)]

    b_idxs = np.random.randint(self.size, size=int(batch_size * 1.5))
    first_t = BUFF.buffer_tidx.get_batch(b_idxs - slice_size + 1)
    last_t  = BUFF.buffer_tidx.get_batch(b_idxs)

    batch_idxs = b_idxs[first_t == last_t][:batch_size]

    transitions = []
    for i in range(-slice_size + 1, 1):
      transitions.append([])
      for buf in BUFF.items:
        item = BUFF[buf].get_batch(batch_idxs + i)
        transitions[-1].append(item)

    return transitions

  def sample_n_step_transitions(self, batch_size, n_steps, gamma, batch_idxs=None):
    if batch_idxs is None:
      batch_idxs = np.random.randint(self.size, size=batch_size)
      
    if self.pool is not None:
      res = self.pool.map(n_step_samples, zip(np.array_split(batch_idxs, self.n_cpu), [n_steps] * self.n_cpu, [gamma] * self.n_cpu))
      res = [np.concatenate(x, 0) for x in zip(*res)]
      return res

    return n_step_samples((batch_idxs, n_steps, gamma))

  def sample_future(self, batch_size, batch_idxs=None):
    if batch_idxs is None:
      batch_idxs = np.random.randint(self.size, size=batch_size)

    if self.pool is not None:
      res = self.pool.map(future_samples, np.array_split(batch_idxs, self.n_cpu))
      res = [np.concatenate(x, 0) for x in zip(*res)]
      return res

    return future_samples(batch_idxs)

  def sample_from_goal_buffer(self, buffer, batch_size, batch_idxs=None):
    """buffer is one of 'ag', 'dg', 'bg'"""
    if buffer == 'ag':
      sample_fn = achieved_samples
    elif buffer == 'dg':
      sample_fn = actual_samples
    elif buffer == 'bg':
      sample_fn = behavioral_samples
    else:
      raise NotImplementedError

    if batch_idxs is None:
      batch_idxs = np.random.randint(self.size, size=batch_size)

    if self.pool is not None:
      res = self.pool.map(sample_fn, np.array_split(batch_idxs, self.n_cpu))
      res = [np.concatenate(x, 0) for x in zip(*res)]
      return res

    return sample_fn(batch_idxs)

  def __len__(self): return self.size

  def _get_state(self): 
    d = dict(
      trajectories = self.trajectories,
      total_trajectory_len = self.total_trajectory_len,
      current_trajectory = self.current_trajectory
    )
    for bufname in self.BUFF.items + ['buffer_tleft', 'buffer_tidx', 'buffer_success']:
      if bufname in self.BUFF:
        d[bufname] = self.BUFF[bufname]._get_state()
    return d

  def _set_state(self, d):
    for k in ['trajectories','total_trajectory_len','current_trajectory']:
      self.__dict__[k] = d[k]
    for bufname in self.BUFF.items + ['buffer_tleft', 'buffer_tidx', 'buffer_success']:
      if bufname in self.BUFF:
        self.BUFF[bufname]._set_state(*d[bufname])

  def close(self):
    if self.pool is not None:
      self.pool.close()

  @property
  def size(self):
    return len(BUFF.buffer_tidx)

  @property
  def num_trajectories(self):
    return len(self.trajectories)
