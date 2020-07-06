import numpy as np
from collections import OrderedDict
from mrl.utils.misc import AttrDict
from multiprocessing import RawValue

class RingBuffer(object):
  """This is a collections.deque in numpy, with pre-allocated memory"""

  def __init__(self, maxlen, shape, dtype=np.float32, data=None):
    """
    A buffer object, when full restarts at the initial position

    :param maxlen: (int) the max number of numpy objects to store
    :param shape: (tuple) the shape of the numpy objects you want to store
    :param dtype: (str) the name of the type of the numpy object you want to store
    """
    self.maxlen = maxlen
    self.start = RawValue('L')
    self.length = RawValue('L')
    self.shape = shape
    if data is None:
      self.data = np.zeros((maxlen, ) + shape, dtype=dtype)
    else:
      assert data.shape == (maxlen, ) + shape
      assert data.dtype == dtype
      self.data = data

  def _get_state(self):
    # Only restore the values in the data
    end_idx = self.start.value + self.length.value
    indices = range(self.start.value, end_idx)
    return self.start.value, self.length.value, self.data.take(indices, axis=0, mode='wrap')

  def _set_state(self, start, length, data):
    self.start.value = start
    self.length.value = length
    self.data[:length] = data
    self.data = np.roll(self.data, start, axis=0)

  def __len__(self):
    return self.length.value

  def __getitem__(self, idx):
    if idx < 0 or idx >= self.length.value:
      raise KeyError()
    return self.data[(self.start.value + idx) % self.maxlen]

  def get_batch(self, idxs):
    """
    get the value at the indexes

    :param idxs: (int or numpy int) the indexes
    :return: (np.ndarray) the stored information in the buffer at the asked positions
    """
    return self.data[(self.start.value + idxs) % self.length.value]

  def append(self, var):
    """
    Append an object to the buffer

    :param var: (np.ndarray) the object you wish to add
    """
    if self.length.value < self.maxlen:
      # We have space, simply increase the length.
      self.length.value += 1
    elif self.length.value == self.maxlen:
      # No space, "remove" the first item.
      self.start.value = (self.start.value + 1) % self.maxlen
    else:
      # This should never happen.
      raise RuntimeError()

    self.data[(self.start.value + self.length.value - 1) % self.maxlen] = var

  def _append_batch_with_space(self, var):
    """
    Append a batch of objects to the buffer, *assuming* there is space.

    :param var: (np.ndarray) the batched objects you wish to add
    """
    len_batch = len(var)
    start_pos = (self.start.value + self.length.value) % self.maxlen

    self.data[start_pos : start_pos + len_batch] = var
    
    if self.length.value < self.maxlen:
      self.length.value += len_batch
      assert self.length.value <= self.maxlen, "this should never happen!"
    else:
      self.start.value = (self.start.value + len_batch) % self.maxlen

    return np.arange(start_pos, start_pos + len_batch)

  def append_batch(self, var):
    """
    Append a batch of objects to the buffer.

    :param var: (np.ndarray) the batched objects you wish to add
    """
    len_batch = len(var)
    assert len_batch < self.maxlen, 'trying to add a batch that is too big!'
    start_pos = (self.start.value + self.length.value) % self.maxlen
    
    if start_pos + len_batch <= self.maxlen:
      # If there is space, add it
      idxs = self._append_batch_with_space(var)
    else:
      # No space, so break it into two batches for which there is space
      first_batch, second_batch = np.split(var, [self.maxlen - start_pos])
      idxs1 = self._append_batch_with_space(first_batch)
      # use append on second call in case len_batch > self.maxlen
      idxs2 = self._append_batch_with_space(second_batch)
      idxs = np.concatenate((idxs1, idxs2))
    return idxs

class ReplayBuffer(object):
  def __init__(self, limit, item_shape, dtypes=None):
    """
    The replay buffer object

    :param limit: (int) the max number of transitions to store
    :param item_shape: a list of tuples of (str) item name and (tuple) the shape for item
      Ex: [("observations0", env.observation_space.shape),\
          ("actions",env.action_space.shape),\
          ("rewards", (1,)),\
          ("observations1",env.observation_space.shape ),\
          ("terminals1", (1,))]
    :param dtypes: list of dtype tuples; useful for storing things as float16.
    """
    self.limit = limit

    self.items = OrderedDict()
    if dtypes is None:
      dtypes = [(np.float32, np.float32)] * len(item_shape)

    self.in_types, self.out_types = zip(*dtypes)

    for (name, shape), dtype in zip(item_shape, self.in_types):
      self.items[name] = RingBuffer(limit, shape=shape, dtype=dtype)

  def sample(self, batch_size):
    """
    sample a random batch from the buffer

    :param batch_size: (int) the number of element to sample for the batch
    :return: (list) the sampled batch
    """
    if self.size==0:
      return []

    batch_idxs = np.random.randint(self.size, size=batch_size)

    transition = []
    for buf, dtype in zip(self.items.values(), self.out_types):
      item = buf.get_batch(batch_idxs).astype(dtype)
      transition.append(item)

    return transition

  def add(self, *items):
    """
    Appends a single transition to the buffer

    :param items: a list of values for the transition to append to the replay buffer,
        in the item order that we initialized the ReplayBuffer with.
    """
    for buf, value in zip(self.items.values(), items):
      buf.append(value)

  def add_batch(self, *items):
    """
    Append a batch of transitions to the buffer.

    :param items: a list of batched transition values to append to the replay buffer,
        in the item order that we initialized the ReplayBuffer with.
    """
    if (items[0].shape) == 1 or len(items[0]) == 1:
      self.add(*items)
      return

    for buf, batched_values in zip(self.items.values(), items):
      idxs = buf.append_batch(batched_values)

    return idxs

  def __len__(self):
    return self.size

  def _get_state(self):
    d = dict()
    for item, buf in self.items.items():
      d[item] = buf._get_state()
    return d

  def _set_state(self, d):
    for item, buf in self.items.items():
      buf._set_state(*d[item])

  @property
  def size(self):
    # Get the size of the RingBuffer on the first item type
    return len(next(iter(self.items.values())))
