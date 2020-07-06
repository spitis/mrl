from gym.spaces import Discrete, Box
import numpy as np

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def discrete_to_box_wrapper(env, bound=4.):
  """takes a discrete environment, and turns it into a box environment"""
  assert isinstance(env.action_space, Discrete), "must pass a discrete environment!"
  old_step = env.step
  n = env.action_space.n
  env.action_space = Box(low = -bound, high = bound, shape=(n,))

  def step(action):
    action = np.clip(action, -bound, bound)
    action = softmax(action)
    action = np.random.choice(range(n), p=action)
    
    return old_step(action)

  env.step = step

  return env