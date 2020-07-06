import os, sys, subprocess
import numpy as np
from gym import spaces
import roboschool
from experiments.coda.pong.RoboschoolPong_v0_2017may1 import SmallReactivePolicy as Pol1
from experiments.coda.pong.pong_env import CustomPong
import time
from tqdm import tqdm

def collect_real_data(datasize, seed=0, noise_level=0.7):
  
  print("Collecting real data...")
  states = []
  actions = []
  rewards = []
  next_states = []
  dones = []

  e = CustomPong()
  np.random.seed(seed)
  e.seed(seed)
  pi = Pol1(spaces.Box(np.ones((13,))*-1, np.ones((13,))), e.action_space)

  state = e.reset()

  for i in tqdm(range(datasize)):
    if np.random.random() > noise_level:
      action = pi.act(np.concatenate((state, [0])))
    else:
      action = e.action_space.sample()
    next_state, r, done, _ = e.step(action)
    
    #e.render()
    states.append(state)
    actions.append(action)
    rewards.append(r)
    next_states.append(next_state)
    if done:
      state = e.reset()
    else:
      state = next_state

  states = np.array(states)
  actions = np.array(actions)
  rewards = np.array(rewards).reshape(-1, 1)
  next_states = np.array(next_states)
  dones = np.zeros_like(rewards)
  dataset = (states, actions, rewards, next_states, dones)
  
  print(f'Successfully collected {len(dataset[0])} real samples!')
  return dataset