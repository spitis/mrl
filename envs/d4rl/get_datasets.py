"""
First, follow install instructions here: https://github.com/rail-berkeley/d4rl
(First commit requires clone and local install.)
"""

import gym
import d4rl # Import required to register environments

DATASET_NAMES = [
  'halfcheetah-random-v0',
  'halfcheetah-medium-v0',
  'halfcheetah-expert-v0',
  'halfcheetah-medium-replay-v0',
  'halfcheetah-medium-expert-v0',
  'walker2d-random-v0',
  'walker2d-medium-v0',
  'walker2d-expert-v0',
  'walker2d-medium-replay-v0',
  'walker2d-medium-expert-v0',
  'hopper-random-v0',
  'hopper-medium-v0',
  'hopper-expert-v0',
  'hopper-medium-replay-v0',
  'hopper-medium-expert-v0',
  'ant-random-v0',
  'ant-medium-v0',
  'ant-expert-v0',
  'ant-medium-replay-v0',
  'ant-medium-expert-v0'
]

if __name__ == '__main__':
  for env in DATASET_NAMES:
    e = gym.make(env)
    d = e.get_dataset()
    print(f'Got dataset for {env}! It has shape {d["observations"].shape}!')