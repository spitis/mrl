# Copyright (c) 2019, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: MIT
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/MIT

from envs.goalgan.ant_maze.create_maze_env import create_maze_env

import gym
import numpy as np
import torch
from gym.utils import seeding

GOAL_GRID = [np.array([3.25, 3.25]), np.array([3.25, 3.75]), np.array([3.25, 4.25]), np.array([3.25, 4.75]), np.array([3.75, 3.25]), np.array([3.75, 3.75]), 
np.array([3.75, 4.25]), np.array([3.75, 4.75]), np.array([4.25, 3.25]), np.array([4.25, 3.75]), np.array([4.25, 4.25]), np.array([4.25, 4.75]), 
np.array([4.75, 3.25]), np.array([4.75, 3.75]), np.array([4.75, 4.25]), np.array([4.75, 4.75]), np.array([1.25, 3.25]), np.array([1.25, 3.75]), 
np.array([1.25, 4.25]), np.array([1.25, 4.75]), np.array([1.75, 3.25]), np.array([1.75, 3.75]), np.array([1.75, 4.25]), np.array([1.75, 4.75]), 
np.array([2.25, 3.25]), np.array([2.25, 3.75]), np.array([2.25, 4.25]), np.array([2.25, 4.75]), np.array([2.75, 3.25]), np.array([2.75, 3.75]), 
np.array([2.75, 4.25]), np.array([2.75, 4.75]), np.array([3.25, 1.25]), np.array([3.25, 1.75]), np.array([3.25, 2.25]), np.array([3.25, 2.75]), 
np.array([3.75, 1.25]), np.array([3.75, 1.75]), np.array([3.75, 2.25]), np.array([3.75, 2.75]), np.array([4.25, 1.25]), np.array([4.25, 1.75]), 
np.array([4.25, 2.25]), np.array([4.25, 2.75]), np.array([4.75, 1.25]), np.array([4.75, 1.75]), np.array([4.75, 2.25]), np.array([4.75, 2.75]), 
np.array([-0.75,  3.25]), np.array([-0.75,  3.75]), np.array([-0.75,  4.25]), np.array([-0.75,  4.75]), np.array([-0.25,  3.25]), np.array([-0.25,  3.75]), 
np.array([-0.25,  4.25]), np.array([-0.25,  4.75]), np.array([0.25, 3.25]), np.array([0.25, 3.75]), np.array([0.25, 4.25]), np.array([0.25, 4.75]), 
np.array([0.75, 3.25]), np.array([0.75, 3.75]), np.array([0.75, 4.25]), np.array([0.75, 4.75]), np.array([ 3.25, -0.75]), np.array([ 3.25, -0.25]), 
np.array([3.25, 0.25]), np.array([3.25, 0.75]), np.array([ 3.75, -0.75]), np.array([ 3.75, -0.25]), np.array([3.75, 0.25]), np.array([3.75, 0.75]), 
np.array([ 4.25, -0.75]), np.array([ 4.25, -0.25]), np.array([4.25, 0.25]), np.array([4.25, 0.75]), np.array([ 4.75, -0.75]), np.array([ 4.75, -0.25]), 
np.array([4.75, 0.25]), np.array([4.75, 0.75]), np.array([ 1.25, -0.75]), np.array([ 1.25, -0.25]), np.array([1.25, 0.25]), np.array([1.25, 0.75]), 
np.array([ 1.75, -0.75]), np.array([ 1.75, -0.25]), np.array([1.75, 0.25]), np.array([1.75, 0.75]), np.array([ 2.25, -0.75]), np.array([ 2.25, -0.25]), 
np.array([2.25, 0.25]), np.array([2.25, 0.75]), np.array([ 2.75, -0.75]), np.array([ 2.75, -0.25]), np.array([2.75, 0.25]), np.array([2.75, 0.75]), 
np.array([-0.75, -0.75]), np.array([-0.75, -0.25]), np.array([-0.75,  0.25]), np.array([-0.75,  0.75]), np.array([-0.25, -0.75]), np.array([-0.25, -0.25]), 
np.array([-0.25,  0.25]), np.array([-0.25,  0.75]), np.array([ 0.25, -0.75]), np.array([ 0.25, -0.25]), np.array([0.25, 0.25]), np.array([0.25, 0.75]), 
np.array([ 0.75, -0.75]), np.array([ 0.75, -0.25]), np.array([0.75, 0.25]), np.array([0.75, 0.75])]

class AntMazeEnv(gym.GoalEnv):
  """Wraps the GoalGan Env in a gym goal env."""
  def __init__(self, eval=False):


    self.done_env = False
    self.dist_threshold = 0.5
    state_dims = 30
    
    self.goal_dims = [0, 1]
    self.eval_dims = [0, 1]
    if eval:
      self.done_env = True

    self.maze = create_maze_env('AntMaze') # this returns a gym environment
    self.seed()
    self.max_steps = 500

    self.action_space = self.maze.action_space
    observation_space = gym.spaces.Box(-np.inf, np.inf, (state_dims,)) 
    goal_space        = gym.spaces.Box(-np.inf, np.inf, (len(self.goal_dims),)) # first few coords of state
    self.observation_space = gym.spaces.Dict({
        'observation': observation_space,
        'desired_goal': goal_space,
        'achieved_goal': goal_space
    })
    self.num_steps = 0

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    self.maze.wrapped_env.seed(seed)
    return [seed]

  def sample_goal(self):
    idx = self.np_random.choice(len(GOAL_GRID))
    return GOAL_GRID[idx].astype(np.float32)

  def step(self, action):
    next_state, _, _, _ = self.maze.step(action)
    next_state = next_state.astype(np.float32)

    s_xy = next_state[self.goal_dims]
    reward = self.compute_reward(s_xy, self.g_xy, None)
    info = {}
    self.num_steps += 1
    
    is_success = np.allclose(0., reward)
    done = is_success and self.done_env
    info['is_success'] = is_success
    if self.num_steps >= self.max_steps and not done:
      done = True
      info['TimeLimit.truncated'] = True

    obs = {
      'observation': next_state,
      'achieved_goal': s_xy,
      'desired_goal': self.g_xy,
    }
          
    return obs, reward, done, info

  def reset(self): 
    self.num_steps = 0

    ## Not exactly sure why we are reseeding here, but it's what the SR env does
    _ = self.maze.wrapped_env.seed(self.np_random.randint(np.iinfo(np.int32).max))
    s = self.maze.reset().astype(np.float32)
    _ = self.maze.wrapped_env.seed(self.np_random.randint(np.iinfo(np.int32).max))
    self.g_xy = self.sample_goal()

    return {
      'observation': s,
      'achieved_goal': s[self.goal_dims],
      'desired_goal': self.g_xy,
    }

  def render(self):
    self.maze.render()

  def compute_reward(self, achieved_goal, desired_goal, info):
    if len(achieved_goal.shape) == 2:
      ag = achieved_goal[:,self.eval_dims]
      dg = desired_goal[:,self.eval_dims]
    else:
      ag = achieved_goal[self.eval_dims]
      dg = desired_goal[self.eval_dims]
    d = np.linalg.norm(ag - dg, axis=-1)
    return -(d >= self.dist_threshold).astype(np.float32)