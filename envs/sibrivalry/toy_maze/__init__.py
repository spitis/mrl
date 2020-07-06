# Copyright (c) 2019, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: MIT
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/MIT

from .maze_env import Env
import gym
import numpy as np
import torch
import matplotlib


class Obstacle:
  def __init__(self, top_left, bottom_right):
    self.top_left = top_left
    self.bottom_right = bottom_right

  def in_collision(self, points):
    if len(points.shape) == 1:
      return self.top_left[0] <= points[0] <= self.bottom_right[0] and \
           self.bottom_right[1] <= points[1] <= self.top_left[1]
    else:
      return np.logical_and(np.logical_and(np.logical_and(
        self.top_left[0] <= points[:,0], points[:,0] <= self.bottom_right[0]),
        self.bottom_right[1] <= points[:,1]), 
        points[:,1] <= self.top_left[1])


  def get_patch(self):
    # Create a Rectangle patch
    rect = matplotlib.patches.Rectangle((self.top_left[0], self.bottom_right[1]),
                                        self.bottom_right[0] - self.top_left[0],
                                        self.top_left[1] - self.bottom_right[1],
                                        linewidth=1,
                                        edgecolor='k',
                                        hatch='x',
                                        facecolor='none')
    return rect


class SimpleMazeEnv(gym.GoalEnv):
  """This is a long horizon (80+ step optimal trajectories), but also very simple point navigation task"""
  def __init__(self, test=False):
    self.pos_min = 0.0
    self.pos_max = 1.0
    self.dx = dx = 0.025
    self.dy = dy = 0.025
    s2 = 1/np.sqrt(2)*dx
    self.action_vec = np.array([[dx, 0], [-dx, 0], [0, dy], [0, -dy], [s2, s2], [s2, -s2], [-s2, s2], [-s2, -s2]])
    self.dist_threshold = 0.10
    self.obstacles = [Obstacle([0.2, 0.8], [0.4, 0]), Obstacle([0.6, 1], [0.8, 0.2])]
    self.action_space = gym.spaces.Discrete(8)
    observation_space = gym.spaces.Box(-np.inf, np.inf, (2, ))
    goal_space = gym.spaces.Box(-np.inf, np.inf, (2, ))
    self.observation_space = gym.spaces.Dict({
      'observation': observation_space,
      'desired_goal': goal_space,
      'achieved_goal': goal_space
    })

    self.s_xy = self.get_free_point()
    self.g_xy = self.get_free_point()

    self.max_steps = 250
    self.num_steps = 0
    self.test = test

  def seed(self, seed=None):
    np.random.seed(seed)

  def compute_reward(self, achieved_goal, desired_goal, info):
    d = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
    goal_rew = -(d >= self.dist_threshold).astype(np.float32)
    coll_rew = self.in_collision(achieved_goal) * -20
    return goal_rew + coll_rew
  
  def render(self):
    raise NotImplementedError

  def step(self, action):
    #d_pos = action / np.linalg.norm(action) * self.dx
    d_pos = self.action_vec[action]
    self.s_xy = np.clip(self.s_xy + d_pos, self.pos_min, self.pos_max)
    reward = self.compute_reward(self.s_xy, self.g_xy, None)

    info = {}
    self.num_steps += 1

    if self.test:
      done = np.allclose(0., reward)
      info['is_success'] = done
    else:
      done = False
      info['is_success'] = np.allclose(0., reward)

    if self.num_steps >= self.max_steps and not done:
      done = True
      info['TimeLimit.truncated'] = True

    obs = {
        'observation': self.s_xy,
        'achieved_goal': self.s_xy,
        'desired_goal': self.g_xy,
    }

    return obs, reward, done, info
      
  def in_collision(self, points):
    return np.any([x.in_collision(points) for x in self.obstacles], axis=0)

  def get_free_point(self):
    max_tries = 100
    point = np.random.rand(2)
    tries = 0
    while self.in_collision(point):
      point = np.random.rand(2)
      tries += 1
      if tries >= max_tries:
        return None
    return point

  def reset(self):
    self.num_steps = 0
    self.s_xy = self.get_free_point()
    self.g_xy = self.get_free_point()
    return {
        'observation': self.s_xy,
        'achieved_goal': self.s_xy,
        'desired_goal': self.g_xy,
    }


class PointMaze2D(gym.GoalEnv):
  """Wraps the Sibling Rivalry 2D point maze in a gym goal env.
  Keeps the first visit done and uses -1/0 rewards.
  """
  def __init__(self, test=False):
    super().__init__()
    self._env = Env(n=50, maze_type='square_large', use_antigoal=False, ddiff=False, ignore_reset_start=True)
    self.maze = self._env.maze
    self.dist_threshold = 0.15

    self.action_space = gym.spaces.Box(-0.95, 0.95, (2, ))
    observation_space = gym.spaces.Box(-np.inf, np.inf, (2, ))
    goal_space = gym.spaces.Box(-np.inf, np.inf, (2, ))
    self.observation_space = gym.spaces.Dict({
        'observation': observation_space,
        'desired_goal': goal_space,
        'achieved_goal': goal_space
    })

    self.s_xy = np.array(self.maze.sample_start())
    self.g_xy = np.array(self.maze.sample_goal(min_wall_dist=0.025 + self.dist_threshold))
    self.max_steps = 50
    self.num_steps = 0
    self.test = test

  def seed(self, seed=None):
    return self.maze.seed(seed=seed)

  def step(self, action):
    try:
      s_xy = np.array(self.maze.move(tuple(self.s_xy), tuple(action)))
    except:
      print('failed to move', tuple(self.s_xy), tuple(action))
      raise

    self.s_xy = s_xy
    reward = self.compute_reward(s_xy, self.g_xy, None)
    info = {}
    self.num_steps += 1

    if self.test:
      done = np.allclose(0., reward)
      info['is_success'] = done
    else:
      done = False
      info['is_success'] = np.allclose(0., reward)

    if self.num_steps >= self.max_steps and not done:
      done = True
      info['TimeLimit.truncated'] = True

    obs = {
        'observation': s_xy,
        'achieved_goal': s_xy,
        'desired_goal': self.g_xy,
    }

    return obs, reward, done, info

  def reset(self):
    self.num_steps = 0
    s_xy = np.array(self.maze.sample_start())
    self.s_xy = s_xy
    g_xy = np.array(self.maze.sample_goal(min_wall_dist=0.025 + self.dist_threshold))
    self.g_xy = g_xy
    return {
        'observation': s_xy,
        'achieved_goal': s_xy,
        'desired_goal': g_xy,
    }

  def render(self):
    raise NotImplementedError

  def compute_reward(self, achieved_goal, desired_goal, info):
    d = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
    return -(d >= self.dist_threshold).astype(np.float32)