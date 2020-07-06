# Copyright (c) 2019, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: MIT
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/MIT

import torch
import numpy as np
from envs.sibrivalry.ant_maze.create_maze_env import create_maze_env



class Env:
    def __init__(self, n=None, maze_type=None, hardmode=False):
        self.n = n
        self.maze_type = maze_type
        self.hardmode = bool(hardmode)

        self.maze = create_maze_env(maze_type)

        self.dist_threshold = 1.0

        if self.maze_type == 'AntMaze':
            self._dscale = 1.0
        else:
            raise NotImplementedError

        self._state = dict(state=None, goal=None, n=None, done=None)
        self._seed = self.maze.wrapped_env.seed()[0]

        self.reset()

    @property
    def state_size(self):
        return 30

    @property
    def goal_size(self):
        return 2

    @property
    def action_size(self):
        return 8

    @staticmethod
    def to_tensor(x):
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x.astype(np.float32))
        else:
            return torch.FloatTensor(x)

    @staticmethod
    def to_coords(x):
        if isinstance(x, (tuple, list)):
            return x[0], x[1]
        if isinstance(x, torch.Tensor):
            x = x.data.numpy()
        return float(x[0]), float(x[1])

    @property
    def action_range(self):
        return self.to_tensor(self.maze.action_space.high)

    @property
    def state(self):
        return self._state['state'].view(-1).detach()

    @property
    def goal(self):
        return self._state['goal'].view(-1).detach()

    @property
    def achieved(self):
        return self.goal if self.is_success else self.state[:2]

    @property
    def is_done(self):
        return bool(self._state['done'])

    @property
    def is_success(self):
        d = self.dist(self.goal, self.state)
        return d <= self.dist_threshold

    @property
    def next_phase_reset(self):
        return {'state': self._seed, 'goal': self.achieved}

    @property
    def sibling_reset(self):
        return {'state': self._seed, 'goal': self.goal}

    def dist(self, goal, outcome):
        # return torch.sum(torch.abs(goal - outcome))
        return torch.sqrt(torch.sum(torch.pow(goal[:2] - outcome[:2], 2))) / self._dscale

    def sample_goal(self, s_xy=None):
        if self.maze_type == 'AntMaze':
            if self.hardmode:
                g_x = np.random.uniform(low=-3.5, high=3.5)
                g_y = np.random.uniform(low=12.5, high=19.5)
                g = self.to_tensor(np.array([g_x, g_y]).astype(np.float32))
            else:
                g_x, g_y = 0.0, 0.0
                outer_valid = False
                while not outer_valid:
                    valid = False
                    while not valid:
                        g_x = np.random.uniform(low=-3.5, high=19.5)
                        g_y = np.random.uniform(low=-3.5, high=19.5)
                        if g_x > 13:
                            valid = True
                        elif g_y < 3.5 or g_y > 12.5:
                            valid = True
                    g = self.to_tensor(np.array([g_x, g_y]).astype(np.float32))
                    if s_xy is None:
                        outer_valid = True
                    else:
                        d = self.dist(g, s_xy)
                        outer_valid = d > self.dist_threshold
            return g
        else:
            raise NotImplementedError

    def reset(self, state=None, goal=None):
        if state is None:
            self._seed = self.maze.wrapped_env.seed()[0]
            s = self.to_tensor(self.maze.reset())
            _ = self.maze.wrapped_env.seed()
        else:
            self._seed = self.maze.wrapped_env.seed(int(state))[0]
            s = self.to_tensor(self.maze.reset())
            _ = self.maze.wrapped_env.seed()

        if goal is None:
            goal = self.sample_goal(s_xy=s[:2])

        self._state = {
            'state': s,
            'goal': goal,
            'n': 0,
            'done': False
        }

    def step(self, action):
        next_state, _, _, _ = self.maze.step(action)
        self._state['state'] = self.to_tensor(next_state)
        self._state['n'] += 1
        self._state['done'] = (self._state['n'] >= self.n) or self.is_success