# Copyright (c) 2019, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: MIT
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/MIT

import torch
import torch.nn as nn
import numpy as np
import torch.distributed as dist
from torch.distributions import Normal, Beta
from envs.sibrivalry.ant_maze.create_maze_env import create_maze_env



class Env:
    def __init__(self, n, maze_type, hardmode=False):
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

    def dist(self, goal, outcome):
        # return torch.sum(torch.abs(goal - outcome))
        return torch.sqrt(torch.sum(torch.pow(goal[:2] - outcome[:2], 2))) / self._dscale

    def sample_goal(self):
        if self.maze_type == 'AntMaze':
            if self.hardmode:
                g_x = np.random.uniform(low=-3.5, high=11.5)
                g_y = np.random.uniform(low=12.5, high=19.5)
            else:
                g_x, g_y = 0.0, 0.0
                valid = False
                while not valid:
                    g_x = np.random.uniform(low=-3.5, high=19.5)
                    g_y = np.random.uniform(low=-3.5, high=19.5)
                    if g_x > 13:
                        valid = True
                    elif g_y < 3.5 or g_y > 12.5:
                        valid = True
            g = np.array([g_x, g_y]).astype(np.float32)
            return self.to_tensor(g)
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
            goal = self.sample_goal()

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


class Policy(nn.Module):
    def __init__(self, a_range):
        super().__init__()
        self.a_range = a_range
        self.layers = nn.Sequential(
            nn.Linear(32, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 8),
        )

    def forward(self, s, g):
        """Produce an action"""
        return torch.tanh(self.layers(torch.cat([s, g], dim=1)) * 0.005) * self.a_range


class StochasticPolicy(nn.Module):
    def __init__(self, a_range):
        super().__init__()
        self.a_range = a_range
        self.layers = nn.Sequential(
            nn.Linear(32, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 16),
            nn.Softplus()
        )

    def action_stats(self, s, g):
        x = torch.cat([s, g], dim=1)
        action_stats = self.layers(x) + 1 + 1e-6
        return action_stats[:, :8], action_stats[:, 8:]

    def scale_action(self, logit):
        # Scale to [-1, 1]
        logit = 2 * (logit - 0.5)
        # Scale to the action range
        action = logit * self.a_range
        return action

    def forward(self, s, g, greedy=False, action_logit=None):
        """Produce an action"""
        c0, c1 = self.action_stats(s, g)
        action_mode = (c0 - 1) / (c0 + c1 - 2)
        m = Beta(c0, c1)

        # Sample.
        if action_logit is None:
            if greedy:
                action_logit = action_mode
            else:
                action_logit = m.sample()

            n_ent = -m.entropy().mean()
            lprobs = m.log_prob(action_logit)
            action = self.scale_action(action_logit)
            return action, action_logit, lprobs, n_ent

        # Evaluate the action previously taken
        else:
            n_ent = -m.entropy().mean(dim=1)
            lprobs = m.log_prob(action_logit)
            action = self.scale_action(action_logit)
            return lprobs, n_ent, action


class Critic(nn.Module):
    def __init__(self, use_antigoal):
        super().__init__()
        self.use_antigoal = use_antigoal
        self.layers = nn.Sequential(
            nn.Linear(42 if self.use_antigoal else 40, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def q_no_grad(self, s, a, g, ag):
        for p in self.parameters():
            p.requires_grad = False

        q = self(s, a, g, ag)

        for p in self.parameters():
            p.requires_grad = True

        return q

    def forward(self, s, a, g, ag):
        """Produce an action"""
        if self.use_antigoal:
            return self.layers(torch.cat([s, a, g, ag], dim=1)).view(-1)
        else:
            return self.layers(torch.cat([s, a, g], dim=1)).view(-1)


class Value(nn.Module):
    def __init__(self, use_antigoal):
        super().__init__()
        self.use_antigoal = use_antigoal
        self.layers = nn.Sequential(
            nn.Linear(34 if self.use_antigoal else 32, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def q_no_grad(self, s, g, ag):
        for p in self.parameters():
            p.requires_grad = False

        v = self(s, g, ag)

        for p in self.parameters():
            p.requires_grad = True

        return v

    def forward(self, s, g, ag):
        if self.use_antigoal:
            return self.layers(torch.cat([s, g, ag], dim=1)).view(-1)
        else:
            return self.layers(torch.cat([s, g], dim=1)).view(-1)


class Agent(nn.Module):
    def __init__(self, n, maze_type, noise=0.1, epsilon=0.2, hardmode=False):
        super().__init__()

        self.noise = max(0.0, float(noise))
        self.epsilon = max(0.0, min(1.0, float(epsilon)))

        self.env = Env(n=n, maze_type=maze_type, hardmode=hardmode)
        self.policy = Policy(a_range=self.env.action_range)
        self.episode = []

    @property
    def rollout(self):
        states = torch.stack([e['pre_achieved'] for e in self.episode] + [self.episode[-1]['achieved']]).data.numpy()
        xs = states[:, 0]
        ys = states[:, 1]
        return [xs, ys]

    def reset(self, state=None, goal=None):
        self.env.reset(state, goal)
        self.episode = []

    def step(self, do_eval=False):
        s = self.env.state
        g = self.env.goal
        pre_achieved = self.env.achieved
        a = self.policy(s.view(1, -1), g.view(1, -1)).view(-1)

        if not do_eval:
            if np.random.rand() < self.epsilon:
                a = np.random.uniform(
                    low=-self.env.action_range.data.numpy(),
                    high=self.env.action_range.data.numpy(),
                    size=(8,)
                )
                a = torch.from_numpy(a.astype(np.float32))
            else:
                z = Normal(torch.zeros_like(a), torch.ones_like(a) * self.noise * self.env.action_range)
                a = a + z.sample()
        ar = self.env.action_range[0]
        a = torch.clamp(a, -ar, ar)

        self.env.step(a.data.numpy())
        complete = float(self.env.is_success) * torch.ones(1)
        terminal = float(self.env.is_done) * torch.ones(1)
        s_next = self.env.state
        achieved = self.env.achieved
        r = -1 * torch.ones(1)

        self.episode.append({
            'state': s,
            'goal': g,
            'action': a,
            'pre_achieved': pre_achieved,
            'next_state': s_next,
            'achieved': achieved,
            'terminal': terminal.view([]),
            'complete': complete.view([]),
            'reward': r.view([]),
        })

    def play_episode(self, reset_dict={}, do_eval=False):
        self.reset(**reset_dict)
        while not self.env.is_done:
            self.step(do_eval)


class StochasticAgent(nn.Module):
    def __init__(self, n, maze_type, policy=None, hardmode=False):
        super().__init__()

        self.env = Env(n=n, maze_type=maze_type, hardmode=hardmode)
        if isinstance(policy, StochasticPolicy):
            self.policy = policy
        else:
            self.policy = StochasticPolicy(a_range=self.env.action_range)
        self.episode = []

    @property
    def rollout(self):
        states = torch.stack([e['pre_achieved'] for e in self.episode] + [self.episode[-1]['achieved']]).data.numpy()
        xs = states[:, 0]
        ys = states[:, 1]
        return [xs, ys]

    def reset(self, state=None, goal=None):
        self.env.reset(state, goal)
        self.episode = []

    def step(self, do_eval=False):
        s = self.env.state
        g = self.env.goal
        pre_achieved = self.env.achieved
        a, logit, log_prob, n_ent = self.policy(s.view(1, -1), g.view(1, -1), greedy=do_eval)
        a = a.view(-1)
        logit = logit.view(-1)
        log_prob = log_prob.sum()

        self.env.step(a.data.numpy())
        complete = float(self.env.is_success) * torch.ones(1)
        terminal = float(self.env.is_done) * torch.ones(1)
        s_next = self.env.state
        achieved = self.env.achieved
        r = -1 * torch.ones(1)

        self.episode.append({
            'state': s,
            'goal': g,
            'pre_achieved': pre_achieved,
            'action': a,
            'action_logit': logit,
            'log_prob': log_prob.view([]),
            'n_ent': n_ent.view([]),
            'next_state': s_next,
            'achieved': achieved,
            'terminal': terminal.view([]),
            'complete': complete.view([]),
            'reward': r.view([]),
        })

    def play_episode(self, reset_dict={}, do_eval=False):
        self.reset(**reset_dict)
        while not self.env.is_done:
            self.step(do_eval)


class GoalChainAgentDDPG(nn.Module):
    def __init__(self,
                 n=100, maze_type='AntMaze',
                 gamma=0.98, polyak=0.95,
                 noise=0.1, epsilon=0.2,
                 action_l2_lambda=0.01,
                 ddiff=False,
                 use_dense=True,
                 use_antigoal=True):
        super().__init__()

        self.max_goal_prop = 0

        self.gamma = gamma
        self.polyak = polyak
        self.epsilon = epsilon
        self.action_l2_lambda = action_l2_lambda
        self.ddiff = bool(ddiff)
        self.use_antigoal = use_antigoal
        self.use_dense = use_dense

        self.a0 = Agent(n=n, maze_type=maze_type, noise=noise, epsilon=self.epsilon)
        self.a1 = Agent(n=n, maze_type=maze_type, noise=noise, epsilon=self.epsilon)
        self._sync_agents()

        self.q_module = Critic(self.use_antigoal)
        self.q_target = Critic(self.use_antigoal)
        self.q_target.load_state_dict(self.q_module.state_dict())

        self.p_module = self.a0.policy
        # self.p_target = StochasticPolicy(self.a0.env.action_range)
        self.p_target = Policy(self.a0.env.action_range)
        self.p_target.load_state_dict(self.p_module.state_dict())

        # VERY IMPORTANT that the training algorithm uses train_steps to track the number of episodes played
        self.train_steps = nn.Parameter(torch.zeros(1))
        self.train_steps.requires_grad = False

        self.distance = []
        self.distance_ag = []
        self._ep_summary = []

        # For clamping the q target
        if self.use_dense:
            self._q_clamp = [-2 / (1 - self.gamma), 2]
        else:
            self._q_clamp = [-1 / (1 - self.gamma), 0]

    def reset(self):
        return

    @property
    def was_success(self):
        return bool(self.a0.env.is_success)

    @property
    def curr_ep(self):
        return self.a0.episode

    def save_checkpoint(self, filepath):
        self._sync_agents()
        torch.save(self.state_dict(), filepath)

    def load_checkpoint(self, filepath):
        checkpoint = torch.load(filepath)
        self.load_state_dict(checkpoint)

    def _sync_agents(self):
        self.a1.load_state_dict(self.a0.state_dict())

    def soft_update(self):
        for p, p_targ in zip(self.q_module.parameters(), self.q_target.parameters()):
            p_targ.data *= self.polyak
            p_targ.data += (1 - self.polyak) * p.data

        for p, p_targ in zip(self.p_module.parameters(), self.p_target.parameters()):
            p_targ.data *= self.polyak
            p_targ.data += (1 - self.polyak) * p.data

        self._sync_agents()

    def play_episode(self, reset_dict={}, do_eval=False):
        self.distance = []
        self.distance_ag = []
        self._ep_summary = []

        self.a0.play_episode(reset_dict, do_eval=do_eval)
        self.a1.play_episode(reset_dict=self.a0.env.next_phase_reset, do_eval=True)

    def relabel_episode(self):
        antigoal = self.a1.env.achieved

        for e in self.a0.episode:
            e['antigoal'] = antigoal

            dg_nxt = self.a0.env.dist(e['achieved'], e['goal'])
            da_nxt = self.a0.env.dist(e['achieved'], e['antigoal'])

            ddg = -dg_nxt
            dda = -da_nxt

            if self.ddiff:
                ddg += self.a0.env.dist(e['pre_achieved'], e['goal'])
                dda += self.a0.env.dist(e['pre_achieved'], e['antigoal'])

            r_sparse = 1 * e['complete']
            r_dense = ddg - (float(self.use_antigoal) * dda)

            e['reward'] += r_sparse + (float(self.use_dense) * r_dense)

            self.distance.append(dg_nxt.item())
            self.distance_ag.append(da_nxt.item())

    def transitions_for_buffer(self, training=None):
        return self.a0.episode

    def normalize_batch(self, batch_dict):
        """Apply batch normalization to the appropriate inputs within the batch"""
        return batch_dict

    def forward(self, state, action, goal, antigoal, reward, next_state, terminal, complete, **kwargs):
        # Get the q target
        next_policy_actions = self.p_target(next_state, goal)
        q_next = self.q_target(next_state, next_policy_actions, goal, antigoal)
        q_targ = reward + ((1 - complete) * self.gamma * q_next)
        q_targ = torch.clamp(q_targ, *self._q_clamp)

        # Get the Q values associated with the observed transitions
        q = self.q_module(state, action, goal, antigoal)

        # Loss for the q_module
        q_loss = torch.pow(q - q_targ.detach(), 2).mean()

        # We want to optimize the actions wrt their q value (without getting q module gradients)
        policy_actions = self.p_module(state, goal)
        p_loss = -self.q_target.q_no_grad(state, policy_actions, goal, antigoal).mean()

        l2 = torch.mean(policy_actions ** 2)
        l2_loss = l2 * self.action_l2_lambda

        succeeded = float(self.was_success)
        dist_to_g = float(self.distance[-1])
        dist_to_a = float(self.distance_ag[-1])
        avg_r = reward.mean()
        avg_q = q.mean()
        self._ep_summary = [
            succeeded, dist_to_g, dist_to_a, avg_r.item(), avg_q.item(), q_loss.item(), p_loss.item(), l2.item()
        ]

        return p_loss + q_loss + l2_loss

    def episode_summary(self):
        keys = [k for k in self.a0.episode[0].keys()]
        batched_ep = {}
        for key in keys:
            batched_ep[key] = torch.stack([e[key] for e in self.a0.episode]).detach()

        _ = self(**batched_ep)

        return [float(x) for x in self._ep_summary]


class GoalChainAgentDDPGHER(GoalChainAgentDDPG):
    def __init__(self, k=4, **kwargs):
        self.k = k
        # Switch the default here to NOT use dense reward (by default, this is true HER w/ sparse reward)
        if 'use_dense' not in kwargs:
            kwargs['use_dense'] = False
        super().__init__(**kwargs)

    def transitions_for_buffer(self, training=None):
        her_transitions = []

        for t in range(len(self.curr_ep)):
            perm_idx = [int(i) for i in np.random.permutation(np.arange(t, len(self.curr_ep)))]
            her_goal_indices = perm_idx[:self.k]
            for idx in her_goal_indices:
                her_goal = self.curr_ep[idx]['achieved']

                dg_pre = self.a0.env.dist(her_goal, self.curr_ep[t]['state'])
                dg_nxt = self.a0.env.dist(her_goal, self.curr_ep[t]['next_state'])

                if dg_pre == 0: # This transition makes no sense. Skip it
                    continue

                da_pre = self.a0.env.dist(self.curr_ep[t]['antigoal'], self.curr_ep[t]['state'])
                da_nxt = self.a0.env.dist(self.curr_ep[t]['antigoal'], self.curr_ep[t]['next_state'])

                ddg = -dg_nxt
                dda = -da_nxt

                if self.ddiff:
                    ddg += dg_pre
                    dda += da_pre

                now_complete = bool((dg_nxt <= self.a0.env.dist_threshold).item())
                r_sparse = float(now_complete)
                r_dense = ddg - (float(self.use_antigoal) * dda)

                new_reward = -1 + r_sparse + (float(self.use_dense) * r_dense)

                new_trans = {k: v.clone().detach() for k, v in self.curr_ep[t].items()}
                new_trans['goal'] = her_goal.detach()
                new_trans['reward'] = new_reward * torch.ones_like(new_trans['reward'])

                # The episode would have ended here (with the her goal)
                if now_complete:
                    new_trans['terminal'] = torch.ones_like(new_trans['terminal'])
                    new_trans['complete'] = torch.ones_like(new_trans['complete'])

                her_transitions.append(new_trans)

        return her_transitions + self.a0.episode


class GoalChainAgentPPO(nn.Module):
    def __init__(self,
                 n=100, maze_type='AntMaze', hardmode=False,
                 horizon=400, mini_batch_size=100,
                 gamma=0.98, gae_lambda=0.95,
                 entropy_lambda=0.05, action_l2_weight=0.0,
                 ddiff=False,
                 use_antigoal=True, max_goal_prop=0):
        super().__init__()

        self.n = n
        self.use_antigoal = use_antigoal

        self.max_goal_prop = max_goal_prop

        self.gamma = float(gamma)

        self.gae_lambda = float(gae_lambda)
        self.entropy_lambda = float(entropy_lambda)
        self.action_l2_weight = float(action_l2_weight)
        self.ddiff = bool(ddiff)

        self.a0 = StochasticAgent(n=n, maze_type=maze_type, hardmode=hardmode)
        self.a1 = StochasticAgent(n=n, maze_type=maze_type, hardmode=hardmode)

        self.v_module = Value(self.use_antigoal)

        # VERY IMPORTANT that the training algorithm uses train_steps to track the number of episodes played
        self.train_steps = nn.Parameter(torch.zeros(1))
        self.train_steps.requires_grad = False

        self.horizon = int(horizon)
        self.mini_batch_size = int(mini_batch_size)
        self.clip_range = 0.2

        self._mini_buffer = {'state': None}
        self._epoch_transitions = {}

        self.distance = []
        self.distance_ag = []

        self.losses = []

        self._ep_summary = []
        self._batched_ep = None
        self._phase = 0
        self._next_reset = {}

    def reset(self):
        return

    def sync_agents(self):
        self.a1.load_state_dict(self.a0.state_dict())

    @property
    def current_horizon(self):
        mb_state = self._mini_buffer['state']
        if mb_state is None:
            return 0
        else:
            return mb_state.shape[0]

    @property
    def was_success(self):
        return bool(self.a0.env.is_success)

    @property
    def curr_ep(self):
        return self.a0.episode

    def save_checkpoint(self, filepath):
        self.sync_agents()
        torch.save(self.state_dict(), filepath)

    def load_checkpoint(self, filepath):
        checkpoint = torch.load(filepath)
        self.load_state_dict(checkpoint)

    def add_to_mini_buffer(self, batched_episode):
        for k, v in batched_episode.items():
            if self._mini_buffer.get(k, None) is None:
                self._mini_buffer[k] = v.detach()
            else:
                self._mini_buffer[k] = torch.cat([self._mini_buffer[k], v], dim=0)

        curr_horizon = int(self.current_horizon)
        assert all([int(v.shape[0]) == curr_horizon for v in self._mini_buffer.values()])

    def fill_epoch_transitions(self):
        curr_horizon = int(self.current_horizon)
        assert curr_horizon >= self.horizon
        self._epoch_transitions = {}
        for k, v in self._mini_buffer.items():
            self._epoch_transitions[k] = v[:self.horizon]
            if curr_horizon > self.horizon:
                self._mini_buffer[k] = v[self.horizon:]
            else:
                self._mini_buffer[k] = None

    def make_epoch_mini_batches(self, normalize_advantage=False):
        mb_indices = np.split(np.random.permutation(self.horizon), self.horizon // self.mini_batch_size)

        mini_batches = []
        for indices in mb_indices:
            this_batch = {k: v[indices] for k, v in self._epoch_transitions.items()}
            if normalize_advantage and 'advantage' in this_batch:
                mb_mean = this_batch['advantage'].mean(dim=0).view(1, -1)
                mb_std = this_batch['advantage'].std(dim=0).view(1, -1)
                this_batch['advantage'] = (this_batch['advantage'] - mb_mean) / mb_std
            mini_batches.append(this_batch)
        return mini_batches

    def distributed_advantage_normalization(self):
        if 'advantage' not in self._epoch_transitions:
            return
        a = self._epoch_transitions['advantage']

        a_sum = a.sum(dim=0)
        a_sumsq = torch.pow(a, 2).sum(dim=0)
        dist.all_reduce(a_sum)
        dist.all_reduce(a_sumsq)

        n = dist.get_world_size()
        a_mean = a_sum / (self.horizon * n)
        a_var = (a_sumsq / (self.horizon * n)) - (a_mean ** 2)
        a_std = torch.pow(a_var, 0.5) + 1e-8

        n_dim = len(a.shape)
        if n_dim == 1:
            self._epoch_transitions['advantage'] = (a - a_mean) / a_std
        elif n_dim == 2:
            self._epoch_transitions['advantage'] = (a - a_mean.view(1, -1)) / a_std.view(1, -1)
        else:
            raise NotImplementedError

    def reach_horizon(self):
        self.sync_agents()
        while self.current_horizon < self.horizon:
            self.play_episode()
            batched_episode = {k: v.detach() for k, v in self.compress_episode().items()}
            self.add_to_mini_buffer(batched_episode)
        self.fill_epoch_transitions()

    def play_episode(self, reset_dict=None, do_eval=False, force_reset=False):
        self.distance = []
        self.distance_ag = []
        self._ep_summary = []
        self._batched_ep = None

        if do_eval or force_reset:
            self._phase = 0
            self._next_reset = {}

        if reset_dict is None:
            reset_dict = self._next_reset

        self.a0.play_episode(reset_dict, do_eval=do_eval)
        self.a1.play_episode(reset_dict=self.a0.env.next_phase_reset, do_eval=True)
        self.relabel_episode()

        self._phase += 1

        if do_eval or self._phase > self.max_goal_prop:
            self._phase = 0

        if self.a0.env.is_success or self.a1.env.is_success:
            self._phase = 0

        if self._phase == 0:
            self._next_reset = {}
        else:
            self._next_reset = self.a0.env.next_phase_reset

    def relabel_episode(self):
        antigoal = self.a1.env.achieved

        for e in self.a0.episode:
            e['antigoal'] = antigoal

            dg_nxt = self.a0.env.dist(e['achieved'], e['goal'])
            da_nxt = self.a0.env.dist(e['achieved'], e['antigoal'])

            ddg = -dg_nxt
            dda = -da_nxt

            if self.ddiff:
                ddg += self.a0.env.dist(e['pre_achieved'], e['goal'])
                dda += self.a0.env.dist(e['pre_achieved'], e['antigoal'])

            r_sparse = 1 * e['complete']
            r_dense = ddg - (float(self.use_antigoal) * dda)

            e['reward'] += r_sparse + r_dense

            self.distance.append(dg_nxt.item())
            self.distance_ag.append(da_nxt.item())

    def compress_episode(self):
        keys = [
            'state', 'next_state', 'goal', 'antigoal',
            'action', 'n_ent', 'log_prob', 'action_logit',
            'reward', 'terminal', 'complete'
        ]
        batched_episode = {}
        for key in keys:
            batched_episode[key] = torch.stack([et[key] for et in self.a0.episode])

        batched_episode['value'] = self.v_module(
            batched_episode['state'],
            batched_episode['goal'],
            batched_episode['antigoal']
        )

        advs = torch.zeros_like(batched_episode['reward'])
        last_adv = 0
        for t in reversed(range(advs.shape[0])):
            if t == advs.shape[0] - 1:
                has_next = 1.0 - batched_episode['complete'][t]  # Is this a genuine terminal action?
                # has_next = 1.0 - batched_episode['terminal'][t]  # Will there be a timestep after this one?
                next_value = self.v_module(
                    batched_episode['next_state'][-1:],
                    batched_episode['goal'][-1:],
                    batched_episode['antigoal'][-1:]
                )
            else:
                has_next = 1.0  # By our setup, this cannot be a terminal action
                next_value = batched_episode['value'][t + 1]

            delta = batched_episode['reward'][t] + self.gamma * next_value * has_next - batched_episode['value'][t]
            advs[t] = delta + self.gamma * self.gae_lambda * has_next * last_adv
            last_adv = advs[t]

        batched_episode['advantage'] = advs.detach()
        batched_episode['cumulative_return'] = advs.detach() + batched_episode['value'].detach()

        self._batched_ep = batched_episode

        return batched_episode

    def replay_loss(self, mini_batch=None):
        if mini_batch is None:
            mini_batch = self._batched_ep
            fill_summary = True
        else:
            fill_summary = False

        value = self.v_module(
            mini_batch['state'],
            mini_batch['goal'],
            mini_batch['antigoal']
        )
        v_loss = 0.5 * torch.pow(mini_batch['cumulative_return'] - value, 2).mean()

        log_prob, n_ent, greedy_action = self.a0.policy(
            mini_batch['state'], mini_batch['goal'],
            action_logit=mini_batch['action_logit']
        )
        e_loss = n_ent.mean()
        l2_loss = torch.pow(greedy_action, 2).mean()

        # Defining Loss = - J is equivalent to max J
        log_prob = log_prob.sum(dim=1)
        ratio = torch.exp(log_prob - mini_batch['log_prob'])

        pg_losses1 = -mini_batch['advantage'] * ratio
        pg_losses2 = -mini_batch['advantage'] * torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)
        p_losses = torch.max(pg_losses1, pg_losses2)
        p_loss = p_losses.mean()

        loss = v_loss + p_loss + (self.entropy_lambda * e_loss) + (self.action_l2_weight * l2_loss)

        if fill_summary:
            succeeded = float(self.was_success)
            dist_to_g = float(self.distance[-1])
            dist_to_a = float(self.distance_ag[-1])
            avg_r = mini_batch['reward'].mean()
            avg_v = mini_batch['value'].mean()
            self._ep_summary = [
                succeeded, dist_to_g, dist_to_a, avg_r.item(), avg_v.item(), v_loss.item(), p_loss.item(), e_loss.item()
            ]

        return loss

    def forward(self, mini_batch=None):
        return self.replay_loss(mini_batch)

    def episode_summary(self):
        if not self._batched_ep:
            _ = self.compress_episode()
        if not self._ep_summary:
            _ = self.replay_loss()
        return [float(x) for x in self._ep_summary]


class GoalChainAgentPairGo(GoalChainAgentPPO):
    def __init__(self,
                 n=100, maze_type='AntMaze', hardmode=False,
                 horizon=400, mini_batch_size=100,
                 gamma=0.98, gae_lambda=0.95,
                 entropy_lambda=0.05, action_l2_weight=0.0,
                 ddiff=False,
                 n_rollouts=2,
                 use_antigoal=True,
                 value_ignore_antigoal=False
                 ):
        super().__init__()

        self.n = n
        self.use_antigoal = use_antigoal

        self.gamma = float(gamma)

        self.gae_lambda = float(gae_lambda)
        self.entropy_lambda = float(entropy_lambda)
        self.action_l2_weight = float(action_l2_weight)
        self.ddiff = bool(ddiff)
        self.value_antigoal_coeff = 0.0 if value_ignore_antigoal else 1.0

        self.n_rollouts = max(2, int(n_rollouts))

        _dummy_env = Env(n=n, maze_type=maze_type)
        self.policy = StochasticPolicy(_dummy_env.action_range)
        self.v_module = Value(self.use_antigoal)

        self.agents = [
            StochasticAgent(
                n=n, maze_type=maze_type, policy=self.policy, hardmode=hardmode
            ) for _ in range(self.n_rollouts)
        ]
        self.a0 = self.agents[0]
        self.a1 = self.agents[1]

        # VERY IMPORTANT that the training algorithm uses train_steps to track the number of episodes played
        self.train_steps = nn.Parameter(torch.zeros(1))
        self.train_steps.requires_grad = False

        self.horizon = int(horizon)
        self.mini_batch_size = int(mini_batch_size)
        self.clip_range = 0.2

        self._mini_buffer = {'state': None}
        self._epoch_transitions = {}

        self.distance = [[] for _ in range(self.n_rollouts)]
        self.distance_ag = [[] for _ in range(self.n_rollouts)]

        self.losses = []

        self._ep_summary = []
        self._compress_me = []
        self._batched_ep = None

    @property
    def avg_success(self):
        return float(np.mean([agent.env.is_success for agent in self.agents]))

    @property
    def avg_dist_to_goal(self):
        return float(np.mean([d[-1] for d in self.distance]))

    @property
    def avg_dist_to_antigoal(self):
        return float(np.mean([d[-1] for d in self.distance_ag]))

    def sync_agents(self):
        return

    def play_episode(self, reset_dict=None, do_eval=False, force_reset=None):
        self.distance = [[] for _ in range(self.n_rollouts)]
        self.distance_ag = [[] for _ in range(self.n_rollouts)]
        self._ep_summary = []
        self._compress_me = []
        self._batched_ep = None

        if reset_dict is None:
            reset_dict = {}

        for agent in self.agents:
            agent.play_episode(reset_dict, do_eval)
            reset_dict = {'state': agent.env._seed, 'goal': agent.env._state['goal'].detach()}
        self.relabel_episode()

    def _organize_episodes(self):
        goal = self.a0.env.goal.detach()
        ep_dicts = []
        for ai, agent in enumerate(self.agents):
            ep_dict = {'ai': ai, 'ep': agent.episode, 'achieved': agent.env.achieved, 'antigoal': None}
            ep_dicts.append(ep_dict)
        ep_dicts = sorted(ep_dicts, key=lambda d: self.a0.env.dist(goal, d['achieved']).item())

        ep_dicts[0]['antigoal'] = ep_dicts[1]['achieved']
        for ep_rank in range(1, self.n_rollouts):
            ep_dicts[ep_rank]['antigoal'] = ep_dicts[ep_rank - 1]['achieved']

        return ep_dicts

    def relabel_episode(self):
        self._compress_me = []

        ep_dicts = self._organize_episodes()

        for rank, ep_dict in enumerate(ep_dicts):
            for e in ep_dict['ep']:
                e['antigoal'] = ep_dict['antigoal']

                dg_nxt = self.a0.env.dist(e['next_state'], e['goal'])
                da_nxt = self.a0.env.dist(e['next_state'], e['antigoal'])

                ddg = -dg_nxt
                dda = -da_nxt

                if self.ddiff:
                    ddg += self.a0.env.dist(e['state'], e['goal'])
                    dda += self.a0.env.dist(e['state'], e['antigoal'])

                r_sparse = 1 * e['complete']
                r_dense = ddg - (float(self.use_antigoal) * dda)

                e['reward'] += r_sparse + r_dense

                ai = ep_dict['ai']
                self.distance[ai].append(dg_nxt.item())
                self.distance_ag[ai].append(da_nxt.item())

            if rank > 0 or da_nxt < dg_nxt or self.agents[ep_dict['ai']].env.is_success:
                self._compress_me.append(ep_dict['ep'])

    def compress_episode(self):
        keys = [
            'state', 'next_state', 'goal', 'antigoal',
            'action', 'n_ent', 'log_prob', 'action_logit',
            'reward', 'terminal', 'complete',
        ]
        batched_episodes = []
        for ep in self._compress_me:
            batched_episode = {key: torch.stack([e[key] for e in ep]) for key in keys}

            batched_episode['value'] = self.v_module(
                batched_episode['state'],
                batched_episode['goal'],
                batched_episode['antigoal'] * self.value_antigoal_coeff
            )

            advs = torch.zeros_like(batched_episode['reward'])
            last_adv = 0
            for t in reversed(range(advs.shape[0])):
                if t == advs.shape[0] - 1:
                    has_next = 1.0 - batched_episode['complete'][t]  # Is this a genuine terminal action?
                    next_value = self.v_module(
                        batched_episode['next_state'][-1:],
                        batched_episode['goal'][-1:],
                        batched_episode['antigoal'][-1:] * self.value_antigoal_coeff
                    )
                else:
                    has_next = 1.0  # By our setup, this cannot be a terminal action
                    next_value = batched_episode['value'][t + 1]

                delta = batched_episode['reward'][t] + self.gamma * next_value * has_next - batched_episode['value'][t]
                advs[t] = delta + self.gamma * self.gae_lambda * has_next * last_adv
                last_adv = advs[t]

            batched_episode['advantage'] = advs.detach()
            batched_episode['cumulative_return'] = advs.detach() + batched_episode['value'].detach()

            batched_episodes.append(batched_episode)

        if len(batched_episodes) == 1:
            batched_ep = batched_episodes[0]

        else:
            keys = batched_episodes[0].keys()
            batched_ep = {
                k: torch.cat([b_ep[k] for b_ep in batched_episodes]) for k in keys
            }

        self._batched_ep = batched_ep

        return batched_ep

    def replay_loss(self, mini_batch=None):
        if mini_batch is None:
            mini_batch = self._batched_ep
            fill_summary = True
        else:
            fill_summary = False

        # sum_w = torch.sum(mini_batch['weight'])

        value = self.v_module(
            mini_batch['state'],
            mini_batch['goal'],
            mini_batch['antigoal'] * self.value_antigoal_coeff
        )
        v_losses = 0.5 * torch.pow(mini_batch['cumulative_return'] - value, 2)
        v_loss = v_losses.mean()
        # v_loss = torch.mean(v_losses * mini_batch['weight'])

        log_prob, n_ent, greedy_action = self.policy(
            mini_batch['state'], mini_batch['goal'],
            action_logit=mini_batch['action_logit']
        )
        e_loss = n_ent.mean()
        # e_loss = torch.mean(n_ent * mini_batch['weight'])

        # Defining Loss = - J is equivalent to max J
        log_prob = log_prob.sum(dim=1)
        ratio = torch.exp(log_prob - mini_batch['log_prob'])

        pg_losses1 = -mini_batch['advantage'] * ratio
        pg_losses2 = -mini_batch['advantage'] * torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)
        p_losses = torch.max(pg_losses1, pg_losses2)
        p_loss = p_losses.mean()
        # p_loss = torch.mean(p_losses * mini_batch['weight'])

        loss = v_loss + p_loss + (self.entropy_lambda * e_loss)

        if fill_summary:
            succeeded = self.avg_success
            dist_to_g = self.avg_dist_to_goal
            dist_to_a = self.avg_dist_to_antigoal
            avg_r = mini_batch['reward'].mean()
            avg_v = mini_batch['value'].mean()
            self._ep_summary = [
                succeeded, dist_to_g, dist_to_a, avg_r.item(), avg_v.item(), v_loss.item(), p_loss.item(), e_loss.item()
            ]

        return loss

    def forward(self, mini_batch=None):
        return self.replay_loss(mini_batch)

    def episode_summary(self):
        if not self._batched_ep:
            _ = self.compress_episode()
        if not self._ep_summary:
            _ = self.replay_loss()
        return [float(x) for x in self._ep_summary]