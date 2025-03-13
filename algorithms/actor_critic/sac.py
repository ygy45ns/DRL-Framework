import torch
import torch.nn as nn
import copy
from torch.distributions.categorical import Categorical
import numpy as np
import torch.nn.functional as F

"""
SAC
"""


def build_net(layer_shape, hid_activation, output_activation):
    '''build net with for loop'''
    layers = []
    for j in range(len(layer_shape) - 1):
        act = hid_activation if j < len(layer_shape) - 2 else output_activation
        layers += [nn.Linear(layer_shape[j], layer_shape[j + 1]), act()]
    return nn.Sequential(*layers)


class Double_Q_Net(nn.Module):
    def __init__(self, state_dim, action_dim, hid_shape):
        super(Double_Q_Net, self).__init__()
        layers = [state_dim] + list(hid_shape) + [action_dim]

        self.Q1 = build_net(layers, nn.ReLU, nn.Identity)
        self.Q2 = build_net(layers, nn.ReLU, nn.Identity)

    def forward(self, s):
        q1 = self.Q1(s)
        q2 = self.Q2(s)
        return q1, q2


class Policy_Net(nn.Module):
    def __init__(self, state_dim, action_dim, hid_shape):
        super(Policy_Net, self).__init__()
        layers = [state_dim] + list(hid_shape) + [action_dim]
        self.P = build_net(layers, nn.ReLU, nn.Identity)

    def forward(self, s):
        logits = self.P(s)
        probs = F.softmax(logits, dim=1)
        return probs


class ReplayBuffer(object):
    def __init__(self, state_dim, dvc, max_size=int(1e6)):
        self.max_size = max_size
        self.dvc = dvc
        self.ptr = 0
        self.size = 0

        self.s = torch.zeros((max_size, state_dim), dtype=torch.float, device=self.dvc)
        self.a = torch.zeros((max_size, 1), dtype=torch.long, device=self.dvc)
        self.r = torch.zeros((max_size, 1), dtype=torch.float, device=self.dvc)
        self.s_next = torch.zeros((max_size, state_dim), dtype=torch.float, device=self.dvc)
        self.dw = torch.zeros((max_size, 1), dtype=torch.bool, device=self.dvc)

    def add(self, s, a, r, s_next, dw):
        self.s[self.ptr] = torch.from_numpy(s).to(self.dvc)
        self.a[self.ptr] = a
        self.r[self.ptr] = r
        self.s_next[self.ptr] = torch.from_numpy(s_next).to(self.dvc)
        self.dw[self.ptr] = dw

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = torch.randint(0, self.size, device=self.dvc, size=(batch_size,))
        return self.s[ind], self.a[ind], self.r[ind], self.s_next[ind], self.dw[ind]


class SACD_agent():
    def __init__(self, args):
        self.args = args
        self.tau = 0.005
        self.H_mean = 0
        self.replay_buffer = ReplayBuffer(args.state_dim, args.dvc, max_size=int(1e6))

        self.actor = Policy_Net(args.state_dim, args.action_dim, args.hidden_shape).to(args.dvc)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.lr)

        self.q_critic = Double_Q_Net(args.state_dim, args.action_dim, args.hidden_shape).to(args.dvc)
        self.q_critic_optimizer = torch.optim.Adam(self.q_critic.parameters(), lr=args.lr)
        self.q_critic_target = copy.deepcopy(self.q_critic)
        for p in self.q_critic_target.parameters(): p.requires_grad = False

        if args.adaptive_alpha:
            # We use 0.6 because the recommended 0.98 will cause alpha explosion.
            self.target_entropy = 0.6 * (-np.log(1 / args.action_dim))  # H(discrete)>0
            self.log_alpha = torch.tensor(np.log(args.alpha), dtype=float, requires_grad=True, device=args.dvc)
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=args.lr)

    def select_action(self, state, deterministic):
        with torch.no_grad():
            state = torch.FloatTensor(state[np.newaxis, :]).to(self.args.dvc)  # from (s_dim,) to (1, s_dim)
            probs = self.actor(state)
            if deterministic:
                a = probs.argmax(-1).item()
            else:
                a = Categorical(probs).sample().item()
            return a

    def train(self):
        s, a, r, s_next, dw = self.replay_buffer.sample(self.args.batch_size)
        # ------------------------------------------ Train Critic ----------------------------------------#
        '''Compute the target soft Q value'''
        with torch.no_grad():
            next_probs = self.actor(s_next)  # [b,a_dim]
            next_log_probs = torch.log(next_probs + 1e-8)  # [b,a_dim]
            next_q1_all, next_q2_all = self.q_critic_target(s_next)  # [b,a_dim]
            min_next_q_all = torch.min(next_q1_all, next_q2_all)
            v_next = torch.sum(next_probs * (min_next_q_all - self.args.alpha * next_log_probs), dim=1,
                               keepdim=True)  # [b,1]
            target_Q = r + (~dw) * self.args.gamma * v_next

        '''Update soft Q net'''
        q1_all, q2_all = self.q_critic(s)  # [b,a_dim]
        q1, q2 = q1_all.gather(1, a), q2_all.gather(1, a)  # [b,1]
        q_loss = F.mse_loss(q1, target_Q) + F.mse_loss(q2, target_Q)
        self.q_critic_optimizer.zero_grad()
        q_loss.backward()
        self.q_critic_optimizer.step()

        # ------------------------------------------ Train Actor ----------------------------------------#
        probs = self.actor(s)  # [b,a_dim]
        log_probs = torch.log(probs + 1e-8)  # [b,a_dim]
        with torch.no_grad():
            q1_all, q2_all = self.q_critic(s)  # [b,a_dim]
        min_q_all = torch.min(q1_all, q2_all)

        a_loss = torch.sum(probs * (self.args.alpha * log_probs - min_q_all), dim=1, keepdim=False)  # [b,]

        self.actor_optimizer.zero_grad()
        a_loss.mean().backward()
        self.actor_optimizer.step()

        # ------------------------------------------ Train Alpha ----------------------------------------#
        if self.args.adaptive_alpha:
            with torch.no_grad():
                self.H_mean = -torch.sum(probs * log_probs, dim=1).mean()
            alpha_loss = self.log_alpha * (self.H_mean - self.target_entropy)

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp().item()

        # ------------------------------------------ Update Target Net ----------------------------------#
        for param, target_param in zip(self.q_critic.parameters(), self.q_critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, timestep, EnvName):
        pass

    def load(self, timestep, EnvName):
        pass
