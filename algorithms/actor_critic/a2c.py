import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as distributions
from torch.distributions import Categorical

"""
Advantage Actor Critic (A2C)
baseline version
"""


class RolloutBuffer:
    def __init__(self, args):
        self.states = []
        self.actions = []
        self.rewards = []
        self.args = args

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()


class Actor(nn.Module):  # 策略网络
    def __init__(self, args):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(args.state_dim, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc3 = nn.Linear(args.hidden_dim, args.action_dim)
        self.hs = nn.Hardswish()

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = self.hs(self.fc2(x))
        probs = torch.softmax(self.fc3(x), dim=-1)
        return probs


class Critic(nn.Module):  # 价值网络
    def __init__(self, args):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(args.state_dim, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc3 = nn.Linear(args.hidden_dim, 1)
        self.hs = nn.Hardswish()

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = self.hs(self.fc2(x))
        value = self.fc3(x)
        return value


class ActorCriticAgent:
    def __init__(self, args):
        self.args = args
        self.actor = Actor(args)
        self.critic = Critic(args)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=args.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=args.critic_lr)
        self.buffer = RolloutBuffer(args)

    # inference
    @torch.no_grad()
    def select_action(self, state):
        states = torch.tensor(state, dtype=torch.float32)
        action_prob = self.actor(states)
        dist = Categorical(action_prob)
        action = dist.sample()
        return action.item()

    # training
    def act(self, states, actions):
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions)
        action_probs = self.actor(states)
        dist = Categorical(action_probs)
        action_log_probs = dist.log_prob(actions)
        values = self.critic(states)
        return action_log_probs, values.squeeze()

    def train(self):
        G_list = []
        G_t = 0
        for reward in reversed(self.buffer.rewards):
            G_t = self.args.gamma * G_t + reward
            G_list.append(G_t)
        G_list.reverse()
        G_tensor = torch.tensor(G_list, dtype=torch.float32)

        if self.args.normalize:
            G_tensor = (G_tensor - G_tensor.mean()) / (G_tensor.std() + 1e-6)

        log_probs, values = self.act(self.buffer.states, self.buffer.actions)
        advantage = G_tensor - values.detach()
        actor_loss = -(log_probs * advantage).mean()
        critic_loss = F.mse_loss(values, G_tensor)

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()
        self.buffer.clear()

    def save(self, check_point_path):
        pass

    def load(self, check_point_path):
        pass
