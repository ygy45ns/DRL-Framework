import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import torch.nn.functional as F

"""
PPO
"""


class RolloutBuffer:
    def __init__(self, args):
        self.states = []
        self.avail_actions = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.masks = []
        self.args = args

    def clear(self):
        self.states.clear()
        self.avail_actions.clear()
        self.actions.clear()
        self.rewards.clear()
        self.next_states.clear()
        self.masks.clear()

    def get_batches(self):
        n = len(self.states)
        indices = np.arange(n, dtype=np.int64)
        batch_start = np.arange(0, n, self.args.batch_size)
        np.random.shuffle(indices)
        batches = [indices[i:i + self.args.batch_size] for i in batch_start]
        return batches


class Actor(nn.Module):  # 策略网络
    def __init__(self, args):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(args.state_dim, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc3 = nn.Linear(args.hidden_dim, args.action_dim)
        self.hs = nn.Hardswish()

    def forward(self, state):
        x = torch.tanh(self.fc1(state))
        x = torch.tanh(self.fc2(x))
        probs = F.softmax(self.fc3(x), dim=-1)
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
        x = torch.relu(self.fc2(x))
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
        state = torch.as_tensor(state, dtype=torch.float32)
        action_prob = self.actor(state)
        dist = Categorical(action_prob)
        action = dist.sample()
        return action.item()

    # test
    @torch.no_grad()
    def select_argmax_action(self, state):
        state = torch.as_tensor(state, dtype=torch.float32)
        action_prob = self.actor(state)
        return torch.argmax(action_prob).item()

    # training
    def evaluate_action(self, states, actions):
        action_prob = self.actor(states)
        dist = Categorical(action_prob)
        action_log_probs = dist.log_prob(actions)
        action_entropy = dist.entropy()
        return action_log_probs, action_entropy

    def learn(self):
        # tensor buffer
        masks = torch.as_tensor(self.buffer.masks, dtype=torch.float32)

        rewards = torch.as_tensor(self.buffer.rewards, dtype=torch.float32)

        if self.args.standardize_rewards:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-6)

        returns = self.calculate_returns(rewards, masks)

        states = torch.as_tensor(self.buffer.states, dtype=torch.float32)
        next_states = torch.as_tensor(self.buffer.next_states, dtype=torch.float32)
        actions = torch.as_tensor(self.buffer.actions, dtype=torch.int64)

        # GAE calculation
        with torch.no_grad():
            old_state_values = self.critic(states).squeeze()
            old_next_state_values = self.critic(next_states).squeeze()
        advantages = self.calculate_GAE(rewards, old_state_values, old_next_state_values, masks)

        # critic loss target
        if self.args.use_returns:  # true
            targets = returns
        else:
            targets = advantages + old_state_values
        # PPO
        with torch.no_grad():
            old_log_probs, _ = self.evaluate_action(states, actions)

        for i in range(self.args.K_epoch):
            batches = self.buffer.get_batches()
            # batches = torch.as_tensor(batches, dtype=torch.int64)
            for batch in batches:
                if len(batch) != self.args.batch_size:
                    continue
                new_log_probs, prob_entropy = self.evaluate_action(states[batch], actions[batch])
                ratio = (new_log_probs - old_log_probs[batch]).exp()
                surr1 = ratio * advantages[batch]
                surr2 = ratio.clamp(1 - self.args.clip, 1 + self.args.clip) * advantages[batch]
                actor_loss = - torch.min(surr1, surr2).mean()

                # entropy
                if self.args.use_entropy:  # false
                    actor_loss += - self.args.entropy_coefficient * prob_entropy.mean()

                new_state_values = self.critic(states[batch]).squeeze()
                # value clip
                if self.args.use_clipped_value:  # false
                    critic_loss1 = 0.5 * (
                            old_state_values[batch] + torch.clamp(new_state_values - old_state_values[batch],
                                                                  - self.args.clip, self.args.clip) - targets[
                                batch]) ** 2
                    critic_loss2 = 0.5 * (new_state_values - targets[batch]) ** 2
                    critic_loss = torch.max(critic_loss1, critic_loss2).mean()
                else:
                    critic_loss = F.mse_loss(new_state_values, targets[batch])

                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                actor_loss.backward()
                critic_loss.backward()

                # gradient clip
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.args.clip_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.args.clip_grad_norm)

                self.actor_optimizer.step()
                self.critic_optimizer.step()
        self.buffer.clear()

    def calculate_returns(self, rewards, masks):
        returns = []
        return_ = 0
        for reward, mask in zip(reversed(rewards), reversed(masks)):
            return_ = reward + self.args.gamma * return_ * mask
            returns.append(return_)
        returns.reverse()
        returns = torch.as_tensor(returns, dtype=torch.float32)
        return returns

    def calculate_GAE(self, rewards, state_values, next_state_values, masks):
        advantage = 0
        advantages = []
        for reward, value, next_value, mask in zip(reversed(rewards), reversed(state_values),
                                                   reversed(next_state_values), reversed(masks)):
            # TD-error
            delta = reward + self.args.gamma * next_value * mask - value
            advantage = delta + self.args.gamma * self.args.lambda_ * advantage * mask
            advantages.append(advantage)
        advantages.reverse()
        advantages = torch.as_tensor(advantages, dtype=torch.float32)
        return advantages

    def do_store_reward(self, time, reward):
        self.reward_store[time] = reward

    def do_store_delay(self, time, delay):
        self.delay_store[time] = delay

    def reset_store(self):
        self.reward_store = np.zeros([self.n_time])
        self.delay_store = np.zeros([self.n_time])
