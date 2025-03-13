import gym
import numpy as np
import torch
from algorithms.actor_critic.sac import SACD_agent
import os
import yaml
from types import SimpleNamespace as SN
from commons.utils import plot_figure

with open(os.path.join(os.path.dirname(__file__), "configs", "actor_critic", "sac.yaml"), "r") as f:
    try:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)
    except yaml.YAMLError as exc:
        assert False, "default.yaml error: {}".format(exc)
args = SN(**config_dict)


def test_agent(env, agent):
    episode_reward = 0
    for i in range(args.test_num):
        state = env.reset()
        done = False
        while not done:
            action = agent.select_action(state, deterministic=True)
            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            state = next_state
    return episode_reward / args.test_num


if __name__ == '__main__':
    train_env = gym.make(args.env_name)
    test_env = gym.make(args.env_name)

    train_env.seed(args.seed)
    test_env.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    args.state_dim = train_env.observation_space.shape[0]
    args.action_dim = train_env.action_space.n

    agent = SACD_agent(args)
    episode_reward_history = []
    current_steps = 0

    while current_steps < args.total_steps:
        state = train_env.reset()
        done = False
        while not done:
            current_steps += 1
            if current_steps < 1e4:
                action = train_env.action_space.sample()
            else:
                action = agent.select_action(state, deterministic=False)
            next_state, reward, done, info = train_env.step(action)
            # buffer
            agent.replay_buffer.add(state, action, reward, next_state, done)

            state = next_state

            if (current_steps >=
                    1e4 and current_steps % args.train_steps == 0):
                for j in range(args.train_steps):
                    agent.train()
            if current_steps % args.test_steps == 0:
                average_episode_reward = test_agent(test_env, agent)
                episode_reward_history.append(average_episode_reward)
                print(f'| step : {current_steps:6} | Episode Reward: {average_episode_reward:5.1f} |')
    file_name = f'SAC.png'
    plot_figure(episode_reward_history, "Episode", "Reward", file_name)
