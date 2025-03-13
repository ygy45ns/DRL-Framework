# DRL Framework

## 📌 Introduction
**DRL Framework** is a deep reinforcement learning (DRL) framework that implements multiple popular DRL algorithms. It supports **Actor Critic-based** and **Value-based** algorithms, with configuration files and runnable scripts to help users quickly get started and extend the framework.

## 🚀 Supported Algorithms
### 1️⃣ **Actor Critic-Based Methods**
- A2C (Advantage Actor-Critic)
- AC (Actor-Critic)
- PPO (Proximal Policy Optimization)
- SAC (Soft Actor-Critic)

### 2️⃣ **Value-Based Methods**
- DQN (Deep Q-Network)
- DDQN (Double DQN)

## 📂 Project Structure

```
DRL-Framework/
├── algorithms/
|	├──actor_critic/          # Actor-Critic Algorithm implementation
│  |	├── a2c.py             # Advantage Actor-Critic
│  |	├── ppo.py             # Proximal Policy Optimization
│  |	├── sac.py             # Soft Actor-Critic
│  |	└── ac.py              # Actor-Critic
|  |
|  └── value_based/           # Value-Based Algorithm implementation
│   	├── dqn.py             # Deep Q-Network
│   	└── ddqn.py            # Double DQN
|
├── configs/               # Hyperparameter Configuration
│   ├── actor_critic/      # AC-based
│   └── value_based/       # Value-based
|
├── commons/               # Common Module
│   └── utils.py           # Plot
│
├── scripts/               # Run
│   ├── run_a2c.py
│   ├── run_ppo.py
│   └── ...				   # (others)
└── README.md              # Description
```

## ⚙️ Install

### Environmental Requirements

- Python ≥ 3.8.0
- PyTorch ≥ 2.1.2
- Gymnasium ≥ 1.0.0
- Matplotlib>=3.5.1
- Numpy>=1.24.1
- ...

### Installation Steps

1. Clone repository

   ```
   git clone https://github.com/your-username/DRL-Framework.git
   cd DRL-Framework
   ```

2. Install dependencies

   ```
   pip install -r requirements.txt
   ```
