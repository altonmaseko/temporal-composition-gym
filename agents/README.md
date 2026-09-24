# Reinforcement Learning Agents

This directory contains the training scripts for the algorithmic baselines used to solve the Temporal Composition Gym environments. All scripts are designed to be run from the command line and support dynamically swapping environments via arguments.

---

## 1. Deep Q-Network (DQN)

The `train_dqn.py` script implements the standard memory-less baseline for discrete action spaces using `stable-baselines3`. 

> 📁 [`train_dqn.py`](train_dqn.py)

### Implementation Details
The script dynamically switches between a standard Multilayer Perceptron (`MlpPolicy`) for vector observations and a Convolutional Neural Network (`CnnPolicy`) for image observations. 

```python
# Environment-specific configurations inside train_dqn.py
config = {
    "TemporalComp/HazardousDelivery-v0": {
        "policy": "MlpPolicy",
        "learning_rate": 1e-3,
        "buffer_size": 50000
    },
    "TemporalComp/SequentialColour-v0": {
        "policy": "CnnPolicy",
        "learning_rate": 1e-4,
        "buffer_size": 100000
    }
}
```

### Supported Environments
Because DQN inherently relies on a discrete action space, it cannot be run on Continuous Control environments. It can run on:
- `TemporalComp/HazardousDelivery-v0`
- `TemporalComp/SequentialColour-v0`

### How to Run
```bash
python agents/train_dqn.py --env TemporalComp/HazardousDelivery-v0 --timesteps 100000
```

---

## 2. Proximal Policy Optimization (PPO)

The `train_ppo.py` script implements the industry-standard memory-less baseline for both discrete and continuous action spaces using `stable-baselines3`.

> 📁 [`train_ppo.py`](train_ppo.py)

### Implementation Details
PPO is an on-policy actor-critic algorithm. Like DQN, our script dynamically selects the correct feature extractor based on the environment, passing it into the initialization block.

```python
model = PPO(
    policy=env_settings["policy"],
    env=env,
    learning_rate=env_settings["learning_rate"],
    n_steps=env_settings["n_steps"],
    batch_size=env_settings["batch_size"],
    verbose=1
)
```

### Supported Environments
PPO handles both discrete and continuous action spaces robustly, making it compatible with all environments:
- `TemporalComp/HazardousDelivery-v0`
- `TemporalComp/DeferredMaintenance-v0`
- `TemporalComp/SequentialColour-v0`

### How to Run
```bash
python agents/train_ppo.py --env TemporalComp/DeferredMaintenance-v0 --timesteps 100000
```

---

## 3. Recurrent PPO (PPO + LSTM)

The `train_recurrent_ppo.py` script augments standard PPO with implicit memory using `sb3-contrib`. This is our core implicit baseline to test how well Long Short-Term Memory (LSTM) networks can infer and retain temporal logic compared to explicit Automata models.

> 📁 [`train_recurrent_ppo.py`](train_recurrent_ppo.py)

### Implementation Details
The algorithm replaces the standard feedforward policies with LSTM variants (`MlpLstmPolicy` and `CnnLstmPolicy`). It unrolls trajectories during optimization to pass gradients backward through time.

```python
from sb3_contrib import RecurrentPPO

model = RecurrentPPO(
    policy=env_settings["policy"], # e.g., 'MlpLstmPolicy'
    env=env,
    learning_rate=env_settings["learning_rate"],
    n_steps=env_settings["n_steps"],
    verbose=1
)
```

### Supported Environments
Like standard PPO, it supports all observation and action spaces in the Gym:
- `TemporalComp/HazardousDelivery-v0`
- `TemporalComp/DeferredMaintenance-v0`
- `TemporalComp/SequentialColour-v0`

### How to Run
```bash
python agents/train_recurrent_ppo.py --env TemporalComp/HazardousDelivery-v0 --timesteps 100000
```

---

## 4. Deep Recurrent Q-Network (DRQN)

The `train_drqn.py` script implements the implicit sequence model for discrete environments, following a clean, single-file PyTorch architecture.

> 📁 [`drqn/train_drqn.py`](drqn/train_drqn.py)

### Implementation Details
Standard DQN requires Markovian states (all necessary information is in the current observation). DRQN swaps the inner fully connected layers of DQN with an LSTM and utilizes a custom `EpisodicReplayBuffer`. This allows the network to learn by unrolling trajectories via Backpropagation Through Time (BPTT).

```python
class DRQN(nn.Module):
    def __init__(self, env):
        # ...
        self.fc1 = nn.Linear(obs_shape, 64)
        self.lstm = nn.LSTM(64, 64, batch_first=True)
        self.fc2 = nn.Linear(64, env.action_space.n)
```

### Supported Environments
Currently, this custom implementation supports discrete observation arrays and discrete action spaces:
- `TemporalComp/HazardousDelivery-v0`

### How to Run
```bash
python agents/drqn/train_drqn.py --env-id TemporalComp/HazardousDelivery-v0 --total-timesteps 100000
```

---

## 5. Q-Learning for Reward Machines (QRM)

The `train_qrm.py` script introduces explicit temporal composition. Instead of forcing an LSTM to memorize the past implicitly, QRM utilizes a finite state automaton (the Reward Machine) to track task progress.

> 📁 [`qrm/train_qrm.py`](qrm/train_qrm.py)

### Implementation Details
QRM requires an Environment Wrapper (`rm_wrapper.py`) that translates raw environmental state changes into logical propositions. The QRM network then simultaneously learns a separate Q-function for *every* possible state in the Reward Machine, updating them all in parallel via off-policy counterfactuals.

```python
# The QRM Network outputs Q-values for all Reward Machine states at once
self.fc_q = nn.Linear(64, num_rm_states * num_actions)

# ...
# Counterfactual off-policy updates: 
# "What would the reward be if the agent was in state u_i?"
for u_i in range(rm.num_states):
    nxt_u, rw, dn = rm.step(u_i, b_props[i])
    # Compute TD error for state u_i...
```

### Supported Environments
Requires a discrete action space and an environment emitting logical propositions:
- `TemporalComp/HazardousDelivery-v0`

### How to Run
```bash
python agents/qrm/train_qrm.py --env-id TemporalComp/HazardousDelivery-v0 --total-timesteps 100000
```

---

## 6. Proximal Policy Optimization for Reward Machines (PPO-RM)

The `train_ppo_rm.py` script applies explicit automata modeling to the industry-standard on-policy PPO algorithm.

> 📁 [`ppo_rm/train_ppo_rm.py`](ppo_rm/train_ppo_rm.py)

### Implementation Details
Because PPO is strictly on-policy, it cannot perform counterfactual off-policy updates like QRM. Instead, PPO-RM uses a clever `PPORMObservationWrapper` that appends the current state of the Reward Machine as a one-hot vector directly onto the agent's observation. This essentially translates the non-Markovian task back into a fully observable Markovian process, allowing standard `stable-baselines3` PPO to solve it effortlessly.

```python
class PPORMObservationWrapper(gym.Wrapper):
    def _get_obs(self, obs):
        one_hot = np.zeros(self.rm.num_states, dtype=np.float32)
        if self.u != -1: # active state
            one_hot[self.u] = 1.0
        return np.concatenate([obs, one_hot])
```

### Supported Environments
Because it wraps standard PPO, it supports both continuous and discrete environments, provided they emit logical propositions:
- `TemporalComp/HazardousDelivery-v0`

### How to Run
```bash
python agents/ppo_rm/train_ppo_rm.py --env-id TemporalComp/HazardousDelivery-v0 --timesteps 100000
```
