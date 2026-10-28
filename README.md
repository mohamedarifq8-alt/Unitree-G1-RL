<h1 align="center">🚀 Unitree G1 Humanoid Locomotion Control via Deep Reinforcement Learning</h1>

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![JAX](https://img.shields.io/badge/JAX-High_Performance_Computing-9A5BCC?logo=python&logoColor=white)](https://github.com/google/jax)
[![Brax](https://img.shields.io/badge/Brax-Hardware_Accelerated_RL-FF6F00)](https://github.com/google/brax)
[![MuJoCo](https://img.shields.io/badge/MuJoCo-Physics_Engine-1A1A1A?logo=google&logoColor=white)](https://mujoco.org/)
[![Stable Baselines3](https://img.shields.io/badge/Algorithms-PPO-4B8BBE)]()
[![Kaggle](https://img.shields.io/badge/Kaggle-Cloud_GPUs-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/)

<p align="center">
  <b>Humanoid Locomotion Control for the Unitree G1 Robot via Deep Reinforcement Learning — Fully Trained on Kaggle</b>
</p>

---

## 📑 Table of Contents

- [1. Project Overview](#overview)
- [2. Tech Stack & Keywords](#tech-stack)
- [3. PPO Hyperparameters Architecture](#ppo)
- [4. Reward Shaping Strategy](#reward-shaping)
- [5. Curriculum & Learning Rate Annealing](#curriculum)
- [6. 📊 Training Stages & Performance Evolution](#stages)
- [7. Challenges & Debugging](#challenges)
- [8. Future Architecture](#future)

---
<img width="450" height="300" alt="Screenshot 1" src="https://github.com/mohamedarifq8-alt/Unitree-G1-RL/blob/main/videos/Screenshot%20(1).png?raw=true" />
<img width="450" height="300" alt="Screenshot 2" src="https://github.com/mohamedarifq8-alt/Unitree-G1-RL/blob/main/videos/Screenshot%20(2).png?raw=true" />

<a name="overview"></a>
## 1. 🌐 Project Overview

Development and training of an advanced **Locomotion Controller** for the **Unitree G1** humanoid robot using **Deep Reinforcement Learning** algorithms. The project was implemented entirely on the **Kaggle** cloud computing platform, leveraging **GPUs** through a fully GPU-accelerated physics simulation pipeline (**End-to-End GPU Simulation**).

This enabled running **4,096 parallel environments** and training the robot over **200 million timesteps** with exceptional computational efficiency. The project produced a **control policy** characterized by dynamic stability, robustness against external disturbances, energy efficiency, and accurate **Joystick Tracking** of steering commands.

---

<a name="tech-stack"></a>
## 2. 🛠️ Tech Stack & Keywords

| Domain | Technologies |
|---|---|
| **Compute Platform** | Kaggle Cloud GPUs |
| **Algorithms** | Proximal Policy Optimization (PPO), Continuous Control, Hierarchical Reinforcement Learning (HRL) |
| **Simulation & Physics** | MuJoCo, Domain Randomization, Sim2Real Transfer |
| **Frameworks** | JAX, Brax, End-to-End GPU Simulation, Vectorized Environments (vmap), Parallel Computing |
| **Model Engineering** | Reward Shaping, Learning Rate Annealing, State Machine Control, Policy Gradients |

---

<a name="ppo"></a>
## 3. ⚙️ PPO Hyperparameters Architecture

To achieve this stability, the **Hyperparameters** were tuned with high engineering precision. The code below shows the training engine settings, followed by a mathematical explanation of the key variables:

```python
sys.argv = [
    'train_jax_ppo',
    '--env_name=G1JoystickFlatTerrain',
    '--num_envs=4096',
    '--num_eval_envs=64',
    '--policy_hidden_layer_sizes=512,256,128',
    '--value_hidden_layer_sizes=512,256,128',
    '--policy_obs_key=state',
    '--value_obs_key=privileged_state',
    '--unroll_length=20',
    '--batch_size=256',
    '--num_minibatches=16',
    '--num_updates_per_batch=4',
    '--num_evals=10',
    f'--num_timesteps={config["timesteps"]}',
    '--episode_length=1000',
    '--learning_rate=3e-4',
    '--entropy_cost=0.005',
    '--discounting=0.97',
    '--clipping_epsilon=0.2',
    '--reward_scaling=1.0',
    '--max_grad_norm=1.0',
    '--num_videos=0',
    f'--logdir={BASE_DIR}'
]
```

### 📖 Key Parameter Explanations

- **`--num_envs=4096`** — Harnessing JAX's `vmap` to spawn **4,096 robots** training in parallel inside GPU memory simultaneously, which massively accelerated data collection.
- **`--policy_hidden_layer_sizes=512,256,128`** — The deep neural network architecture (**Actor Network**) responsible for making locomotion decisions, designed with tapering layers to extract complex physical features.
- **`--clipping_epsilon=0.2`** — The heart of the PPO algorithm; it acts as a safety valve that prevents the neural network from updating its weights in a drastic, destructive manner, protecting the robot from forgetting its balance skill during training.
- **`--entropy_cost=0.005`** — The "curiosity" coefficient. It was pinned at this value to prevent the policy from becoming rigid, allowing the robot to invent flexible motions to evade random push disturbances (**Pushes**).
- **`--discounting=0.97`** — The discount factor (**Gamma**), tuned to assign high importance to future rewards (avoiding falls later on) rather than only immediate rewards.

---

<a name="reward-shaping"></a>
## 4. 🎯 Reward Shaping Strategy

The reward function was designed to overcome erratic robot behaviors (**Jittering**) and laziness, by imposing strict constraints:

- **Eliminating passive survival bonuses (`alive = 0.0`)** — Forcing the network to earn points exclusively through accurately tracking walking commands.
- **Harsh falling penalty (`termination = -100.0`)** — Making balance a non-negotiable top priority to guarantee survival.
- **Energy consumption constraints** — Applying precise penalties on excessive motor torque output (`torques = -0.0001`) and random joint acceleration (`dof_acc = -2.5e-7`), resulting in a smooth, natural-looking gait that reduces mechanical stress.

---

<a name="curriculum"></a>
## 5. 📈 Curriculum & Learning Rate Annealing

To avoid the phenomenon of "**Catastrophic Forgetting**" and ensure neural weight stability, the **200 million timesteps** were split into **5 connected training stages** using a precise descending learning rate schedule:

| Stage | Name | Learning Rate (LR) | Objective |
|:---:|---|:---:|---|
| 1 | **Foundation** | `3e-4` | Building the core ability to balance and avoid falling. |
| 2 | **Command Tracking** | `1.5e-4` | Improving responsiveness to speed and direction steering commands. |
| 3 | **Dynamic Stabilization** | `5e-5` | Mastering lateral maneuvers and resisting random push disturbances (Pushes). |
| 4 | **Micro Hardening** | `1e-5` | Reaching the valley of convergence and cementing skills against perturbation. |
| 5 | **Final Polish** | `5e-6` | Microscopic weight refinements to lower energy consumption (Torque Smoothing) and achieve maximum motion fluidity and economy. |

---

<a name="stages"></a>
## 6. 📊 Training Stages & Performance Evolution

Training was designed as **5 connected stages** (totaling **200 million timesteps**) to ensure locomotion stability and avoid catastrophic forgetting, with a **gradual learning rate decay** (Learning Rate Annealing):

| # | Stage | LR | Reward (Peak ➔ Final) | Engineering Milestone & Outcome |
|:---:|---|:---:|:---:|---|
| 1 | **Foundation** | `3e-4` | `1.9 ➔ 2.4` | Building the basic walking skeleton and surviving falls through strict penalties (`termination=-100`). |
| 2 | **Refinement** | `1.5e-4` | `10.1 ➔ 7.4` | A massive leap in Joystick command tracking, with emerging locomotion oscillation due to the widening of update steps. |
| 3 | **Stabilization** | `5e-5` | `13.8 ➔ 11.1` | Dynamic stability, mastering lateral maneuvers, and avoiding falls during sudden direction changes. |
| 4 | **Smoothing** | `1e-5` | `16.3 ➔ 13.7` | Reaching full convergence, with excellent resistance to random push disturbances (Pushes). |
| 5 | **Final Polish** | `5e-6` | `18.6 ➔ 16.4` | Microscopic weight polishing to reduce energy consumption (Torque Smoothing) and achieve a silky-smooth gait. |

---

<a name="challenges"></a>
## 7. 🧩 Challenges & Debugging

- **Overcoming the CPU-GPU Bottleneck:**
  `jax.vmap` was used to fuse **4,096 simulation environments** into a single parallel processing array inside GPU memory, eliminating data transfer latency (CPU-GPU Bottleneck) and massively speeding up the training cycle.

- **Fixing the Checkpoint Loading Bug:**
  When resuming training between stages, a critical software bug emerged that relied on file creation dates (**OS Timestamps**) on the Kaggle OS after unzipping, causing old weights to be loaded by mistake and threatening to wipe the training memory. The bug was identified, and the search algorithm was rewritten to rely on **strict alphabetical ordering of folders** instead of timestamps, guaranteeing that the latest weights are always invoked and training resumes exactly from the correct point.

- **Video Recording & Random Tracking Camera Issue:**
  During the results extraction phase for policy evaluation, the robot moved fast and drifted out of the fixed camera frame, making training documentation impossible. The problem was solved programmatically by diving into the environment settings and enabling the **Track Camera** to dynamically bind to the robot's **Center of Mass**, keeping the robot centered in frame and documenting the maneuvers with cinematic precision.

---

<a name="future"></a>
## 8. 🔮 Future Architecture

The current model (**Walking_Controller**) represents the core brain within a multi-task control architecture (**State Machine**). Work is currently underway to develop an independent environment for training the "**Stand-Up Expert**" (**StandUp_Controller**) for autonomous recovery from falls. The two brains will be fused through a central controller based on **IMU Sensor Fusion** for dynamic, instantaneous switching between control policies — culminating in a fully autonomous robot system that is robust in real-world environments.

---

## 👨‍💻 Author
**Mohammed Arif Mahyoub Haider**

*Electrical Engineer - Computer and Industrial Control*

---
<p align="center">
  <sub>🤖 Unitree G1 | PPO + MuJoCo + JAX/Brax | Trained on Kaggle — 4096 Parallel Environments | 200M Timesteps</sub>
</p>
