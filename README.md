# World Models (JAX) on CarRacing-v3

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/JAX-0.8+-orange.svg)](https://github.com/google/jax)
[![arXiv](https://img.shields.io/badge/arXiv-1803.10122-b31b1b.svg)](https://arxiv.org/abs/1803.10122)

A JAX/Equinox implementation of [World Models (Ha & Schmidhuber, 2018)](https://worldmodels.github.io/) applied to the Gymnasium `CarRacing-v3` environment.

![Agent View](docs/debug_grid.png)
*(Visualization of Real Observation vs. VAE Reconstruction vs. RNN Dream)*

## 1. Overview

The agent consists of three independent components trained sequentially:

1.  **Vision (V):** A Convolutional VAE that compresses the 64x64x3 game frame into a 32-dimensional latent vector ($z$).
2.  **Memory (M):** An MDN-RNN (LSTM + Mixture Density Network) that predicts the next latent state ($z_{t+1}$) and reward ($r_{t+1}$) given the current state and action. This serves as the agent's "Dream" world.
3.  **Controller (C):** A simple linear model that maps the concatenated state $[z_t, h_t]$ to actions. It is evolved using **CMA-ES** inside the RNN's dream environment.

### Project Structure
```
.
├── src/                 # Core model definitions (JAX/Equinox)
│   ├── vae.py           # VAE architecture (Vision)
│   ├── rnn.py           # MDN-RNN architecture (Memory)
│   └── controller.py    # Linear Controller (Policy)
├── scripts/             # Helper tools
│   ├── data_collection/ # Distributed rollout collection
│   └── tools/           # Debugging & visualization
├── train_dream.py       # Evolution Strategy (CMA-ES) loop
├── train_rnn.py         # World Model training script
├── process_data.py      # Data preprocessing pipeline
└── test_agent.py        # Final agent evaluation
```

## Sim2Real Strategy: Asymmetric Loss

This implementation introduces an **asymmetric loss function** that addresses the critical "Sim2Real gap" in World Models. The standard approach often leads to "optimism bias" where the RNN hallucinates safer outcomes than reality provides.

The asymmetric loss punishes overestimation of rewards (optimism) 5x more than underestimation (pessimism):

```python
# Penalize "Optimism" (Pred > Actual) significantly more than "Pessimism"
asymmetric_weight = jnp.where(diff > 0, 5.0, 1.0)
loss_reward = jnp.mean(asymmetric_weight * (diff ** 2))
```

This enables robust transfer from dream to reality, achieving human-level performance (843.0 score) without requiring massive datasets.

## 2. Installation

### Prerequisites
*   Python 3.11+
*   CUDA-enabled GPU (Recommended)
*   **Mac Users:** You may need to install `swig` to build Box2D (`brew install swig`).

### Setup
Using [uv](https://github.com/astral-sh/uv) (Recommended):
```bash
# Clone the repo
git clone https://github.com/Sha01in/world-models-jax.git
cd world-models-jax

# Initialize and sync environment
uv sync
```

### Verify Installation
Check if JAX can access your GPU:
```bash
python scripts/tools/check_gpu.py
```

Or using standard pip:
```bash
pip install -r requirements.txt
```

## 3. Usage Pipeline

The World Model is trained in a strict pipeline. Each step depends on the previous one.

### Step 1: Data Collection
Collect initial data to train the Vision model.
```bash
python collect_data.py
# Select Option 1: Random (Brownian Noise)
```
*Goal: ~2,000 - 5,000 episodes.*

*Note: Unlike the original paper which suggests 10,000 random episodes, I use a **Curriculum Learning** approach (see Section 4). I start with a smaller random dataset, train the agent, find where it fails, and then collect specific "failure" data. This is more efficient than random sampling.*

### Step 2: Train Vision (VAE)
Train the VAE to compress images.
```bash
python run_vae_training.py
```

### Step 3: Process Data
Encode all collected images into latent vectors ($z$) and save them for RNN training.
```bash
python process_data.py
```

### Step 4: Train Memory (RNN)
Train the MDN-RNN to predict the future.
```bash
python train_rnn.py
```
*Note: This implementation uses an **Asymmetric Loss** to punish "Optimism" (predicting high rewards when crashing), which fixes the Sim2Real gap.*

### Step 5: Train Controller (Dreaming)
Evolve the controller inside the RNN.
```bash
python train_dream.py
```

### Step 6: Test & Visualize
Run the trained agent in the real environment.
```bash
python test_agent.py
```

## 4. Iterative Improvement (Sim2Real2Sim)

To achieve high scores (>800) without needing massive random datasets, I use an iterative data collection strategy:

1.  **Recovery Data:** (Option 3) Heuristic driver with noise to teach the RNN how to recover from bad states.
2.  **Aggressive Data:** (Option 4) Heuristic driver that drives too fast, teaching the RNN about friction limits.
3.  **On-Policy Failures:** (Option 5) Run your current agent, let it crash, and add that data to the training set.

**Workflow:**
1.  Collect ~2k Random Episodes -> Train V, M, C.
2.  Observe Failures (e.g., Agent spins out on sharp turns).
3.  Collect ~500 "On-Policy" or "Aggressive" episodes.
4.  Retrain M (RNN) with the new data.
5.  Retrain C (Controller).
6.  Repeat.

This **Active Learning** loop fixes the "Sim2Real Gap" (where the RNN hallucinates that driving on grass is safe) much faster than simply adding more random data.

## 5. Results

### The Winning Recipe
To achieve the score of **843.0**, I used the following dataset composition (~4,000 episodes total):
*   **2,000 Random Episodes:** Initial training of V and M.
*   **500 Recovery Episodes:** Heuristic driver with noise (teaching recovery).
*   **500 Aggressive Episodes:** Heuristic driver entering corners too fast (teaching friction limits).
*   **500 On-Policy Failure Episodes:** **Critical Step.** I ran the agent, let it crash (due to "optimism delusions"), and added this specific data to the training set.

*   **Episode Score:** 843.0 (Solved)
*   **Behavior:** Robust navigation of sharp turns; recovery from minor slips.

*Note on Reproducibility: While the training scripts use fixed seeds for JAX operations (`PRNGKey(0)`), the data collection process (Sim2Real) involves real-time interaction with the Box2D physics engine, which can have non-deterministic elements across different hardware/OS. Exact score matching may vary, but the general learning curve should be consistent.*

## 6. Credits
*   Original Paper: [World Models](https://arxiv.org/abs/1803.10122) by David Ha and Juergen Schmidhuber.
*   Environment: [Gymnasium](https://github.com/Farama-Foundation/Gymnasium).
*   Framework: [JAX](https://github.com/google/jax) & [Equinox](https://github.com/patrick-kidger/equinox).

## 7. Technical Notes: The JAX Advantage

This project demonstrates a **Hybrid Architecture**:

1.  **GPU-Accelerated Dreaming:**
    The most significant advantage of JAX is in `train_dream.py`. I use `jax.vmap` to simulate the RNN "dreams" for the **entire population (256 agents)** simultaneously on the GPU. This turns the Evolution Strategy evaluation which is usually a slow sequential process into a single efficient batched operation.

2.  **CPU-Bound Reality:**
    Since `CarRacing-v3` is based on Box2D (CPU physics), the *real* environment interaction cannot be JIT-compiled. I handle this via:
    *   **Data Collection:** Standard Python `multiprocessing` to run parallel environments on CPU.
    *   **Inference:** Using JAX on CPU (worker threads) or GPU (main agent) depending on the bottleneck.

This approach leverages JAX where it excels (massive parallel simulation) while accommodating standard Gym environments.

