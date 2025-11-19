# Project Context: World Models Reproduction (JAX)

## 1. Project Overview
**Goal:** Reproduce "World Models" (Ha & Schmidhuber, 2018) using JAX/Equinox on `CarRacing-v3`.
**Core Logic:** Train a VAE (Vision) and MDN-RNN (Memory) on collected data. Use the RNN as a differentiable simulator ("Dream") to evolve a Linear Controller via CMA-ES. Transfer the controller back to the real Gym environment.

## 2. Environment & Stack
*   **OS:** Windows 10 via **WSL2 (Ubuntu)**.
*   **Hardware:** RTX 4070 Ti (CUDA 12).
*   **Python:** 3.11 (Managed via `uv`).
*   **Libraries:**
    *   `jax`, `jaxlib` (CUDA enabled).
    *   `equinox` (Neural Networks).
    *   `optax` (Optimizers).
    *   `gymnasium[box2d]` (Environment, v3).
    *   `cma` (Evolution Strategy).
    *   `opencv-python`, `numpy`.

## 3. Architecture
### Vision (V) - `src/vae.py`
*   **Input:** 64x64x3 RGB Images.
*   **Type:** Convolutional VAE.
*   **Latent Dim ($z$):** 32.
*   **Training:** Reconstruction Loss (MSE) + KL Divergence.

### Memory (M) - `src/rnn.py`
*   **Type:** MDN-RNN (LSTM + Mixture Density Network Head).
*   **Hidden Size:** 256.
*   **Input:** Concatenation of $z_t$ (32) + $a_t$ (3).
*   **Outputs:**
    *   $\pi, \mu, \sigma$ (Parameters for 5 Gaussians modeling $z_{t+1}$).
    *   Reward (Scalar prediction).
    *   Done (Binary logit prediction).

### Controller (C) - `src/controller.py`
*   **Type:** Simple Linear Layer (Single Matrix Multiply).
*   **Input:** Concatenation of $z_t$ (32) + $h_t$ (256).
*   **Output:** Action vector (Steering, Gas, Brake).
*   **Optimization:** CMA-ES maximizing cumulative reward inside the RNN "Dream".

## 4. Success Story: Closing the Sim2Real Gap
We successfully trained an agent that drives in the real `CarRacing-v3` environment.

### The Challenge
Initial attempts resulted in a "Sim2Real" gap where the agent would achieve perfect scores (~900) in the RNN Dream but fail immediately (< -50) in reality. The primary issues were:
1.  **Reward Hallucination:** The RNN learned to predict positive rewards even when predicting "crash" visuals.
2.  **Data Distribution:** The dataset was bimodal (Perfect Heuristic Driving vs. Terrible Random Crashes). It lacked "recovery" data (sliding but saving it).
3.  **Controller Constraints:** Hard-coded clamps prevented the agent from making sharp corrections.

### The Solution
1.  **Data Augmentation ("Sim2Real2Sim"):**
    *   **Parallel Data Collection:** Created parallelized scripts (`data_collection_*_parallel.py`) to speed up sampling by 16x. **Prefer these for all future collection.**
    *   Collected 1000 episodes of **Brownian Noise** driving (smooth random walks) to cover the state space.
    *   Collected 500 episodes of **Iterative Failure** (running the broken agent) to teach the RNN exactly what failure looks/feels like.
    *   Collected 500 episodes of **Recovery Data** (Heuristic driver with random perturbations) to teach the agent how to recover from slides.
    *   Applied **Mirror Augmentation** to all data to eliminate left-turn bias.

2.  **Loss Reweighting:**
    *   Modified `train_rnn.py` to heavily penalize Reward and Done errors.
    *   Weights: `MDN: 1.0`, `Reward: 10.0`, `Done: 10.0`.
    *   This forced the RNN to align its "feelings" (Reward) with its "sight" (Visual Latents).

3.  **Relaxed Controller:**
    *   Removed hard clamps on Steering.
    *   Allowed full range `[-1, 1]` steering and `[0, 1]` gas/brake.

### Results
*   **Dream Score:** ~780 (Mean ~570).
*   **Real Score:** consistently positive (> 50) in successful runs.
*   **Behavior:** The agent can now navigate turns and recover from minor slides.

## 5. Key Scripts
*   `train_vae.py`: Train Vision Model.
*   `train_rnn.py`: Train Memory Model (with new weighted loss).
*   `train_dream.py`: Evolve Controller (Best reward ~800).
*   `test_agent.py`: Run inference in Gym (with filmstrip debug and per-episode video).
*   `data_collection_*_parallel.py`: **PREFERRED** parallelized strategies for gathering training data.
