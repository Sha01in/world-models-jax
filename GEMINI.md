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
*   **Training:** Trained on 10k random rollouts. Reconstruction loss + KL Divergence.

### Memory (M) - `src/rnn.py`
*   **Type:** MDN-RNN (LSTM + Mixture Density Network).
*   **Inputs:** Latent $z_t$, Action $a_t$.
*   **Outputs:**
    *   Parameters for 5 Gaussians ($\mu, \log\sigma, \pi$) predicting $z_{t+1}$.
    *   Predicted Reward $r_{t+1}$ (used for training Controller).
    *   Predicted Done $d_{t+1}$ (to stop dreams).
*   **Training:** Trained on "Mixed" Dataset (Good + Bad + Recovery + Aggressive).
*   **Loss:** Negative Log Likelihood (NLL) for $z$, MSE for $r$, BCE for $d$.

### Controller (C) - `src/controller.py`
*   **Type:** Simple Single-Layer Perceptron (Linear).
*   **Input:** Concatenation of $[z_t, h_t]$.
*   **Output:** Action $[Steer, Gas, Brake]$.
*   **Training:** Evolution Strategy (CMA-ES) inside the "Dream" environment.

## 4. Data Strategy (Sim2Real2Sim)
To bridge the gap between the RNN's hallucinations and reality, we use a diverse dataset:
1.  **Random (Brownian):** Smooth random driving to cover the state space.
2.  **Iterative Failure:** Data collected from previous failing agents to teach the RNN about "death spins."
3.  **Recovery (Heuristic + Noise):** Heuristic driver with random perturbations to teach recovery.
4.  **Aggressive (Fast Entry):** Heuristic driver that enters corners too fast to teach friction limits.
5.  **Augmentation:** All data is horizontally flipped (Mirrored) to double the dataset and remove left-turn bias.

## 5. Current Status
*   **VAE:** Functional. Good reconstructions.
*   **RNN:** Functional. Able to predict track curvature and negative rewards for grass.
*   **Controller:** Evolved in Dream to score ~800+.
*   **Real World Performance:**
    *   **Success:** Can navigate simple tracks and some corners.
    *   **Issues:** Still struggles with sharp "Hairpin" turns and sometimes cuts corners too aggressively, leading to spins.
    *   **Diagnosis:** The "Dream" might be slightly more permissive than `CarRacing-v3` physics (especially regarding grass friction), causing the agent to be overconfident.

## 6. Usage
*   `python test_agent.py`: Run the final agent and save video.
*   `python train_dream.py`: Evolve the controller in the RNN.
*   `python train_rnn.py`: Retrain the World Model.
