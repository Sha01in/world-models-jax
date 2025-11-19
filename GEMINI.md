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
    *   `cma` (Evolution Strategy - `evosax` failed due to dependencies).
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

## 4. Data Pipeline
We faced significant issues with "Depressed RNNs" (predicting constant failure) due to the difficulty of `CarRacing-v3` compared to the paper's `v0`.

1.  **Collection (`01_data_collection_*.py`):**
    *   **Heuristic Data:** 500 episodes using a CV-based lane-following script. Filtered for Score > 20.
    *   **Failure Data:** 300 episodes of random/erratic driving to teach the RNN about crashes/off-road penalties.
2.  **Processing (`03_process_data.py`):**
    *   **Normalization:** Images / 255.0.
    *   **Augmentation:** **Mirroring**. We double the dataset by flipping images horizontally and negating steering. This cured the "Left Turn Bias."
    *   **Total Sequences:** ~2,100.

## 5. Current Status & The "Sim2Real" Gap
**The agent is a "Dream Billionaire" but fails in reality.**

*   **Dream Performance:** ~900 Score (Perfect driving).
*   **Real Performance:** ~-50 Score (Spins out immediately).
*   **Visual Diagnosis:** The VAE reconstruction is decent (road is visible), but the agent turns too early or too late, hits grass, and enters an unrecoverable spin.

## 6. Chronology of Debugging & Fixes
1.  **Stationary Car:** Agent output negative gas (tanh).
    *   *Fix:* Scaled Gas output to [0, 1]. Forced Brake to 0.0.
2.  **Spinning Left:** RNN learned bias from counter-clockwise tracks.
    *   *Fix:* Implemented Mirror Augmentation in data processing.
3.  **Blindness:** VAE was trained only on random noise.
    *   *Fix:* Retrained VAE on Mixed (Heuristic + Crash) dataset for 30 epochs.
4.  **Zoom-In Bug:** CarRacing zooms camera for first 50 frames, confusing RNN.
    *   *Fix:* Added 50-frame "Warmup" in `test_agent.py` (Force straight drive).
5.  **Dream Overfitting:** RNN allows "Slot Car" physics (perfect turns with max steering).
    *   *Current Attempt:* Clamped Steering to `[-0.5, 0.5]` and locked Gas to `0.5` (Cruise Control) in `src/controller.py`.
    *   *Result:* Agent now spins in the *opposite* direction or fails to hold the turn.

## 7. Active Files for Context
*   **`train_dream.py`**: The evolution loop. Currently uses `TEMPERATURE = 1.15`.
*   **`src/controller.py`**: Defines the action constraints. Currently heavily constrained (Steer clamped, Gas constant).
*   **`test_agent.py`**: Runs the inference in Gym. Includes video recording and Action Repeat logic.

## 8. Immediate Next Steps / Hypotheses
*   **Hypothesis A (RNN Fidelity):** The RNN loss (~-20) might still be too high for `CarRacing-v3` physics. The "Dream" physics are too slippery or too grippy compared to reality.
*   **Hypothesis B (Controller Crippling):** The hard clamps on Steering/Gas might be *preventing* recovery. If the car slips, it needs >0.5 steer to catch the slide, but we clamped it.
*   **Hypothesis C (MDN Mode Collapse):** The RNN might be outputting a multi-modal prediction (Straight OR Turn), and sampling in the dream picks the "Lucky" mode, while reality picks the "Physics" mode.