# World Models (JAX) on CarRacing-v3

A JAX/Equinox implementation of [World Models (Ha & Schmidhuber, 2018)](https://worldmodels.github.io/) applied to the Gymnasium `CarRacing-v3` environment.

![Agent View](debug_grid_ep4.png)
*(Visualization of Real Observation vs. VAE Reconstruction vs. RNN Dream)*

## 1. Overview

The agent consists of three independent components trained sequentially:

1.  **Vision (V):** A Convolutional VAE that compresses the 64x64x3 game frame into a 32-dimensional latent vector ($z$).
2.  **Memory (M):** An MDN-RNN (LSTM + Mixture Density Network) that predicts the next latent state ($z_{t+1}$) and reward ($r_{t+1}$) given the current state and action. This serves as the agent's "Dream" world.
3.  **Controller (C):** A simple linear model that maps the concatenated state $[z_t, h_t]$ to actions. It is evolved using **CMA-ES** inside the RNN's dream environment.

## 2. Installation

### Prerequisites
*   Python 3.10+
*   CUDA-enabled GPU (Recommended)

### Setup
```bash
# Clone the repo
git clone https://github.com/yourusername/world-models-jax.git
cd world-models-jax

# Install dependencies
pip install -r requirements.txt
```

## 3. Usage Pipeline

The World Model is trained in a strict pipeline. Each step depends on the previous one.

### Step 1: Data Collection
Collect initial random data to train the Vision model.
```bash
python collect_data.py
# Select Option 1: Random (Brownian Noise)
```
*Goal: ~10,000 episodes.*

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

To achieve high scores (>800), you must close the **Sim2Real Gap**. The RNN often hallucinates that driving on grass is safe. To fix this:

1.  **Collect Failure Data:** Run `collect_data.py` (Option 5: On-Policy) to collect data where the agent fails.
2.  **Re-Process Data:** Run `process_data.py` to include the new data.
3.  **Retrain RNN:** Run `train_rnn.py`. The model will learn that those states lead to low rewards.
4.  **Retrain Controller:** Run `train_dream.py` with the smarter World Model.

## 5. Results

*   **Episode Score:** 843.0 (Solved)
*   **Behavior:** Robust navigation of sharp turns; recovery from minor slips.

## 6. Credits
*   Original Paper: [World Models](https://arxiv.org/abs/1803.10122) by David Ha and Juergen Schmidhuber.
*   Environment: [Gymnasium](https://github.com/Farama-Foundation/Gymnasium).
*   Framework: [JAX](https://github.com/google/jax) & [Equinox](https://github.com/patrick-kidger/equinox).

