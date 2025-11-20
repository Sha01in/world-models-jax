import gymnasium as gym
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import cv2
import os
import time
import signal
from src.vae import VAE
from src.rnn import MDNRNN
from src.controller import get_action
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

# Settings
NUM_WORKERS = 12        # Reduced from 16 to 12 to prevent CPU starvation/Zombie issues
NUM_EPISODES = 500      # Total episodes needed
MAX_STEPS = 600
DATA_DIR = "data/rollouts_on_policy"
IMG_SIZE = 64
LATENT_DIM = 32
HIDDEN_SIZE = 256
ACTION_DIM = 3

# Paths
VAE_PATH = "checkpoints/vae.eqx"
RNN_PATH = "checkpoints/rnn.eqx"
CONTROLLER_PATH = "checkpoints/controller_dream.npz"

def collect_episode_worker(seed):
    """
    Worker function to collect a single episode.
    Running in a separate process with its own JAX CPU instance.
    """
    # 1. Force JAX to use CPU to avoid GPU contention
    jax.config.update("jax_platform_name", "cpu")
    
    # 2. Load Models (Fresh copy for this process)
    key = jax.random.PRNGKey(0)
    
    vae = VAE(latent_dim=LATENT_DIM, key=key)
    vae = eqx.tree_deserialise_leaves(VAE_PATH, vae)
    
    rnn = MDNRNN(latent_dim=LATENT_DIM, action_dim=ACTION_DIM, 
                 hidden_size=HIDDEN_SIZE, key=key)
    rnn = eqx.tree_deserialise_leaves(RNN_PATH, rnn)
    
    controller_data = np.load(CONTROLLER_PATH)
    controller_params = jnp.array(controller_data['params'])
    
    # 3. Define JIT functions specific to this process
    @jax.jit
    def encode(img):
        # img: (64, 64, 3) -> (3, 64, 64)
        x = jnp.array(img, dtype=jnp.float32) / 255.0
        x = jnp.transpose(x, (2, 0, 1))
        # No batch dimension needed for single instance inference
        features = vae.encoder(x)
        features = jnp.reshape(features, (-1,))
        mu = vae.mu_head(features)
        return mu

    @jax.jit
    def rnn_step(z, a, h, c):
        rnn_in = jnp.concatenate([z, a], axis=0)
        (log_pi, mu, log_sigma, r_pred, d_pred), (h_new, c_new) = rnn(rnn_in, (h, c))
        # Expected Z
        pi = jnp.exp(log_pi)
        expected_z = jnp.sum(pi * mu, axis=0)
        return h_new, c_new, expected_z

    @jax.jit
    def decide_action(z, h):
        return get_action(controller_params, z, h)

    # 4. Simulation Loop
    try:
        # Unique seed
        np.random.seed(seed)
        
        # Create Env
        env = gym.make("CarRacing-v3", render_mode="rgb_array")
        obs, _ = env.reset()
        
        h = jnp.zeros(HIDDEN_SIZE)
        c = jnp.zeros(HIDDEN_SIZE)
        
        obs_seq, action_seq, reward_seq, done_seq = [], [], [], []
        
        # Warmup Action
        current_action = np.array([0.0, 0.5, 0.0], dtype=np.float32)
        
        for t in range(MAX_STEPS):
            # Resize
            obs_small = cv2.resize(obs, (IMG_SIZE, IMG_SIZE))
            obs_seq.append(obs_small)
            
            # Inference
            z = encode(obs_small)
            
            if t < 50:
                # Warmup: Drive straight
                action = np.array([0.0, 0.5, 0.0], dtype=np.float32)
            else:
                # Policy
                action_jax = decide_action(z, h)
                action = np.array(action_jax)
            
            action_seq.append(action)
            
            # Step
            obs, reward, term, trunc, _ = env.step(action)
            reward_seq.append(reward)
            done_seq.append(term or trunc)
            
            # Update Memory
            h, c, _ = rnn_step(z, jnp.array(action), h, c)
            
            if term or trunc:
                break
        
        env.close()
        
        # Save Data
        # Use seed as unique ID
        save_path = os.path.join(DATA_DIR, f"ep_{seed}.npz")
        np.savez_compressed(save_path,
                            obs=np.array(obs_seq),
                            actions=np.array(action_seq),
                            rewards=np.array(reward_seq),
                            dones=np.array(done_seq))
        return True

    except Exception as e:
        print(f"Worker {seed} failed: {e}")
        return False

def main():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    
    print(f"Starting Distributed Data Collection.")
    print(f"Workers: {NUM_WORKERS}")
    print(f"Target: {NUM_EPISODES} episodes")
    print("JAX Mode: CPU (Forced per worker)")
    
    # Use a set of seeds as task IDs
    seeds = list(range(int(time.time()), int(time.time()) + NUM_EPISODES))
    
    # Use mp_context='spawn' to ensure clean process start without inheriting JAX state
    import multiprocessing as mp
    ctx = mp.get_context('spawn')
    
    with ProcessPoolExecutor(max_workers=NUM_WORKERS, mp_context=ctx) as executor:
        results = list(tqdm(executor.map(collect_episode_worker, seeds), total=NUM_EPISODES))
        
    success_count = sum(results)
    print(f"Collection Complete. Success: {success_count}/{NUM_EPISODES}")

if __name__ == "__main__":
    main()