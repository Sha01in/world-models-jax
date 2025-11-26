import gymnasium as gym
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import cv2
import os
import sys
import os
import time
import argparse

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.vae import VAE
from src.rnn import MDNRNN
from src.controller import get_action
from src.config import get_config
from src.env_utils import make_env
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

# Settings
NUM_WORKERS = 12        # Reduced from 16 to 12 to prevent CPU starvation/Zombie issues
NUM_EPISODES = 2000     # Default, can be overridden
MAX_STEPS = 1000        # Increased for Doom

def collect_episode_worker(args_tuple):
    """
    Worker function to collect a single episode.
    Running in a separate process with its own JAX CPU instance.
    """
    seed, env_name, config, vae_path, rnn_path, controller_path, data_dir = args_tuple

    # 1. Force JAX to use CPU to avoid GPU contention
    jax.config.update("jax_platform_name", "cpu")
    
    # 2. Load Models (Fresh copy for this process)
    key = jax.random.PRNGKey(0)
    
    vae = VAE(latent_dim=config.latent_dim, key=key)
    vae = eqx.tree_deserialise_leaves(vae_path, vae)
    
    rnn = MDNRNN(latent_dim=config.latent_dim, action_dim=config.action_dim, 
                 hidden_size=config.hidden_size, key=key)
    rnn = eqx.tree_deserialise_leaves(rnn_path, rnn)
    
    controller_data = np.load(controller_path)
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
        return get_action(controller_params, z, h, config.action_dim)

    # 4. Simulation Loop
    try:
        # Unique seed
        np.random.seed(seed)
        
        # Create Env
        # Use make_env from env_utils which handles DoomWrapper and resizing
        env = make_env(env_name, render_mode="rgb_array")
        obs, _ = env.reset(seed=seed)
        
        h = jnp.zeros(config.hidden_size)
        c = jnp.zeros(config.hidden_size)
        
        obs_seq, action_seq, reward_seq, done_seq = [], [], [], []
        
        # Warmup Action
        # For Doom, 0.0 is No-op (Wait)
        # For CarRacing, [0, 0, 0] is No-op
        if config.is_doom:
             warmup_action = np.zeros(config.action_dim, dtype=np.float32)
        else:
             warmup_action = np.array([0.0, 0.5, 0.0], dtype=np.float32)

        for t in range(MAX_STEPS):
            # Obs is already resized by wrapper in env_utils
            obs_small = obs
            obs_seq.append(obs_small)
            
            # Inference
            z = encode(obs_small)
            
            if t < 50:
                # Warmup
                action = warmup_action
                # For RNN input, we need JAX array
                action_jax = jnp.array(action)
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
            h, c, _ = rnn_step(z, action_jax, h, c)
            
            if term or trunc:
                break
        
        env.close()
        
        # Save Data
        # Use seed as unique ID
        save_path = os.path.join(data_dir, f"on_policy_ep_{seed}.npz")
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
    parser = argparse.ArgumentParser(description="Parallel Data Collection (On-Policy)")
    parser.add_argument("--env", type=str, default="CarRacing-v3", help="Environment name")
    parser.add_argument("--episodes", type=int, default=NUM_EPISODES, help="Number of episodes")
    parser.add_argument("--workers", type=int, default=NUM_WORKERS, help="Number of workers")
    args = parser.parse_args()
    
    env_name = args.env
    num_episodes = args.episodes
    num_workers = args.workers
    
    config = get_config(env_name)
    
    # Paths
    checkpoint_dir = os.path.join("checkpoints", env_name)
    vae_path = os.path.join(checkpoint_dir, "vae.eqx")
    rnn_path = os.path.join(checkpoint_dir, "rnn.eqx")
    controller_path = os.path.join(checkpoint_dir, "controller_dream.npz")
    
    data_dir = os.path.join("data/rollouts", env_name, "on_policy")
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
    
    print(f"Starting Distributed Data Collection for {env_name}.")
    print(f"Workers: {num_workers}")
    print(f"Target: {num_episodes} episodes")
    print("JAX Mode: CPU (Forced per worker)")
    
    # Prepare args for workers
    # We pass paths instead of objects to avoid pickling issues with JAX/Equinox objects
    seeds = list(range(int(time.time()), int(time.time()) + num_episodes))
    worker_args = [(seed, env_name, config, vae_path, rnn_path, controller_path, data_dir) for seed in seeds]
    
    # Use mp_context='spawn' to ensure clean process start without inheriting JAX state
    import multiprocessing as mp
    ctx = mp.get_context('spawn')
    
    with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as executor:
        results = list(tqdm(executor.map(collect_episode_worker, worker_args), total=num_episodes))
        
    success_count = sum(results)
    print(f"Collection Complete. Success: {success_count}/{num_episodes}")

if __name__ == "__main__":
    main()