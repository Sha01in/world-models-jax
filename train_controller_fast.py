import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import gymnasium as gym
import cma
import os
import time
import multiprocessing

# Import our models
from src.vae import VAE
from src.rnn import MDNRNN
from src.controller import get_action

# Settings
POPULATION_SIZE = 128   # Increased population for better stability
MAX_STEPS = 1000

# PARALLELISM TUNING
# We want as many envs as possible without choking the OS.
# On a modern CPU (WSL2), 32 is usually safe.
NUM_ENVS = 32           
NUM_GENERATIONS = 100    

# Model Configs
LATENT_DIM = 32
HIDDEN_SIZE = 256
ACTION_DIM = 3
INPUT_DIM = LATENT_DIM + HIDDEN_SIZE

# Checkpoints
VAE_PATH = "checkpoints/vae.eqx"
RNN_PATH = "checkpoints/rnn.eqx"
BEST_CONTROLLER_PATH = "checkpoints/controller_best.npz"

def load_models():
    vae = VAE(latent_dim=LATENT_DIM, key=jax.random.PRNGKey(0))
    vae = eqx.tree_deserialise_leaves(VAE_PATH, vae)
    
    rnn = MDNRNN(latent_dim=LATENT_DIM, action_dim=ACTION_DIM, 
                 hidden_size=HIDDEN_SIZE, key=jax.random.PRNGKey(0))
    rnn = eqx.tree_deserialise_leaves(RNN_PATH, rnn)
    return vae, rnn

def main():
    # 1. Setup Environment
    # "async" runs envs in separate processes.
    print(f"Launching {NUM_ENVS} parallel environments...")
    env = gym.make_vec(
        "CarRacing-v3", 
        num_envs=NUM_ENVS, 
        vectorization_mode="async", 
        render_mode="rgb_array"
    )
    
    # 2. Load Models
    print("Loading models to GPU...")
    vae, rnn = load_models()
    
    # Optimized Inference Functions
    @jax.jit
    def encode_frame(img_batch):
        # (B, 64, 64, 3) -> (B, 3, 64, 64) -> float
        x = jnp.transpose(img_batch, (0, 3, 1, 2)).astype(jnp.float32) / 255.0
        _, mu, _ = jax.vmap(vae)(x, jax.random.split(jax.random.PRNGKey(0), x.shape[0]))
        return mu

    @jax.jit
    def rnn_step(z, a, h, c):
        rnn_input = jnp.concatenate([z, a], axis=1)
        (_, _, _), (h_new, c_new) = jax.vmap(rnn)(rnn_input, (h, c))
        return h_new, c_new

    @jax.jit
    def select_actions(params_batch, z, h):
        return jax.vmap(get_action)(params_batch, z, h)

    # 3. Initialize CMA-ES
    num_params = (ACTION_DIM * INPUT_DIM) + ACTION_DIM
    print(f"Controller has {num_params} parameters.")
    print(f"Evolution: Pop {POPULATION_SIZE} | Parallel Envs {NUM_ENVS}")
    
    es = cma.CMAEvolutionStrategy(num_params * [0], 0.1, {
        'popsize': POPULATION_SIZE, 
        'verbose': -1
    })

    print(f"Starting evolution...")
    
    for gen in range(NUM_GENERATIONS):
        gen_start = time.time()
        
        solutions = es.ask()
        candidates = np.array(solutions)
        fitness = np.zeros(POPULATION_SIZE)
        
        # Loop over chunks of the population
        # If Pop=128 and Envs=32, this loop runs 4 times.
        for i in range(0, POPULATION_SIZE, NUM_ENVS):
            batch_size = min(NUM_ENVS, POPULATION_SIZE - i)
            batch_candidates = candidates[i : i + batch_size]
            
            # Reset only the envs we are using
            # Note: gymnasium vector envs reset all at once usually, 
            # but we just ignore the extras if batch_size < NUM_ENVS
            obs, _ = env.reset()
            current_obs = obs[:batch_size]
            
            import cv2
            # Vectorized resize? Hard with numpy/cv2, list comp is okay for 32 items
            obs_small = np.array([cv2.resize(o, (64, 64)) for o in current_obs])
            
            h = jnp.zeros((batch_size, HIDDEN_SIZE))
            c = jnp.zeros((batch_size, HIDDEN_SIZE))
            
            cumulative_reward = np.zeros(batch_size)
            dones = np.zeros(batch_size, dtype=bool)
            
            # --- THE EPISODE LOOP ---
            for t in range(MAX_STEPS):
                z = encode_frame(obs_small)
                
                batch_params_jax = jnp.array(batch_candidates)
                actions = select_actions(batch_params_jax, z, h)
                actions_np = np.array(actions)
                
                # Pad actions if batch_size < NUM_ENVS (Gym limitation)
                if batch_size < NUM_ENVS:
                    padded_actions = np.zeros((NUM_ENVS, 3))
                    padded_actions[:batch_size] = actions_np
                    full_obs, full_rewards, full_term, full_trunc, _ = env.step(padded_actions)
                    next_obs = full_obs[:batch_size]
                    rewards = full_rewards[:batch_size]
                    terminated = full_term[:batch_size]
                    truncated = full_trunc[:batch_size]
                else:
                    next_obs, rewards, terminated, truncated, _ = env.step(actions_np)
                
                done_mask = np.logical_or(terminated, truncated)
                active_mask = (1 - dones)
                cumulative_reward += rewards * active_mask
                dones = np.logical_or(dones, done_mask)
                
                # Early exit if EVERYONE in this batch is dead
                if np.all(dones):
                    break
                
                if t < MAX_STEPS - 1:
                    h, c = rnn_step(z, actions, h, c)
                    current_obs = next_obs
                    obs_small = np.array([cv2.resize(o, (64, 64)) for o in current_obs])
            # ------------------------

            fitness[i : i + batch_size] = cumulative_reward
            
        es.tell(solutions, -fitness)
        
        # Stats
        best_reward = np.max(fitness)
        mean_reward = np.mean(fitness)
        duration = time.time() - gen_start
        print(f"Gen {gen+1} | Best: {best_reward:.1f} | Mean: {mean_reward:.1f} | Time: {duration:.1f}s")
        
        # Save Best
        best_idx = np.argmax(fitness)
        best_params = candidates[best_idx]
        np.savez(BEST_CONTROLLER_PATH, params=best_params, score=best_reward)
        
    env.close()

if __name__ == "__main__":
    main()