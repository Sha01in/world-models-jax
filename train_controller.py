import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import gymnasium as gym
import cma  # <--- We are using standard cma now
import os
import time
import sys

# Import our models
from src.vae import VAE
from src.rnn import MDNRNN
from src.controller import get_action

# Settings
POPULATION_SIZE = 64    
NUM_GENERATIONS = 50    
NUM_ENVS = 8            
MAX_STEPS = 1000        

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
    # Load VAE
    vae = VAE(latent_dim=LATENT_DIM, key=jax.random.PRNGKey(0))
    vae = eqx.tree_deserialise_leaves(VAE_PATH, vae)
    
    # Load RNN
    rnn = MDNRNN(latent_dim=LATENT_DIM, action_dim=ACTION_DIM, 
                 hidden_size=HIDDEN_SIZE, key=jax.random.PRNGKey(0))
    rnn = eqx.tree_deserialise_leaves(RNN_PATH, rnn)
    return vae, rnn

def main():
    # 1. Setup Environment
    # We use "make_vec" to run 8 environments in parallel on CPU
    env = gym.make_vec("CarRacing-v3", num_envs=NUM_ENVS, vectorization_mode="async", render_mode="rgb_array")
    
    # 2. Load Models (VAE + RNN) onto GPU
    print("Loading models...")
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

    # 3. Initialize CMA-ES (CPU Based)
    num_params = (ACTION_DIM * INPUT_DIM) + ACTION_DIM
    print(f"Controller has {num_params} parameters.")
    print(f"Using pycma for optimization (Population: {POPULATION_SIZE})")
    
    # Initialize CMA-ES
    # sigma0 = 0.1 (initial standard deviation)
    es = cma.CMAEvolutionStrategy(num_params * [0], 0.1, {'popsize': POPULATION_SIZE, 'verbose': -1})

    print(f"Starting evolution for {NUM_GENERATIONS} generations...")
    
    for gen in range(NUM_GENERATIONS):
        start_time = time.time()
        
        # Ask for candidate parameters (List of lists)
        solutions = es.ask()
        
        # Convert to JAX array for the batch inference
        candidates = np.array(solutions)
        fitness = np.zeros(POPULATION_SIZE)
        
        # Evaluation Loop (Batched over environments)
        for i in range(0, POPULATION_SIZE, NUM_ENVS):
            batch_size = min(NUM_ENVS, POPULATION_SIZE - i)
            batch_candidates = candidates[i : i + batch_size]
            
            # Reset Envs
            obs, _ = env.reset()
            current_obs = obs[:batch_size] 
            
            # Resize
            import cv2
            obs_small = np.array([cv2.resize(o, (64, 64)) for o in current_obs])
            
            # Init RNN
            h = jnp.zeros((batch_size, HIDDEN_SIZE))
            c = jnp.zeros((batch_size, HIDDEN_SIZE))
            
            cumulative_reward = np.zeros(batch_size)
            dones = np.zeros(batch_size, dtype=bool)
            
            # Run Episode
            for t in range(MAX_STEPS):
                # A. Vision
                z = encode_frame(obs_small)
                
                # B. Action
                batch_params_jax = jnp.array(batch_candidates)
                actions = select_actions(batch_params_jax, z, h)
                actions_np = np.array(actions)
                
                # C. Step
                next_obs, rewards, terminated, truncated, _ = env.step(actions_np)
                
                done_mask = np.logical_or(terminated, truncated)[:batch_size]
                active_mask = (1 - dones)
                cumulative_reward += rewards[:batch_size] * active_mask
                dones = np.logical_or(dones, done_mask)
                
                if np.all(dones):
                    break
                
                # D. Memory
                if t < MAX_STEPS - 1:
                    h, c = rnn_step(z, actions, h, c)
                    current_obs = next_obs[:batch_size]
                    obs_small = np.array([cv2.resize(o, (64, 64)) for o in current_obs])

            fitness[i : i + batch_size] = cumulative_reward
            
        # CMA-ES minimizes, so we negated rewards in the JAX version.
        # Here we do the same: pass negative rewards to minimize "negative reward" (maximize reward)
        es.tell(solutions, -fitness)
        
        # Logging
        best_reward = np.max(fitness)
        mean_reward = np.mean(fitness)
        print(f"Gen {gen+1} | Best: {best_reward:.1f} | Mean: {mean_reward:.1f} | Time: {time.time()-start_time:.1f}s")
        
        # Save Best
        best_idx = np.argmax(fitness)
        best_params = candidates[best_idx]
        np.savez(BEST_CONTROLLER_PATH, params=best_params, score=best_reward)
        
    env.close()

if __name__ == "__main__":
    main()