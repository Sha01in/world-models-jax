import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import cma
import glob
import time
import os
import sys
import argparse
from src.rnn import MDNRNN
from src.controller import get_action
from src.config import get_config

# Settings (Defaults)
POPULATION_SIZE = 256       
BATCH_SIZE = 2048           
DREAM_LENGTH = 1000         
NUM_GENERATIONS = 100       
TEMPERATURE = 1.25          
NUM_GAUSSIANS = 5

def load_rnn(rnn_path, config):
    if not os.path.exists(rnn_path):
        print(f"\n[ERROR] Checkpoint not found: {rnn_path}")
        print("You must train the RNN before the Dreamer can run.")
        print("Run: python train_rnn.py --env <env_name>\n")
        sys.exit(1)
        
    key = jax.random.PRNGKey(0)
    model = MDNRNN(latent_dim=config.latent_dim, action_dim=config.action_dim, 
                   hidden_size=config.hidden_size, key=key)
    model = eqx.tree_deserialise_leaves(rnn_path, model)
    return model

def load_initial_zs(data_dir):
    files = glob.glob(os.path.join(data_dir, "*.npz"))
    if not files:
        print(f"\n[ERROR] No data series found in '{data_dir}'")
        print("You must process collected data before training.")
        print("Run: python process_data.py --env <env_name>\n")
        sys.exit(1)
        
    np.random.shuffle(files)
    all_z = []
    print("Loading seed data for dreams...")
    for f in files[:50]: 
        with np.load(f) as data:
            all_z.append(data['mu']) 
    return np.concatenate(all_z, axis=0)

def main():
    parser = argparse.ArgumentParser(description="Train Controller in Dream World")
    parser.add_argument("--generations", type=int, default=NUM_GENERATIONS, help="Number of generations to evolve")
    parser.add_argument("--pop_size", type=int, default=POPULATION_SIZE, help="Population size")
    parser.add_argument("--dream_length", type=int, default=DREAM_LENGTH, help="Steps per dream episode")
    parser.add_argument("--env", type=str, default="CarRacing-v3", help="Environment name")
    args = parser.parse_args()

    # Override globals (optional, or pass args)
    population_size = args.pop_size
    num_generations = args.generations
    dream_length = args.dream_length
    env_name = args.env
    
    config = get_config(env_name)
    
    # Paths
    checkpoint_dir = os.path.join("checkpoints", env_name)
    rnn_path = os.path.join(checkpoint_dir, "rnn.eqx")
    best_controller_path = os.path.join(checkpoint_dir, "controller_dream.npz")
    data_dir = os.path.join("data/series", env_name)

    # 1. Load Resources
    rnn = load_rnn(rnn_path, config)
    real_zs = load_initial_zs(data_dir)
    real_zs = jnp.array(real_zs)
    
    # 2. Define The Dream Engine (JIT Compiled)
    # IMPORTANT: JIT compilation depends on static shapes. 
    # If dream_length changes, this needs to re-compile.
    @jax.jit
    def run_dream_batch(params_batch, start_z, key):
        # Initialize LSTM State
        h = jnp.zeros((params_batch.shape[0], config.hidden_size))
        c = jnp.zeros((params_batch.shape[0], config.hidden_size))
        
        def step_fn(carry, _):
            z, h, c, active, cum_reward, current_key = carry
            
            # A. Controller Action
            action = jax.vmap(get_action, in_axes=(0, 0, 0, None))(params_batch, z, h, config.action_dim)
            
            # B. RNN Prediction
            rnn_input = jnp.concatenate([z, action], axis=1)
            (log_pi, mu, log_sigma, reward, done_logit), (h_next, c_next) = jax.vmap(rnn)(rnn_input, (h, c))
            
            # C. Sample Next Z
            k_key, z_key, next_key = jax.random.split(current_key, 3)
            
            # Squeeze to match expected shape (Batch, 5)
            # log_pi comes in as (Batch, 5, 1). We need (Batch, 5)
            log_pi_flat = log_pi.squeeze(-1) 
            
            # Sample mixture index k: (Batch,)
            k = jax.random.categorical(k_key, log_pi_flat)
            
            # Gather specific mu/sigma for k
            batch_indices = jnp.arange(params_batch.shape[0])
            mu_k = mu[batch_indices, k, :]          # (Batch, 32)
            log_sigma_k = log_sigma[batch_indices, k, :]
            sigma_k = jnp.exp(log_sigma_k) * TEMPERATURE 
            
            # Sample Z
            eps = jax.random.normal(z_key, shape=mu_k.shape)
            z_next = mu_k + sigma_k * eps
            
            # D. Update Reward/Done
            prob_done = jax.nn.sigmoid(done_logit).squeeze(-1) # Squeeze (B,1) -> (B,)
            is_alive = (prob_done < 0.5).astype(jnp.float32)
            
            # Mask reward if dead
            new_active = active * is_alive
            step_reward = reward.squeeze(-1) * new_active
            
            new_cum_reward = cum_reward + step_reward
            
            return (z_next, h_next, c_next, new_active, new_cum_reward, next_key), None

        init_active = jnp.ones(params_batch.shape[0])
        init_reward = jnp.zeros(params_batch.shape[0])
        
        final_carry, _ = jax.lax.scan(step_fn, 
                                      (start_z, h, c, init_active, init_reward, key), 
                                      None, 
                                      length=dream_length) # Use dynamic arg
        
        _, _, _, _, final_rewards, _ = final_carry
        return final_rewards

    # 3. Setup CMA-ES
    # Input dim for controller is latent + hidden
    input_dim = config.latent_dim + config.hidden_size
    num_params = (config.action_dim * input_dim) + config.action_dim
    
    print(f"Dream Training for {env_name}: {population_size} agents, {dream_length} steps, {num_generations} gens.")
    print(f"Controller Params: {num_params} (Input={input_dim}, Output={config.action_dim})")
    
    es = cma.CMAEvolutionStrategy(num_params * [0], 0.1, {'popsize': population_size, 'verbose': -1})
    
    print("Starting Dream...")
    
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    for gen in range(num_generations):
        start_time = time.time()
        
        solutions = es.ask()
        candidates = jnp.array(solutions)
        
        key = jax.random.PRNGKey(gen)
        rand_indices = jax.random.randint(key, (population_size,), 0, len(real_zs))
        start_zs = real_zs[rand_indices]
        
        rewards = run_dream_batch(candidates, start_zs, key)
        rewards_np = np.array(rewards)
        
        es.tell(solutions, -rewards_np)
        
        best = np.max(rewards_np)
        mean = np.mean(rewards_np)
        print(f"Gen {gen+1} | Best Reward: {best:.1f} | Mean: {mean:.1f} | Time: {time.time()-start_time:.3f}s")
        
        best_idx = np.argmax(rewards_np)
        np.savez(best_controller_path, params=candidates[best_idx], score=best)

if __name__ == "__main__":
    main()