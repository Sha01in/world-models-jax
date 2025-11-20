import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import cma
import glob
import os
import time
from src.rnn import MDNRNN
from src.controller import get_action

# Settings
POPULATION_SIZE = 256       
BATCH_SIZE = 2048           
DREAM_LENGTH = 1000         
NUM_GENERATIONS = 100       
TEMPERATURE = 1.25          

# Model Paths
RNN_PATH = "checkpoints/rnn.eqx"
BEST_CONTROLLER_PATH = "checkpoints/controller_dream.npz"

# Model Configs
LATENT_DIM = 32
HIDDEN_SIZE = 256
ACTION_DIM = 3
INPUT_DIM = LATENT_DIM + HIDDEN_SIZE
NUM_GAUSSIANS = 5

def load_rnn():
    key = jax.random.PRNGKey(0)
    model = MDNRNN(latent_dim=LATENT_DIM, action_dim=ACTION_DIM, 
                   hidden_size=HIDDEN_SIZE, key=key)
    model = eqx.tree_deserialise_leaves(RNN_PATH, model)
    return model

def load_initial_zs():
    files = glob.glob("data/series/*.npz")
    np.random.shuffle(files)
    all_z = []
    print("Loading seed data for dreams...")
    for f in files[:50]: 
        with np.load(f) as data:
            all_z.append(data['mu']) 
    return np.concatenate(all_z, axis=0)

def main():
    # 1. Load Resources
    rnn = load_rnn()
    real_zs = load_initial_zs()
    real_zs = jnp.array(real_zs)
    
    # 2. Define The Dream Engine (JIT Compiled)
    @jax.jit
    def run_dream_batch(params_batch, start_z, key):
        # Initialize LSTM State
        h = jnp.zeros((params_batch.shape[0], HIDDEN_SIZE))
        c = jnp.zeros((params_batch.shape[0], HIDDEN_SIZE))
        
        def step_fn(carry, _):
            z, h, c, active, cum_reward, current_key = carry
            
            # A. Controller Action
            action = jax.vmap(get_action)(params_batch, z, h)
            
            # B. RNN Prediction
            rnn_input = jnp.concatenate([z, action], axis=1)
            (log_pi, mu, log_sigma, reward, done_logit), (h_next, c_next) = jax.vmap(rnn)(rnn_input, (h, c))
            
            # C. Sample Next Z
            k_key, z_key, next_key = jax.random.split(current_key, 3)
            
            # --- FIX: Squeeze the last dimension ---
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
                                      length=DREAM_LENGTH)
        
        _, _, _, _, final_rewards, _ = final_carry
        return final_rewards

    # 3. Setup CMA-ES
    num_params = (ACTION_DIM * INPUT_DIM) + ACTION_DIM
    print(f"Dream Training: {POPULATION_SIZE} agents, {DREAM_LENGTH} steps.")
    
    es = cma.CMAEvolutionStrategy(num_params * [0], 0.1, {'popsize': POPULATION_SIZE, 'verbose': -1})
    
    print("Starting Dream...")
    
    for gen in range(NUM_GENERATIONS):
        start_time = time.time()
        
        solutions = es.ask()
        candidates = jnp.array(solutions)
        
        key = jax.random.PRNGKey(gen)
        rand_indices = jax.random.randint(key, (POPULATION_SIZE,), 0, len(real_zs))
        start_zs = real_zs[rand_indices]
        
        rewards = run_dream_batch(candidates, start_zs, key)
        rewards_np = np.array(rewards)
        
        es.tell(solutions, -rewards_np)
        
        best = np.max(rewards_np)
        mean = np.mean(rewards_np)
        print(f"Gen {gen+1} | Best Reward: {best:.1f} | Mean: {mean:.1f} | Time: {time.time()-start_time:.3f}s")
        
        best_idx = np.argmax(rewards_np)
        np.savez(BEST_CONTROLLER_PATH, params=candidates[best_idx], score=best)

if __name__ == "__main__":
    main()