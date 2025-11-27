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
from src.controller import get_action_linear, get_action_mlp
from src.config import get_config
from src.jax_cma import CMA_ES

# Settings (Defaults)
POPULATION_SIZE = 256       
MINI_BATCH_SIZE = 32
BATCH_SIZE = 2048           
DREAM_LENGTH = 2100         
NUM_GENERATIONS = 100       
TEMP_START = 1.25
TEMP_END = 1.25       
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
    parser.add_argument("--temperature", type=float, default=1.15, help="Fixed temperature (default: 1.15)")
    parser.add_argument("--temp_start", type=float, default=None, help="Start temperature (overrides fixed temperature)")
    parser.add_argument("--temp_end", type=float, default=None, help="End temperature (overrides fixed temperature)")
    parser.add_argument("--env", type=str, default="CarRacing-v3", help="Environment name")
    parser.add_argument("--controller_type", type=str, default="linear", choices=["linear", "mlp"], help="Controller architecture")
    parser.add_argument("--hidden_size", type=int, default=64, help="Hidden size for MLP controller")
    parser.add_argument("--strategy", type=str, default="es", choices=["es", "cma", "jax_cma"], help="Optimization strategy: 'es', 'cma', or 'jax_cma'")
    parser.add_argument("--output", type=str, default="controller_dream.npz", help="Output filename for the controller")
    args = parser.parse_args()

    # Override globals (optional, or pass args)
    population_size = args.pop_size
    num_generations = args.generations
    dream_length = args.dream_length
    env_name = args.env
    
    # Temperature Logic
    if args.temp_start is not None or args.temp_end is not None:
        # Annealing mode (explicitly requested)
        temp_start = args.temp_start if args.temp_start is not None else args.temperature
        temp_end = args.temp_end if args.temp_end is not None else args.temperature
        print(f"Temperature Mode: Annealing ({temp_start} -> {temp_end})")
    else:
        # Fixed mode (default)
        temp_start = args.temperature
        temp_end = args.temperature
        print(f"Temperature Mode: Fixed ({args.temperature})")

    config = get_config(env_name)
    
    # Paths
    checkpoint_dir = os.path.join("checkpoints", env_name)
    rnn_path = os.path.join(checkpoint_dir, "rnn.eqx")
    best_controller_path = os.path.join(checkpoint_dir, args.output)
    data_dir = os.path.join("data/series", env_name)

    # 1. Load Resources
    rnn = load_rnn(rnn_path, config)
    real_zs = jnp.array(load_initial_zs(data_dir))
    
    # 2. Define The Dream Engine (JIT Compiled)
    # IMPORTANT: JIT compilation depends on static shapes. 
    # If dream_length changes, this needs to re-compile.
    @jax.jit
    def run_dream_batch(params_batch, start_z, key, temperature):
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
            sigma_k = jnp.exp(log_sigma_k) * temperature 
            
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

    # 3. Setup Controller
    # Input dim for controller is latent + hidden
    input_dim = config.latent_dim + config.hidden_size
    
    if args.controller_type == "linear":
        num_params = (config.action_dim * input_dim) + config.action_dim
        controller_fn = get_action_linear
        print(f"Controller Type: Linear")
    elif args.controller_type == "mlp":
        hidden_dim = args.hidden_size
        # Layer 1 (Input -> Hidden) + Layer 2 (Hidden -> Output)
        num_params = (input_dim * hidden_dim) + hidden_dim + (hidden_dim * config.action_dim) + config.action_dim
        controller_fn = lambda p, z, h, ad: get_action_mlp(p, z, h, ad, hidden_dim)
        print(f"Controller Type: MLP (Hidden={hidden_dim})")
    else:
        raise ValueError(f"Unknown controller type: {args.controller_type}")
    
    # Redefine run_dream_batch to use the selected controller_fn
    @jax.jit
    def run_dream_batch(params_batch, start_z, key, temperature):
        # Initialize LSTM State
        h = jnp.zeros((params_batch.shape[0], config.hidden_size))
        c = jnp.zeros((params_batch.shape[0], config.hidden_size))
        
        def step_fn(carry, _):
            z, h, c, active, cum_reward, current_key = carry
            
            # A. Controller Action
            action = jax.vmap(controller_fn, in_axes=(0, 0, 0, None))(params_batch, z, h, config.action_dim)
            
            # B. RNN Prediction
            rnn_input = jnp.concatenate([z, action], axis=1)
            (log_pi, mu, log_sigma, reward, done_logit), (h_next, c_next) = jax.vmap(rnn)(rnn_input, (h, c))
            
            # C. Sample Next Z
            k_key, z_key, next_key = jax.random.split(current_key, 3)
            
            # Squeeze to match expected shape (Batch, 5)
            log_pi_flat = log_pi.squeeze(-1) 
            
            # Sample mixture index k: (Batch,)
            k = jax.random.categorical(k_key, log_pi_flat)
            
            # Gather specific mu/sigma for k
            batch_indices = jnp.arange(params_batch.shape[0])
            mu_k = mu[batch_indices, k, :]          
            log_sigma_k = log_sigma[batch_indices, k, :]
            sigma_k = jnp.exp(log_sigma_k) * temperature 
            
            # Sample Z
            eps = jax.random.normal(z_key, shape=mu_k.shape)
            z_next = mu_k + sigma_k * eps
            
            # D. Update Reward/Done
            prob_done = jax.nn.sigmoid(done_logit).squeeze(-1) 
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
                                      length=dream_length) 
        
        _, _, _, _, final_rewards, _ = final_carry
        return final_rewards

    print(f"Dream Training for {env_name}: {population_size} agents, {dream_length} steps, {num_generations} gens.")
    print(f"Controller Params: {num_params} (Input={input_dim}, Output={config.action_dim})")
    print(f"Temperature Annealing: {temp_start} -> {temp_end}")
    print(f"Strategy: {args.strategy.upper()}")
    
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)

    # --- Strategy Selection ---
    
    # 1. JAX-Native ES (OpenAI-ES style)
    if args.strategy == "es":
        learning_rate = 0.01
        sigma = 0.1
        
        # Initialize Mean Parameters
        key = jax.random.PRNGKey(0)
        mu_params = jax.random.normal(key, (num_params,)) * 0.01 # Small init
        
        # JIT the update step for maximum speed
        # JIT the update step (Tell)
        @jax.jit
        def es_update(mu, eps, rewards):
            # 4. Rank transformation
            ranks = jnp.argsort(jnp.argsort(rewards)) # 0 to N-1
            centered_ranks = (ranks / (population_size - 1)) - 0.5
            
            # 5. Update mu
            update = jnp.dot(centered_ranks, eps)
            new_mu = mu + learning_rate * (update / (population_size * sigma))
            return new_mu

        print("Starting Dream (JAX-ES)...")
        
        for gen in range(num_generations):
            start_time = time.time()
            
            # Anneal Temperature
            progress = gen / num_generations
            current_temp = temp_start + (temp_end - temp_start) * progress
            
            key, subkey = jax.random.split(key)
            
            # Sample random start states
            rand_indices = jax.random.randint(subkey, (population_size,), 0, len(real_zs))
            start_zs = real_zs[rand_indices]
            
            # 1. Sample perturbations (Ask)
            # We do this in Python or JIT? JIT is fine for sampling.
            key, sample_key = jax.random.split(key)
            half_pop = population_size // 2
            eps = jax.random.normal(sample_key, (half_pop, num_params))
            eps = jnp.concatenate([eps, -eps], axis=0)
            candidates = mu_params + sigma * eps
            
            # 2. Evaluate (Python Loop to avoid OOM)
            rewards_list = []
            num_batches = population_size // MINI_BATCH_SIZE
            
            # Split keys for batches
            batch_keys = jax.random.split(key, num_batches)
            
            for i in range(num_batches):
                start_idx = i * MINI_BATCH_SIZE
                end_idx = (i + 1) * MINI_BATCH_SIZE
                
                batch_cands = candidates[start_idx:end_idx]
                batch_zs = start_zs[start_idx:end_idx]
                batch_key = batch_keys[i]
                
                batch_rewards = run_dream_batch(batch_cands, batch_zs, batch_key, current_temp)
                batch_rewards.block_until_ready()
                rewards_list.append(batch_rewards)
                
            rewards = jnp.concatenate(rewards_list)
            
            # 3. Update (Tell)
            mu_params = es_update(mu_params, eps, rewards)
            
            # Logging
            rewards_np = np.array(rewards)
            best = np.max(rewards_np)
            mean = np.mean(rewards_np)
            
            print(f"Gen {gen+1} | T={current_temp:.2f} | Best: {best:.1f} | Mean: {mean:.1f} | Time: {time.time()-start_time:.3f}s")
            
            # Save Best
            # Save Best
            best_idx = np.argmax(rewards_np)
            np.savez(best_controller_path, params=candidates[best_idx], score=best, type=args.controller_type, hidden_size=args.hidden_size)

    # 2. Original CMA-ES (using cma library)
    elif args.strategy == "cma":
        # Initialize CMA-ES
        # Sigma 0.1 is standard for this task
        es = cma.CMAEvolutionStrategy(num_params * [0], 0.1, {'popsize': population_size})
        
        print("Starting Dream (CMA-ES)...")
        
        key = jax.random.PRNGKey(0)
        
        for gen in range(num_generations):
            start_time = time.time()
            
            if es.stop():
                print("CMA-ES converged!")
                break
                
            # Anneal Temperature
            progress = gen / num_generations
            current_temp = temp_start + (temp_end - temp_start) * progress
            
            # 1. Ask for solutions
            solutions = es.ask()
            
            # 2. Evaluate (on GPU)
            # Convert list of solutions to JAX array
            candidates = jnp.array(solutions)
            
            key, subkey = jax.random.split(key)
            
            # Sample random start states
            rand_indices = jax.random.randint(subkey, (population_size,), 0, len(real_zs))
            start_zs = real_zs[rand_indices]
            
            # Run Dream Batch
            rewards = run_dream_batch(candidates, start_zs, subkey, current_temp)
            
            # 3. Tell (Update CMA)
            # CMA minimizes, so we pass negative rewards
            rewards_np = np.array(rewards) # Convert to numpy for cma lib
            es.tell(solutions, -rewards_np)
            
            # Logging
            best = np.max(rewards_np)
            mean = np.mean(rewards_np)
            
            print(f"Gen {gen+1} | T={current_temp:.2f} | Best: {best:.1f} | Mean: {mean:.1f} | Time: {time.time()-start_time:.3f}s")
            
            # Save Best
            # es.result is (xbest, fbest, evals, best, stds, ...)
            # We can also just take the best from this batch
            best_idx = np.argmax(rewards_np)
            np.savez(best_controller_path, params=candidates[best_idx], score=best, type=args.controller_type, hidden_size=args.hidden_size)
            
            # Optional: Print CMA internal info occasionally
            if (gen + 1) % 10 == 0:
                es.disp()

    # 3. Custom JAX-Native CMA-ES
    elif args.strategy == "jax_cma":
        
        print("Starting Dream (Custom JAX-CMA)...")
        
        # Initialize Strategy
        strategy = CMA_ES(num_params=num_params, pop_size=population_size, sigma_init=0.1)
        key = jax.random.PRNGKey(0)
        key, subkey = jax.random.split(key)
        state = strategy.init(subkey)
        
        # JIT the entire generation loop
        # JIT the strategy methods separately
        ask_fn = jax.jit(strategy.ask)
        tell_fn = jax.jit(strategy.tell)

        for gen in range(num_generations):
            start_time = time.time()
            
            # Anneal Temperature
            progress = gen / num_generations
            current_temp = temp_start + (temp_end - temp_start) * progress
            
            key, subkey = jax.random.split(key)
            
            # Sample random start states
            rand_indices = jax.random.randint(subkey, (population_size,), 0, len(real_zs))
            start_zs = real_zs[rand_indices]
            
            # 1. Ask
            candidates, new_state = ask_fn(state)
            
            # 2. Evaluate (Python Loop to avoid OOM)
            rewards_list = []
            num_batches = population_size // MINI_BATCH_SIZE
            
            # Split keys for batches to ensure diversity
            batch_keys = jax.random.split(new_state.key, num_batches)
            
            for i in range(num_batches):
                start_idx = i * MINI_BATCH_SIZE
                end_idx = (i + 1) * MINI_BATCH_SIZE
                
                batch_cands = candidates[start_idx:end_idx]
                batch_zs = start_zs[start_idx:end_idx]
                batch_key = batch_keys[i]
                
                # run_dream_batch is already JITted
                batch_rewards = run_dream_batch(batch_cands, batch_zs, batch_key, current_temp)
                # Block until ready to free memory?
                batch_rewards.block_until_ready()
                rewards_list.append(batch_rewards)
                
            rewards = jnp.concatenate(rewards_list)
            
            # 3. Tell
            # CMA minimizes, so pass negative rewards
            state = tell_fn(new_state, candidates, -rewards)
            
            # Logging
            rewards_np = np.array(rewards)
            best = np.max(rewards_np)
            mean = np.mean(rewards_np)
            
            print(f"Gen {gen+1} | T={current_temp:.2f} | Best: {best:.1f} | Mean: {mean:.1f} | Time: {time.time()-start_time:.3f}s")
            
            # Save Best
            best_idx = np.argmax(rewards_np)
            np.savez(best_controller_path, params=candidates[best_idx], score=best, type=args.controller_type)

if __name__ == "__main__":
    main()