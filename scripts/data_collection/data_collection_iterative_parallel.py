import gymnasium as gym
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import os
from src.vae import VAE
from src.rnn import MDNRNN
from src.controller import get_action
from tqdm import tqdm

# Settings
NUM_ENVS = 8            # Reduced envs because JAX inference is heavy
NUM_EPISODES = 500
MAX_STEPS = 600
DATA_DIR = "data/rollouts_iterative"
IMG_SIZE = 64
LATENT_DIM = 32
HIDDEN_SIZE = 256
ACTION_DIM = 3

# Paths
VAE_PATH = "checkpoints/vae.eqx"
RNN_PATH = "checkpoints/rnn.eqx"
CONTROLLER_PATH = "checkpoints/controller_dream.npz"

def load_models():
    key = jax.random.PRNGKey(0)
    vae = VAE(latent_dim=LATENT_DIM, key=key)
    vae = eqx.tree_deserialise_leaves(VAE_PATH, vae)
    
    rnn = MDNRNN(latent_dim=LATENT_DIM, action_dim=ACTION_DIM, 
                 hidden_size=HIDDEN_SIZE, key=key)
    rnn = eqx.tree_deserialise_leaves(RNN_PATH, rnn)
    
    data = np.load(CONTROLLER_PATH)
    params = jnp.array(data['params'])
    return vae, rnn, params

def make_env():
    env = gym.make("CarRacing-v3", render_mode="rgb_array")
    env = gym.wrappers.ResizeObservation(env, (IMG_SIZE, IMG_SIZE))
    return env

def collect_data():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    # Load JAX models (on GPU)
    vae, rnn, controller_params = load_models()
    
    # Prepare batched inference functions
    # We need to run JAX models on a batch of observations from multiple envs
    
    @jax.jit
    def encode_batch(imgs):
        # imgs: (Batch, 64, 64, 3) -> (Batch, 3, 64, 64) / 255.0
        x = jnp.array(imgs, dtype=jnp.float32) / 255.0
        x = jnp.transpose(x, (0, 3, 1, 2))
        
        # VAE Encoder expects (C, H, W) usually, but let's check VAE definition
        # Standard eqx.nn.Conv2d expects (C, H, W).
        # We need vmap if the VAE is defined for single items.
        
        # Let's assume VAE is single-item. vmap it.
        def encode_single(img):
             features = vae.encoder(img)
             features = jnp.reshape(features, (-1,))
             mu = vae.mu_head(features)
             return mu
             
        return jax.vmap(encode_single)(x)

    @jax.jit
    def get_action_batch(z, h):
        return jax.vmap(get_action, in_axes=(None, 0, 0))(controller_params, z, h)

    @jax.jit
    def rnn_next_batch(z, a, h, c):
        # rnn also needs vmap
        def rnn_single(z_i, a_i, h_i, c_i):
             rnn_in = jnp.concatenate([z_i, a_i], axis=0)
             _, (h_new, c_new) = rnn(rnn_in, (h_i, c_i))
             return h_new, c_new
             
        return jax.vmap(rnn_single)(z, a, h, c)

    print(f"Launching {NUM_ENVS} parallel environments...")
    envs = gym.vector.AsyncVectorEnv([make_env for _ in range(NUM_ENVS)])
    
    print(f"Collecting {NUM_EPISODES} iterative episodes...")
    
    # Buffers
    obs_buffers = [[] for _ in range(NUM_ENVS)]
    action_buffers = [[] for _ in range(NUM_ENVS)]
    reward_buffers = [[] for _ in range(NUM_ENVS)]
    done_buffers = [[] for _ in range(NUM_ENVS)]
    
    # RNN States for all envs
    h_states = jnp.zeros((NUM_ENVS, HIDDEN_SIZE))
    c_states = jnp.zeros((NUM_ENVS, HIDDEN_SIZE))
    
    # Track which step we are in for each env (for warmup)
    step_counts = np.zeros(NUM_ENVS, dtype=int)
    
    # Last action taken (for RNN input)
    # last_actions = jnp.zeros((NUM_ENVS, ACTION_DIM))

    saved_count = 0
    obs, _ = envs.reset()
    
    pbar = tqdm(total=NUM_EPISODES)
    
    while saved_count < NUM_EPISODES:
        # 1. Inference
        # obs is (NUM_ENVS, 64, 64, 3)
        z_batch = encode_batch(obs)
        
        # 2. Logic
        # If step < 50: Warmup (Straight)
        # Else: Controller
        
        # Compute Controller actions for ALL
        policy_actions = get_action_batch(z_batch, h_states)
        policy_actions = np.array(policy_actions)
        
        # Override for warmup
        final_actions = []
        for i in range(NUM_ENVS):
            if step_counts[i] < 50:
                act = np.array([0.0, 0.5, 0.0])
            else:
                act = policy_actions[i]
                # Add Noise
                act += np.random.normal(0, 0.05, size=3)
                act = np.clip(act, [-1, 0, 0], [1, 1, 1])
            final_actions.append(act)
            
        final_actions = np.array(final_actions)
        
        # 3. Step Envs
        next_obs, rewards, terminated, truncated, _ = envs.step(final_actions)
        dones = np.logical_or(terminated, truncated)
        
        # 4. Update RNN State (using the actions we just took)
        # We need to convert final_actions back to JAX array
        jax_actions = jnp.array(final_actions)
        h_states, c_states = rnn_next_batch(z_batch, jax_actions, h_states, c_states)
        
        # 5. Store & Reset Logic
        for i in range(NUM_ENVS):
            obs_buffers[i].append(obs[i])
            action_buffers[i].append(final_actions[i])
            reward_buffers[i].append(rewards[i])
            done_buffers[i].append(dones[i])
            
            step_counts[i] += 1
            
            if dones[i] or len(obs_buffers[i]) >= MAX_STEPS:
                if saved_count < NUM_EPISODES:
                    np.savez_compressed(
                        os.path.join(DATA_DIR, f"iterative_rollout_{saved_count}.npz"),
                        obs=np.array(obs_buffers[i]),
                        actions=np.array(action_buffers[i]),
                        rewards=np.array(reward_buffers[i]),
                        dones=np.array(done_buffers[i])
                    )
                    saved_count += 1
                    pbar.update(1)
                
                # Reset this env's buffer and state
                obs_buffers[i] = []
                action_buffers[i] = []
                reward_buffers[i] = []
                done_buffers[i] = []
                step_counts[i] = 0
                
                # Reset RNN state for this specific batch index
                # Note: JAX arrays are immutable, so we must update the batch
                h_states = h_states.at[i].set(jnp.zeros(HIDDEN_SIZE))
                c_states = c_states.at[i].set(jnp.zeros(HIDDEN_SIZE))
        
        obs = next_obs

    envs.close()
    print("Parallel iterative collection complete.")

if __name__ == "__main__":
    collect_data()
