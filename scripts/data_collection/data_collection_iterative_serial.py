import gymnasium as gym
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import cv2
import os
from src.vae import VAE
from src.rnn import MDNRNN
from src.controller import get_action
from tqdm import tqdm

# Settings
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

def collect_data():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    vae, rnn, controller_params = load_models()
    
    @jax.jit
    def encode(img):
        x = jnp.array(img, dtype=jnp.float32) / 255.0
        x = jnp.transpose(x, (2, 0, 1))
        features = vae.encoder(x)
        features = jnp.reshape(features, (-1,))
        mu = vae.mu_head(features)
        return mu

    @jax.jit
    def get_step_action(z, h):
        return get_action(controller_params, z, h)

    @jax.jit
    def rnn_next(z, a, h, c):
        rnn_in = jnp.concatenate([z, a], axis=0)
        _, (h_new, c_new) = rnn(rnn_in, (h, c))
        return h_new, c_new

    env = gym.make("CarRacing-v3", render_mode="rgb_array")
    
    print(f"Collecting {NUM_EPISODES} iterative episodes (Sim2Real failure data)...")
    
    for i in tqdm(range(NUM_EPISODES)):
        obs, _ = env.reset()
        h = jnp.zeros(HIDDEN_SIZE)
        c = jnp.zeros(HIDDEN_SIZE)
        
        obs_seq, action_seq, reward_seq, done_seq = [], [], [], []
        
        # Current held action
        current_action = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        rnn_action = jnp.array(current_action)
        
        for t in range(MAX_STEPS):
            obs_small = cv2.resize(obs, (IMG_SIZE, IMG_SIZE))
            obs_seq.append(obs_small)
            
            z = encode(obs_small)
            
            # WARMUP (Frames 0-50): Drive Straight
            if t < 50:
                current_action = np.array([0.0, 0.5, 0.0], dtype=np.float32)
                rnn_action = jnp.array(current_action)
            else:
                # NO ACTION REPEAT (matching latest test_agent)
                action_jax = get_step_action(z, h)
                current_action = np.array(action_jax)
                rnn_action = action_jax
            
            # Add slight noise to encourage exploration around the failure
            current_action += np.random.normal(0, 0.05, size=3)
            current_action = np.clip(current_action, [-1, 0, 0], [1, 1, 1])

            action_seq.append(current_action)
            
            obs, reward, term, trunc, _ = env.step(current_action)
            reward_seq.append(reward)
            done = term or trunc
            done_seq.append(done)
            
            h, c = rnn_next(z, rnn_action, h, c)
            
            if done:
                break
        
        # Save
        np.savez_compressed(
            os.path.join(DATA_DIR, f"iterative_rollout_{i}.npz"),
            obs=np.array(obs_seq),
            actions=np.array(action_seq),
            rewards=np.array(reward_seq),
            dones=np.array(done_seq)
        )
            
    env.close()
    print("Iterative collection complete.")

if __name__ == "__main__":
    collect_data()
