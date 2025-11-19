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

# Settings
NUM_EPISODES = 5
RENDER = True            # Save a video of the first episode?
VIDEO_DIR = "videos"

# Configs (Must match training)
LATENT_DIM = 32
HIDDEN_SIZE = 256
ACTION_DIM = 3

# Paths
VAE_PATH = "checkpoints/vae.eqx"
RNN_PATH = "checkpoints/rnn.eqx"
CONTROLLER_PATH = "checkpoints/controller_dream.npz"

def load_models():
    key = jax.random.PRNGKey(0)
    # Load VAE
    vae = VAE(latent_dim=LATENT_DIM, key=key)
    vae = eqx.tree_deserialise_leaves(VAE_PATH, vae)
    
    # Load RNN (We only need it for the hidden state logic, not the weights, 
    # but we load weights to be safe)
    rnn = MDNRNN(latent_dim=LATENT_DIM, action_dim=ACTION_DIM, 
                 hidden_size=HIDDEN_SIZE, key=key)
    rnn = eqx.tree_deserialise_leaves(RNN_PATH, rnn)
    
    # Load Controller
    data = np.load(CONTROLLER_PATH)
    params = jnp.array(data['params'])
    
    return vae, rnn, params

def main():
    vae, rnn, controller_params = load_models()
    
    # Inference Functions
    @jax.jit
    def encode_and_recon(img):
        x = jnp.array(img, dtype=jnp.float32) / 255.0
        x = jnp.transpose(x, (2, 0, 1))
        # DO NOT expand_dims. Pass single image directly to VAE.
        recon, mu, _ = vae(x, key=jax.random.PRNGKey(0))
        return mu, recon

    @jax.jit
    def get_step_action(z, h):
        return get_action(controller_params, z, h)

    @jax.jit
    def rnn_next(z, a, h, c):
        rnn_in = jnp.concatenate([z, a], axis=0)
        _, (h_new, c_new) = rnn(rnn_in, (h, c))
        return h_new, c_new

    env = gym.make("CarRacing-v3", render_mode="rgb_array")
    if not os.path.exists(VIDEO_DIR):
        os.makedirs(VIDEO_DIR)

    print(f"Testing Agent (With Warmup Phase)...")

    # Just 1 episode for the test
    obs, _ = env.reset()
    
    # --- FIX: Initialize variables before the loop ---
    total_reward = 0
    h = jnp.zeros(HIDDEN_SIZE)
    c = jnp.zeros(HIDDEN_SIZE)
    frames_combined = []
    
    for t in range(1000):
        obs_small = cv2.resize(obs, (64, 64))
        
        # 1. Vision + Reconstruction
        z, recon_jax = encode_and_recon(obs_small)
        
        # Video recording logic
        recon_img = jnp.transpose(recon_jax, (1, 2, 0))
        recon_img = jnp.array(recon_img * 255.0, dtype=jnp.uint8)
        recon_np = np.array(recon_img)
        combined = np.hstack((obs_small, recon_np))
        frames_combined.append(combined)
        
        # --- WARMUP PHASE (The Zoom Fix) ---
        if t < 50:
            # Force straight driving during camera zoom
            action = np.array([0.0, 0.5, 0.0], dtype=np.float32)
            # Create JAX version for RNN update
            action_jax = jnp.array(action)
        else:
            # Let the Brain drive
            action_jax = get_step_action(z, h)
            action = np.array(action_jax)
        # ----------------------------------
            
        if t % 20 == 0:
            # Added Steer (index 0) to the print
            print(f"T={t} | Steer: {action[0]:.3f} | Gas: {action[1]:.3f} | Z-Norm: {jnp.linalg.norm(z):.2f}")
        # -
        
        # 3. Step Env
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        
        # 4. Update Memory
        h, c = rnn_next(z, action_jax, h, c)
        
        if terminated or truncated:
            break
    
    print(f"Final Score: {total_reward:.1f}")
    
    # Save Video
    height, width, _ = frames_combined[0].shape
    video_path = os.path.join(VIDEO_DIR, "agent_warmup_test.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    video = cv2.VideoWriter(video_path, fourcc, 30, (width, height))
    
    for f in frames_combined:
        video.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    
    video.release()
    print(f"Video saved to {video_path}")
    env.close()

if __name__ == "__main__":
    main()