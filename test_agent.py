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
    def encode(img):
        # Input: (64, 64, 3)
        # Output: (32,)
        
        # 1. Normalize and Transpose to (3, 64, 64)
        x = jnp.array(img, dtype=jnp.float32) / 255.0
        x = jnp.transpose(x, (2, 0, 1))
        
        # DO NOT expand_dims. Pass single image directly to VAE.
        _, mu, _ = vae(x, key=jax.random.PRNGKey(0))
        
        return mu # Returns (32,) directly
        
    @jax.jit
    def get_step_action(z, h):
        return get_action(controller_params, z, h)

    @jax.jit
    def rnn_next(z, a, h, c):
        # We need the RNN to update the hidden state 'h' and 'c'
        # The RNN expects (Batch, Input), so we expand dims
        rnn_in = jnp.concatenate([z, a], axis=0)
        _, (h_new, c_new) = rnn(rnn_in, (h, c))
        return h_new, c_new

    # Setup Environment
    env = gym.make("CarRacing-v3", render_mode="rgb_array")
    
    if RENDER and not os.path.exists(VIDEO_DIR):
        os.makedirs(VIDEO_DIR)

    print(f"Testing Agent for {NUM_EPISODES} episodes...")

    for episode in range(NUM_EPISODES):
        obs, _ = env.reset()
        
        h = jnp.zeros(HIDDEN_SIZE)
        c = jnp.zeros(HIDDEN_SIZE)
        
        total_reward = 0
        frames = []
        
        for t in range(1000):
            if RENDER and episode == 0:
                frames.append(obs)
            
            # 1. Vision (Resize -> VAE)
            obs_small = cv2.resize(obs, (64, 64))
            z = encode(obs_small)
            
            # 2. Controller
            action_jax = get_step_action(z, h)
            action = np.array(action_jax)
            
            # 3. Step Env
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            
            # 4. Update Memory
            h, c = rnn_next(z, action_jax, h, c)
            
            if terminated or truncated:
                break
        
        print(f"Episode {episode+1}: Score = {total_reward:.1f}")
        
        # Save Video of first episode
        if RENDER and episode == 0:
            height, width, layers = frames[0].shape
            video_path = os.path.join(VIDEO_DIR, "agent_test.mp4")
            # MP4 codec
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
            video = cv2.VideoWriter(video_path, fourcc, 30, (width, height))
            
            for frame in frames:
                # OpenCV uses BGR, Gym uses RGB
                video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            
            cv2.destroyAllWindows()
            video.release()
            print(f"Video saved to {video_path}")

    env.close()

if __name__ == "__main__":
    main()