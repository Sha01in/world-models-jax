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
VIDEO_DIR = "videos"
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

def main():
    vae, rnn, controller_params = load_models()
    
    @jax.jit
    def encode_and_recon(img):
        x = jnp.array(img, dtype=jnp.float32) / 255.0
        x = jnp.transpose(x, (2, 0, 1))
        recon, mu, _ = vae(x, key=jax.random.PRNGKey(0))
        return mu, recon

    @jax.jit
    def get_step_action(z, h):
        return get_action(controller_params, z, h)

    @jax.jit
    def rnn_next(z, a, h, c):
        rnn_in = jnp.concatenate([z, a], axis=0)
        (log_pi, mu, log_sigma, _, _), (h_new, c_new) = rnn(rnn_in, (h, c))
        
        # Calculate expected z (weighted average of Gaussians)
        pi = jnp.exp(log_pi)
        # mu shape: (5, 32), pi shape: (5, 1)
        expected_z = jnp.sum(pi * mu, axis=0) 
        
        return h_new, c_new, expected_z

    env = gym.make("CarRacing-v3", render_mode="rgb_array")
    if not os.path.exists(VIDEO_DIR):
        os.makedirs(VIDEO_DIR)

    print(f"Testing Agent (Full Control + No Action Repeat)...")

    for episode in range(NUM_EPISODES):
        obs, _ = env.reset()
        h = jnp.zeros(HIDDEN_SIZE)
        c = jnp.zeros(HIDDEN_SIZE)
        total_reward = 0
        frames_combined = []
        filmstrip_frames = []
        
        # Current held action
        current_action = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        
        # Previous prediction for surprise calculation
        prev_expected_z = None
        total_surprise = 0.0
        
        for t in range(1000):
            obs_small = cv2.resize(obs, (64, 64))
            z, recon_jax = encode_and_recon(obs_small)
            
            # Calculate Surprise (MSE between predicted z and actual z)
            if prev_expected_z is not None:
                surprise = jnp.mean((z - prev_expected_z) ** 2)
                total_surprise += surprise
            
            # Record Video (Episode 1 only)
            if episode == 0:
                recon_img = jnp.transpose(recon_jax, (1, 2, 0))
                recon_img = jnp.array(recon_img * 255.0, dtype=jnp.uint8)
                combined = np.hstack((obs_small, np.array(recon_img)))
                frames_combined.append(combined)
                
                # Add to filmstrip every 20 frames
                if t % 20 == 0:
                    filmstrip_frames.append(combined)

            # --- LOGIC START ---
            # 1. WARMUP (Frames 0-50): Drive Straight
            if t < 50:
                current_action = np.array([0.0, 0.5, 0.0], dtype=np.float32)
                rnn_action = jnp.array(current_action)
            
            # 2. DRIVING (Frames 50+): Use Brain
            else:
                # NO ACTION REPEAT: Change decisions every frame
                action_jax = get_step_action(z, h)
                current_action = np.array(action_jax)
                rnn_action = action_jax
            
            # 3. Telemetry (Check gas!)
            if t % 100 == 0:
                 h_norm = jnp.linalg.norm(h)
                 print(f"T={t} | Steer: {current_action[0]:.2f} | Gas: {current_action[1]:.2f} | H-Norm: {h_norm:.2f}")
            # -------------------

            obs, reward, term, trunc, _ = env.step(current_action)
            total_reward += reward
            
            h, c, expected_z = rnn_next(z, rnn_action, h, c)
            prev_expected_z = expected_z
            
            if term or trunc:
                break
        
        avg_surprise = total_surprise / t if t > 0 else 0.0
        print(f"Episode {episode+1}: Score = {total_reward:.1f} | Avg Surprise = {avg_surprise:.4f}")
        
        if episode == 0:
            height, width, _ = frames_combined[0].shape
            video_path = os.path.join(VIDEO_DIR, "final_agent.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
            video = cv2.VideoWriter(video_path, fourcc, 30, (width, height))
            for f in frames_combined:
                video.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
            video.release()
            print(f"Video saved to {video_path}")
            
            # Save Filmstrip
            if len(filmstrip_frames) > 0:
                # Arrange in a grid or long strip? Let's do a long strip for simplicity.
                filmstrip_img = np.hstack(filmstrip_frames)
                cv2.imwrite("debug_filmstrip.png", cv2.cvtColor(filmstrip_img, cv2.COLOR_RGB2BGR))
                print("Saved debug_filmstrip.png")

    env.close()

if __name__ == "__main__":
    main()