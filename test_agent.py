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
    def decode_from_z(z):
        # Decode a latent vector back to an image
        recon = vae.decoder(z)
        return recon

    @jax.jit
    def rnn_next(z, a, h, c):
        rnn_in = jnp.concatenate([z, a], axis=0)
        (log_pi, mu, log_sigma, r_pred, d_pred), (h_new, c_new) = rnn(rnn_in, (h, c))
        
        # Calculate expected z (weighted average of Gaussians)
        pi = jnp.exp(log_pi)
        # mu shape: (5, 32), pi shape: (5, 1)
        expected_z = jnp.sum(pi * mu, axis=0) 
        
        return h_new, c_new, expected_z, r_pred

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
        
            # Data collection for analysis
        telemetry_data = {
            'actions': [],
            'rewards': [],
            'z': [],
            'h_norm': [],
            'surprise': [],
            'r_pred': []
        }
        
        for t in range(1000):
            obs_small = cv2.resize(obs, (64, 64))
            z, recon_jax = encode_and_recon(obs_small)
            
            # Calculate Surprise (MSE between predicted z and actual z)
            surprise = 0.0
            if prev_expected_z is not None:
                surprise = jnp.mean((z - prev_expected_z) ** 2)
                total_surprise += surprise
            
            # Record Video (All Episodes)
            # 1. Real Observation (already in obs_small)
            
            # 2. VAE Reconstruction
            recon_img = jnp.transpose(recon_jax, (1, 2, 0))
            recon_img = jnp.array(recon_img * 255.0, dtype=jnp.uint8)
            
            # 3. RNN Dream (Prediction for NEXT frame, shifted by 1 for vis?)
            # Actually, prev_expected_z is the prediction for the CURRENT frame made 1 step ago.
            if prev_expected_z is not None:
                dream_jax = decode_from_z(prev_expected_z)
                dream_img = jnp.transpose(dream_jax, (1, 2, 0))
                dream_img = jnp.array(dream_img * 255.0, dtype=jnp.uint8)
            else:
                dream_img = jnp.zeros_like(recon_img)

            # Combine: [Real | Recon | Dream]
            combined = np.hstack((obs_small, np.array(recon_img), np.array(dream_img)))
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
            
            # 3. Telemetry
            h_norm = jnp.linalg.norm(h)
            if t % 100 == 0:
                 print(f"T={t} | Steer: {current_action[0]:.2f} | Gas: {current_action[1]:.2f} | H-Norm: {h_norm:.2f}")
            
            # Store Telemetry (Need r_pred which is computed *after* this step for *next* step?)
            # Actually rnn_next gives prediction for t+1.
            # So prev_expected_z corresponds to prediction made at t-1 for t.
            # But we don't have r_pred stored from previous step easily unless we change loop vars.
            # Let's append a dummy for now or restructure slightly? 
            # Actually, we call rnn_next at end of loop.
            
            obs, reward, term, trunc, _ = env.step(current_action)
            total_reward += reward
            
            h, c, expected_z, r_pred_val = rnn_next(z, rnn_action, h, c)
            prev_expected_z = expected_z
            
            # Store Telemetry
            telemetry_data['actions'].append(current_action)
            telemetry_data['z'].append(z)
            telemetry_data['h_norm'].append(h_norm)
            telemetry_data['surprise'].append(surprise)
            telemetry_data['rewards'].append(reward)
            try:
                telemetry_data['r_pred'].append(float(r_pred_val.item()))
            except:
                telemetry_data['r_pred'].append(float(r_pred_val[0]))
            
            if term or trunc:
                break
        
        avg_surprise = total_surprise / t if t > 0 else 0.0
        print(f"Episode {episode+1}: Score = {total_reward:.1f} | Avg Surprise = {avg_surprise:.4f}")
        
        # Save Telemetry
        if not os.path.exists("telemetry"):
            os.makedirs("telemetry")
        np.savez(f"telemetry/ep_{episode+1}.npz", 
                 actions=np.array(telemetry_data['actions']),
                 rewards=np.array(telemetry_data['rewards']),
                 z=np.array(telemetry_data['z']),
                 h_norm=np.array(telemetry_data['h_norm']),
                 surprise=np.array(telemetry_data['surprise']),
                 r_pred=np.array(telemetry_data['r_pred']))
        print(f"Saved telemetry/ep_{episode+1}.npz")
        
        # Save Video per episode
        if frames_combined:
            height, width, _ = frames_combined[0].shape
            video_path = os.path.join(VIDEO_DIR, f"final_agent_ep{episode+1}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
            video = cv2.VideoWriter(video_path, fourcc, 30, (width, height))
            for f in frames_combined:
                video.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
            video.release()
            print(f"Video saved to {video_path}")
            
            # Save Filmstrip per episode
            if len(filmstrip_frames) > 0:
                filmstrip_img = np.hstack(filmstrip_frames)
                fs_path = f"debug_filmstrip_ep{episode+1}.png"
                cv2.imwrite(fs_path, cv2.cvtColor(filmstrip_img, cv2.COLOR_RGB2BGR))
                print(f"Saved {fs_path}")

    env.close()

if __name__ == "__main__":
    main()