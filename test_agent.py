import gymnasium as gym
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import cv2
import os
import argparse
from src.vae import VAE
from src.rnn import MDNRNN
from src.controller import get_action
from src.config import get_config
from src.env_utils import make_env

# Settings (Defaults)
NUM_EPISODES = 5
VIDEO_DIR = "videos"
DIAGNOSTICS_DIR = "diagnostics"
VIDEO_SCALE = 6

def load_models(env_name, config):
    checkpoint_dir = os.path.join("checkpoints", env_name)
    vae_path = os.path.join(checkpoint_dir, "vae.eqx")
    rnn_path = os.path.join(checkpoint_dir, "rnn.eqx")
    controller_path = os.path.join(checkpoint_dir, "controller_dream.npz")
    
    key = jax.random.PRNGKey(0)
    vae = VAE(latent_dim=config.latent_dim, key=key)
    vae = eqx.tree_deserialise_leaves(vae_path, vae)
    
    rnn = MDNRNN(latent_dim=config.latent_dim, action_dim=config.action_dim, 
                 hidden_size=config.hidden_size, key=key)
    rnn = eqx.tree_deserialise_leaves(rnn_path, rnn)
    
    data = np.load(controller_path)
    params = jnp.array(data['params'])
    return vae, rnn, params

def main():
    parser = argparse.ArgumentParser(description="Test Trained Agent")
    parser.add_argument("--episodes", type=int, default=NUM_EPISODES, help="Number of episodes to test")
    parser.add_argument("--no_video", action="store_true", help="Disable video saving (faster)")
    parser.add_argument("--env", type=str, default="CarRacing-v3", help="Environment name")
    args = parser.parse_args()

    num_episodes = args.episodes
    save_video = not args.no_video
    env_name = args.env
    
    config = get_config(env_name)

    vae, rnn, controller_params = load_models(env_name, config)
    
    @jax.jit
    def encode_and_recon(img):
        x = jnp.array(img, dtype=jnp.float32) / 255.0
        x = jnp.transpose(x, (2, 0, 1))
        recon, mu, _ = vae(x, key=jax.random.PRNGKey(0))
        return mu, recon

    @jax.jit
    def get_step_action(z, h):
        return get_action(controller_params, z, h, config.action_dim)

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

    # Use render_mode="rgb_array" to get pixels
    env = make_env(env_name, render_mode="rgb_array")
    
    # Video dir per env
    video_dir = os.path.join(VIDEO_DIR, env_name)
    diagnostics_dir = os.path.join(DIAGNOSTICS_DIR, env_name)
    
    if save_video:
        if not os.path.exists(video_dir):
            os.makedirs(video_dir, exist_ok=True)
        if not os.path.exists(diagnostics_dir):
            os.makedirs(diagnostics_dir, exist_ok=True)

    print(f"Testing Agent on {env_name}: {num_episodes} episodes...")

    for episode in range(num_episodes):
        # Generate a random seed for this episode
        seed = np.random.randint(0, 1000000)
        obs, _ = env.reset(seed=seed)
        h = jnp.zeros(config.hidden_size)
        c = jnp.zeros(config.hidden_size)
        total_reward = 0
        frames_combined = []
        filmstrip_frames = []
        
        # Current held action
        current_action = np.zeros(config.action_dim, dtype=np.float32)
        
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
            # Obs is already resized by wrapper
            obs_small = obs
            z, recon_jax = encode_and_recon(obs_small)
            
            # Calculate Surprise (MSE between predicted z and actual z)
            surprise = 0.0
            if prev_expected_z is not None:
                surprise = jnp.mean((z - prev_expected_z) ** 2)
                total_surprise += surprise
            
            # Record Video
            if save_video:
                # 1. Real Observation (already in obs_small)
                # 2. VAE Reconstruction
                recon_img = jnp.transpose(recon_jax, (1, 2, 0))
                recon_img = jnp.array(recon_img * 255.0, dtype=jnp.uint8)
                
                # 3. RNN Dream (Prediction for NEXT frame, shifted by 1 for vis?)
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
            # 1. WARMUP (Frames 0-50): Drive Straight / No-op
            if t < 50:
                if config.is_doom:
                    current_action = np.array([0.0], dtype=np.float32) # No-op?
                    # Wait, doom action is discrete mapped from continuous.
                    # 0.0 -> No-op? Or Left/Right?
                    # In our wrapper: < -0.3 Left, > 0.3 Right. 0.0 is No-op.
                    pass
                else:
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
                 # print(f"T={t} | Action: {current_action} | H-Norm: {h_norm:.2f}")
                 pass
            
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
            except Exception:
                telemetry_data["r_pred"].append(float(r_pred_val[0]))
            
            if term or trunc:
                break
        
        avg_surprise = total_surprise / t if t > 0 else 0.0
        print(f"Episode {episode+1}: Score = {total_reward:.1f} | Avg Surprise = {avg_surprise:.4f}")
        
        # Save Telemetry
        telemetry_dir = os.path.join("telemetry", env_name)
        if not os.path.exists(telemetry_dir):
            os.makedirs(telemetry_dir, exist_ok=True)
            
        np.savez(os.path.join(telemetry_dir, f"ep_{episode+1}.npz"), 
                 seed=seed,
                 actions=np.array(telemetry_data['actions']),
                 rewards=np.array(telemetry_data['rewards']),
                 z=np.array(telemetry_data['z']),
                 h_norm=np.array(telemetry_data['h_norm']),
                 surprise=np.array(telemetry_data['surprise']),
                 r_pred=np.array(telemetry_data['r_pred']))
        
        # Save Video per episode
        if save_video and frames_combined:
            h_orig, w_orig, _ = frames_combined[0].shape
            h_scaled, w_scaled = h_orig * VIDEO_SCALE, w_orig * VIDEO_SCALE
            
            video_path = os.path.join(video_dir, f"final_agent_ep{episode+1}.mp4")
            
            # Try H.264 (avc1) for better quality, fallback to mp4v
            fourcc_names = ['avc1', 'mp4v']
            video = None
            
            for name in fourcc_names:
                fourcc = cv2.VideoWriter_fourcc(*name)
                # CarRacing runs at 50fps
                temp_video = cv2.VideoWriter(video_path, fourcc, 50, (w_scaled, h_scaled))
                if temp_video.isOpened():
                    video = temp_video
                    print(f"Using codec: {name}")
                    break
            
            if video is None:
                 print("Error: Could not create video writer.")
            else:
                for f in frames_combined:
                    # Nearest neighbor scaling to preserve pixel sharpness
                    f_scaled = cv2.resize(f, (w_scaled, h_scaled), interpolation=cv2.INTER_NEAREST)
                    video.write(cv2.cvtColor(f_scaled, cv2.COLOR_RGB2BGR))
                video.release()
                print(f"Video saved to {video_path} ({w_scaled}x{h_scaled})")
            
            # Save Filmstrip per episode
            if len(filmstrip_frames) > 0:
                filmstrip_img = np.hstack(filmstrip_frames)
                fs_path = os.path.join(diagnostics_dir, f"debug_filmstrip_ep{episode+1}.png")
                cv2.imwrite(fs_path, cv2.cvtColor(filmstrip_img, cv2.COLOR_RGB2BGR))
                print(f"Saved {fs_path}")

    env.close()

if __name__ == "__main__":
    main()