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
from src.controller import get_action_linear, get_action_mlp
from src.config import get_config
from src.env_utils import make_env
from tqdm import tqdm

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
    
    # Check for controller type (default to linear for backward compatibility)
    if 'type' in data:
        controller_type = str(data['type'])
    else:
        controller_type = "linear"
        
    return vae, rnn, params, controller_type

    env.close()

def collect_data_parallel(env_name, num_episodes, num_workers, save_data, config, vae, rnn, controller_params, controller_type):
    from gymnasium.vector import AsyncVectorEnv
    import time
    
    # Define env factory
    def make_env_fn():
        return make_env(env_name, render_mode="rgb_array")
        
    # Create Vector Env
    print(f"Initializing {num_workers} environments...")
    envs = AsyncVectorEnv([make_env_fn for _ in range(num_workers)])
    
    # JIT compiled batched inference functions
    @jax.jit
    def encode_batch(imgs):
        # imgs: (B, H, W, C) -> (B, C, H, W) / 255.0
        x = jnp.array(imgs, dtype=jnp.float32) / 255.0
        x = jnp.transpose(x, (0, 3, 1, 2))
        # VAE expects (B, C, H, W)
        # We need to vmap the VAE call or VAE supports batch?
        # Our VAE __call__ takes (C, H, W). We need vmap.
        # Actually, let's check VAE definition. Usually we vmap it.
        # The single-item function:
        def encode_single(img):
            recon, mu, _ = vae(img, key=jax.random.PRNGKey(0))
            return mu
        return jax.vmap(encode_single)(x)

    @jax.jit
    def get_action_batch(zs, hs):
        if controller_type == "linear":
            return jax.vmap(get_action_linear, in_axes=(None, 0, 0, None))(controller_params, zs, hs, config.action_dim)
        elif controller_type == "mlp":
            return jax.vmap(lambda p, z, h, ad: get_action_mlp(p, z, h, ad, hidden_dim=64), in_axes=(None, 0, 0, None))(controller_params, zs, hs, config.action_dim)
        else:
            return jax.vmap(get_action_linear, in_axes=(None, 0, 0, None))(controller_params, zs, hs, config.action_dim)

    @jax.jit
    def rnn_step_batch(zs, actions, hs, cs):
        # zs: (B, Latent), actions: (B, Action), hs: (B, Hidden)
        rnn_in = jnp.concatenate([zs, actions], axis=1)
        (log_pi, mu, log_sigma, r_pred, d_pred), (h_new, c_new) = jax.vmap(rnn)(rnn_in, (hs, cs))
        return h_new, c_new

    # Initialize State
    obs_batch, _ = envs.reset()
    h_batch = jnp.zeros((num_workers, config.hidden_size))
    c_batch = jnp.zeros((num_workers, config.hidden_size))
    
    # Buffers for each worker
    worker_buffers = [{'obs': [], 'actions': [], 'rewards': [], 'dones': []} for _ in range(num_workers)]
    
    episodes_collected = 0
    pbar = tqdm(total=num_episodes, desc="Collecting Data")
    
    data_dir = os.path.join("data/rollouts", env_name, "on_policy")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
        
    while episodes_collected < num_episodes:
        # 1. Encode
        z_batch = encode_batch(obs_batch)
        
        # 2. Action
        action_batch_jax = get_action_batch(z_batch, h_batch)
        action_batch_np = np.array(action_batch_jax)
        
        # 3. Step Envs
        next_obs_batch, reward_batch, term_batch, trunc_batch, _ = envs.step(action_batch_np)
        
        # 4. RNN Step
        h_batch, c_batch = rnn_step_batch(z_batch, action_batch_jax, h_batch, c_batch)
        
        # 5. Store Data & Handle Dones
        for i in range(num_workers):
            # Store current step
            worker_buffers[i]['obs'].append(obs_batch[i])
            worker_buffers[i]['actions'].append(action_batch_np[i])
            worker_buffers[i]['rewards'].append(reward_batch[i])
            done = term_batch[i] or trunc_batch[i]
            worker_buffers[i]['dones'].append(done)
            
            if done:
                # Save Episode
                if episodes_collected < num_episodes:
                    seed = int(time.time() * 1000) + i + episodes_collected
                    save_path = os.path.join(data_dir, f"on_policy_ep_{seed}.npz")
                    np.savez_compressed(save_path,
                                        obs=np.array(worker_buffers[i]['obs']),
                                        actions=np.array(worker_buffers[i]['actions']),
                                        rewards=np.array(worker_buffers[i]['rewards']),
                                        dones=np.array(worker_buffers[i]['dones']))
                    episodes_collected += 1
                    pbar.update(1)
                
                # Reset Buffer
                worker_buffers[i] = {'obs': [], 'actions': [], 'rewards': [], 'dones': []}
                
                # Reset RNN State for this worker
                # We need to modify the JAX array in place or create new one
                # JAX arrays are immutable.
                h_batch = h_batch.at[i].set(jnp.zeros(config.hidden_size))
                c_batch = c_batch.at[i].set(jnp.zeros(config.hidden_size))
                
        obs_batch = next_obs_batch
        
    envs.close()
    pbar.close()
    print(f"Parallel collection complete. Saved {episodes_collected} episodes.")

def main():
    parser = argparse.ArgumentParser(description="Test Trained Agent")
    parser.add_argument("--episodes", type=int, default=NUM_EPISODES, help="Number of episodes to test")
    parser.add_argument("--no_video", action="store_true", help="Disable video saving (faster)")
    parser.add_argument("--save_data", action="store_true", help="Save rollout data for training (Curriculum Learning)")
    parser.add_argument("--env", type=str, default="CarRacing-v3", help="Environment name")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers for data collection")
    parser.add_argument("--seed", type=int, default=None, help="Fixed seed for reproducibility")
    args = parser.parse_args()

    num_episodes = args.episodes
    save_video = not args.no_video
    save_data = args.save_data
    env_name = args.env
    num_workers = args.workers
    
    config = get_config(env_name)

    vae, rnn, controller_params, controller_type = load_models(env_name, config)
    print(f"Loaded Controller Type: {controller_type}")
    
    if num_workers > 1:
        if not save_data:
            print("Warning: Parallel mode implies --save_data. Enabling it.")
            save_data = True
        if save_video:
            print("Warning: Parallel mode disables video saving.")
            
        collect_data_parallel(env_name, num_episodes, num_workers, save_data, config, vae, rnn, controller_params, controller_type)
        return

    @jax.jit
    def encode_and_recon(img):
        x = jnp.array(img, dtype=jnp.float32) / 255.0
        x = jnp.transpose(x, (2, 0, 1))
        recon, mu, _ = vae(x, key=jax.random.PRNGKey(0))
        return mu, recon

    @jax.jit
    def get_step_action(z, h):
        if controller_type == "linear":
            return get_action_linear(controller_params, z, h, config.action_dim)
        elif controller_type == "mlp":
            return get_action_mlp(controller_params, z, h, config.action_dim, hidden_dim=64)
        else:
            # Fallback
            return get_action_linear(controller_params, z, h, config.action_dim)

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
    
    # Data saving dir
    if save_data:
        data_dir = os.path.join("data/rollouts", env_name, "on_policy")
        if not os.path.exists(data_dir):
            os.makedirs(data_dir, exist_ok=True)
    
    if save_video:
        if not os.path.exists(video_dir):
            os.makedirs(video_dir, exist_ok=True)
        if not os.path.exists(diagnostics_dir):
            os.makedirs(diagnostics_dir, exist_ok=True)

    print(f"Testing Agent on {env_name}: {num_episodes} episodes...")

    for episode in range(num_episodes):
        # Generate a random seed for this episode
        if args.seed is not None:
            seed = args.seed + episode
        else:
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
        
        # Raw data for training
        obs_seq = []
        action_seq = []
        reward_seq = []
        done_seq = []
        
        for t in range(2100):
            # Obs is already resized by wrapper
            obs_small = obs
            
            if save_data:
                obs_seq.append(obs_small)
            
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
            
            # Store Training Data
            if save_data:
                action_seq.append(current_action)
                reward_seq.append(reward)
                done_seq.append(term or trunc)
            
            if term or trunc:
                break
        
        avg_surprise = total_surprise / t if t > 0 else 0.0
        print(f"Episode {episode+1}: Score = {total_reward:.1f} | Avg Surprise = {avg_surprise:.4f}")
        
        # Save Training Data
        if save_data:
            save_path = os.path.join(data_dir, f"on_policy_ep_{seed}.npz")
            np.savez_compressed(save_path,
                                obs=np.array(obs_seq),
                                actions=np.array(action_seq),
                                rewards=np.array(reward_seq),
                                dones=np.array(done_seq))

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