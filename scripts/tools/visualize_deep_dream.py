import gymnasium as gym
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import cv2
import os
import argparse
import sys

# Add project root to path
sys.path.append(os.getcwd())

from src.vae import VAE
from src.rnn import MDNRNN
from src.config import get_config
from src.env_utils import make_env

def load_models(env_name, config):
    checkpoint_dir = os.path.join("checkpoints", env_name)
    vae_path = os.path.join(checkpoint_dir, "vae.eqx")
    rnn_path = os.path.join(checkpoint_dir, "rnn.eqx")
    
    key = jax.random.PRNGKey(0)
    vae = VAE(latent_dim=config.latent_dim, key=key)
    vae = eqx.tree_deserialise_leaves(vae_path, vae)
    
    rnn = MDNRNN(latent_dim=config.latent_dim, action_dim=config.action_dim, 
                 hidden_size=config.hidden_size, key=key)
    rnn = eqx.tree_deserialise_leaves(rnn_path, rnn)
    
    return vae, rnn

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="VizdoomTakeCover-v0")
    parser.add_argument("--steps", type=int, default=100, help="Length of dream")
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()
    
    config = get_config(args.env)
    vae, rnn = load_models(args.env, config)
    
    # Setup Environment to get a starting state
    env = make_env(args.env, render_mode="rgb_array")
    obs, _ = env.reset(seed=42)
    
    # Burn-in: Run for 50 steps to get a fireball on screen
    print("Running burn-in to find a fireball...")
    frames = []
    h = jnp.zeros(config.hidden_size)
    c = jnp.zeros(config.hidden_size)
    
    # We need to track the last z to start the dream
    last_z = None
    
    for t in range(60): # 60 steps usually enough for a fireball to appear
        action = env.action_space.sample() # Random actions
        obs, _, _, _, _ = env.step(action)
        
        # Encode
        img = jnp.array(obs, dtype=jnp.float32) / 255.0
        img = jnp.transpose(img, (2, 0, 1))
        _, mu, _ = vae(img, key=jax.random.PRNGKey(0))
        last_z = mu
        
        # Update RNN state
        rnn_action = jnp.zeros(config.action_dim) 
        
        rnn_in = jnp.concatenate([last_z, rnn_action], axis=0)
        (_, _, _, _, _), (h, c) = rnn(rnn_in, (h, c))
        
        frames.append(obs)

    print("Starting Deep Dream...")
    
    # Now we dream!
    dream_frames = []
    prob_dones = []
    z = last_z
    key = jax.random.PRNGKey(100)
    
    for t in range(args.steps):
        # Decode current dream state
        recon = vae.decoder(z)
        recon_img = jnp.transpose(recon, (1, 2, 0))
        recon_img = jnp.array(recon_img * 255.0, dtype=jnp.uint8)
        dream_frames.append(np.array(recon_img))
        
        # Step RNN
        # Action = 0 (Stand still)
        action = jnp.zeros(config.action_dim)
        rnn_in = jnp.concatenate([z, action], axis=0)
        
        (log_pi, mu, log_sigma, _, d_pred), (h, c) = rnn(rnn_in, (h, c))
        
        prob_done = jax.nn.sigmoid(d_pred).item()
        prob_dones.append(prob_done)
        
        # Sample next Z
        key, subkey = jax.random.split(key)
        
        log_pi = log_pi.squeeze()
        k = jax.random.categorical(subkey, log_pi)
        
        mu_k = mu[k]
        log_sigma_k = log_sigma[k]
        sigma_k = jnp.exp(log_sigma_k) * args.temperature
        
        eps = jax.random.normal(key, shape=mu_k.shape)
        z = mu_k + sigma_k * eps
        
    # Save Video
    video_path = "artifacts/deep_dream_diagnostic.mp4"
    h_frame, w_frame, _ = dream_frames[0].shape
    
    # Upscale for visibility
    scale = 4
    h_scaled, w_scaled = h_frame * scale, w_frame * scale
    
    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 20, (w_scaled, h_scaled))
    
    for i, f in enumerate(dream_frames):
        f_scaled = cv2.resize(f, (w_scaled, h_scaled), interpolation=cv2.INTER_NEAREST)
        f_bgr = cv2.cvtColor(f_scaled, cv2.COLOR_RGB2BGR)
        
        # Overlay Death Probability
        prob = prob_dones[i]
        color = (0, 255, 0) if prob < 0.5 else (0, 0, 255) # Green if safe, Red if danger
        text = f"P(Death): {prob:.2f}"
        cv2.putText(f_bgr, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        out.write(f_bgr)
        
    out.release()
    print(f"Saved dream to {video_path}")
    env.close()

if __name__ == "__main__":
    main()
