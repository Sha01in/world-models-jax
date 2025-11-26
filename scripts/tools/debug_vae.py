import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import cv2
import os
import argparse
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.vae import VAE
from src.config import get_config

def load_vae(env_name, config):
    checkpoint_path = os.path.join("checkpoints", env_name, "vae.eqx")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
        
    key = jax.random.PRNGKey(0)
    vae = VAE(latent_dim=config.latent_dim, key=key)
    vae = eqx.tree_deserialise_leaves(checkpoint_path, vae)
    return vae

def main():
    parser = argparse.ArgumentParser(description="Debug VAE Reconstructions")
    parser.add_argument("--env", type=str, default="VizdoomTakeCover-v0", help="Environment name")
    parser.add_argument("--samples", type=int, default=5, help="Number of samples to visualize")
    args = parser.parse_args()
    
    env_name = args.env
    config = get_config(env_name)
    
    # Load VAE
    print(f"Loading VAE for {env_name}...")
    vae = load_vae(env_name, config)
    
    # Load Data
    data_dir = os.path.join("data/rollouts", env_name, "random")
    files = [f for f in os.listdir(data_dir) if f.endswith(".npz")]
    
    if not files:
        print("No data found.")
        return
        
    # Select random file
    filename = np.random.choice(files)
    filepath = os.path.join(data_dir, filename)
    print(f"Loading sample data from {filepath}...")
    
    with np.load(filepath) as data:
        obs = data['obs'] # (T, 64, 64, 3)
        
    # Select random frames
    indices = np.random.choice(len(obs), args.samples, replace=False)
    samples = obs[indices]
    
    # Reconstruct
    print("Reconstructing...")
    
    @jax.jit
    def reconstruct(img):
        x = jnp.array(img, dtype=jnp.float32) / 255.0
        x = jnp.transpose(x, (2, 0, 1)) # (3, 64, 64)
        recon, _, _ = vae(x, key=jax.random.PRNGKey(0))
        return recon
        
    vis_frames = []
    for i in range(args.samples):
        orig = samples[i]
        recon_jax = reconstruct(orig)
        
        recon = jnp.transpose(recon_jax, (1, 2, 0))
        recon = jnp.array(recon * 255.0, dtype=jnp.uint8)
        recon = np.array(recon)
        
        # Concatenate: Original | Reconstruction
        combined = np.hstack((orig, recon))
        vis_frames.append(combined)
        
    # Stack vertically
    final_img = np.vstack(vis_frames)
    
    # Save
    save_path = "vae_debug_tuned.png"
    cv2.imwrite(save_path, cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR))
    print(f"Saved visualization to {save_path}")

if __name__ == "__main__":
    main()
