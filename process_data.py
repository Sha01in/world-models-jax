import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import glob
import os
from tqdm import tqdm
from src.vae import VAE

# Settings
BATCH_SIZE = 128
LATENT_DIM = 32
OUTPUT_DIR = "data/series"
VAE_PATH = "checkpoints/vae.eqx"

# 1. Load ALL datasets (Good + Bad + Random + Iterative)
DATA_PATTERN_GOOD = "data/rollouts/*.npz"
DATA_PATTERN_BAD = "data/rollouts_bad/*.npz"
DATA_PATTERN_RANDOM = "data/rollouts_random/*.npz"
DATA_PATTERN_ITERATIVE = "data/rollouts_iterative/*.npz"
DATA_PATTERN_RECOVERY = "data/rollouts_recovery/*.npz"
DATA_PATTERN_AGGRESSIVE = "data/rollouts_aggressive/*.npz"
DATA_PATTERN_ON_POLICY = "data/rollouts_on_policy/*.npz"

def process_data():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print(f"Loading VAE from {VAE_PATH}...")
    # Initialize VAE
    model = VAE(latent_dim=LATENT_DIM, key=jax.random.PRNGKey(0))
    try:
        model = eqx.tree_deserialise_leaves(VAE_PATH, model)
    except Exception as e:
        print(f"Error loading VAE: {e}")
        return

    # Optimized Encoder function
    @eqx.filter_jit
    def encode_batch(m, x):
        def encode_single(img):
            features = m.encoder(img)
            features = jnp.reshape(features, (-1,))
            mu = m.mu_head(features)
            logvar = m.logvar_head(features)
            return mu, logvar
        return jax.vmap(encode_single)(x)

    # Combine file lists
    files_good = glob.glob(DATA_PATTERN_GOOD)
    files_bad = glob.glob(DATA_PATTERN_BAD)
    files_random = glob.glob(DATA_PATTERN_RANDOM)
    files_iterative = glob.glob(DATA_PATTERN_ITERATIVE)
    files_recovery = glob.glob(DATA_PATTERN_RECOVERY)
    files_aggressive = glob.glob(DATA_PATTERN_AGGRESSIVE)
    files_on_policy = glob.glob(DATA_PATTERN_ON_POLICY)
    
    files = (files_good + files_bad + files_random + files_iterative + 
             files_recovery + files_aggressive + files_on_policy)
    
    # Shuffle to mix them up during processing (optional but good practice)
    np.random.shuffle(files)

    print(f"Processing {len(files)} episodes (All Datasets + Mirroring)...")

    for i, f in enumerate(tqdm(files)):
        try:
            with np.load(f) as data:
                obs = data['obs']     
                actions = data['actions']
                rewards = data['rewards']
                dones = data['dones']
        except Exception as e:
            print(f"Skipping corrupt file {f}: {e}")
            continue

        # --- Helper to encode and save a sequence ---
        def save_sequence(obs_in, actions_in, suffix):
            n_steps = obs_in.shape[0]
            mu_seq = []
            logvar_seq = []

            # Batch processing to save GPU memory
            for j in range(0, n_steps, BATCH_SIZE):
                batch_obs = obs_in[j : j + BATCH_SIZE]
                
                # Prepare for JAX (Normalize + Transpose)
                batch_jax = jnp.array(batch_obs, dtype=jnp.float32) / 255.0
                batch_jax = jnp.transpose(batch_jax, (0, 3, 1, 2))
                
                mu, logvar = encode_batch(model, batch_jax)
                
                mu_seq.append(np.array(mu))
                logvar_seq.append(np.array(logvar))

            mu_data = np.concatenate(mu_seq, axis=0)
            logvar_data = np.concatenate(logvar_seq, axis=0)

            np.savez_compressed(
                os.path.join(OUTPUT_DIR, f"series_{i}{suffix}.npz"),
                mu=mu_data,
                logvar=logvar_data,
                actions=actions_in,
                rewards=rewards,
                dones=dones
            )

        # --- 1. Save Original ---
        save_sequence(obs, actions, "_orig")

        # --- 2. Save Mirrored (The "Anti-Spin" Fix) ---
        # Flip image horizontally (Axis 2 is width for format N,H,W,C)
        obs_flipped = np.flip(obs, axis=2)
        
        # Negate steering (Action index 0)
        actions_flipped = actions.copy()
        actions_flipped[:, 0] *= -1.0 
        
        save_sequence(obs_flipped, actions_flipped, "_flip")

    print("Data processing complete. Dataset size effectively doubled.")

if __name__ == "__main__":
    process_data()