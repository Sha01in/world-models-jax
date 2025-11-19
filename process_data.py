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
DATA_PATTERN = "data/rollouts/*.npz"
OUTPUT_DIR = "data/series"
VAE_PATH = "checkpoints/vae.eqx"

def process_data():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print(f"Loading VAE from {VAE_PATH}...")
    model = VAE(latent_dim=LATENT_DIM, key=jax.random.PRNGKey(0))
    model = eqx.tree_deserialise_leaves(VAE_PATH, model)

    @eqx.filter_jit
    def encode_batch(m, x):
        def encode_single(img):
            features = m.encoder(img)
            features = jnp.reshape(features, (-1,))
            mu = m.mu_head(features)
            logvar = m.logvar_head(features)
            return mu, logvar
        return jax.vmap(encode_single)(x)

    files = glob.glob(DATA_PATTERN)
    print(f"Processing {len(files)} episodes...")

    for i, f in enumerate(tqdm(files)):
        try:
            with np.load(f) as data:
                obs = data['obs']     
                actions = data['actions']
                rewards = data['rewards'] # <--- New
                dones = data['dones']     # <--- New
        except Exception as e:
            print(f"Skipping {f}: {e}")
            continue

        n_steps = obs.shape[0]
        mu_seq = []
        logvar_seq = []

        for j in range(0, n_steps, BATCH_SIZE):
            batch_obs = obs[j : j + BATCH_SIZE]
            batch_jax = jnp.array(batch_obs, dtype=jnp.float32) / 255.0
            batch_jax = jnp.transpose(batch_jax, (0, 3, 1, 2))
            
            mu, logvar = encode_batch(model, batch_jax)
            
            mu_seq.append(np.array(mu))
            logvar_seq.append(np.array(logvar))

        mu_data = np.concatenate(mu_seq, axis=0)
        logvar_data = np.concatenate(logvar_seq, axis=0)

        save_path = os.path.join(OUTPUT_DIR, f"series_{i}.npz")
        np.savez_compressed(
            save_path,
            mu=mu_data,
            logvar=logvar_data,
            actions=actions,
            rewards=rewards,  # <--- New
            dones=dones       # <--- New
        )

    print("Data processing complete.")

if __name__ == "__main__":
    process_data()