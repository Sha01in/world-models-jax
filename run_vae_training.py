import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import numpy as np
import glob
import os
import random
from tqdm import tqdm
from src.vae import VAE

# --- Settings ---
# Load BOTH datasets
DATA_PATTERN_GOOD = "data/rollouts/*.npz"
DATA_PATTERN_BAD = "data/rollouts_bad/*.npz"
CHECKPOINT_DIR = "checkpoints"
MODEL_PATH = os.path.join(CHECKPOINT_DIR, "vae.eqx")

# Training Hyperparameters
LATENT_DIM = 32
LEARNING_RATE = 1e-4
BATCH_SIZE = 128
KL_TOLERANCE = 0.5
EPOCHS = 30
FILES_PER_CHUNK = 50  # Only load 50 episodes into RAM at a time (prevents OOM)

# --- Training Logic (JIT Compiled) ---
def loss_fn(model, batch, key):
    # batch: (B, 3, 64, 64) float32
    recon, mu, logvar = jax.vmap(model)(batch, jax.random.split(key, batch.shape[0]))
    
    # Reconstruction (MSE)
    recon_loss = jnp.sum((batch - recon) ** 2, axis=(1, 2, 3)) 
    
    # KL Divergence
    kl_loss = -0.5 * jnp.sum(1 + logvar - jnp.square(mu) - jnp.exp(logvar), axis=1)
    kl_loss = jnp.maximum(kl_loss, KL_TOLERANCE * LATENT_DIM)
    
    return jnp.mean(recon_loss + kl_loss)

@eqx.filter_jit
def make_step(model, opt_state, batch, key, optimizer):
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model, batch, key)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss

# --- Data Loader ---
def load_chunk(files):
    """Loads a list of .npz files into a single uint8 numpy array"""
    obs_list = []
    for f in files:
        try:
            with np.load(f) as data:
                obs = data['obs'] # (T, 64, 64, 3)
                # Transpose to Channel First: (T, 3, 64, 64)
                obs = np.transpose(obs, (0, 3, 1, 2))
                obs_list.append(obs)
        except Exception as e:
            print(f"\nSkipping corrupt file {f}: {e}")
            
    if not obs_list:
        return None
        
    return np.concatenate(obs_list, axis=0) # Returns uint8 array

# --- Main Loop ---
def main():
    # 1. Find Files
    files_good = glob.glob(DATA_PATTERN_GOOD)
    files_bad = glob.glob(DATA_PATTERN_BAD)
    all_files = files_good + files_bad
    random.shuffle(all_files)
    
    total_files = len(all_files)
    print(f"Found {len(files_good)} Good + {len(files_bad)} Bad = {total_files} Total Episodes")
    
    if total_files == 0:
        print("No data found! Run data collection first.")
        return

    # 2. Initialize Model
    key = jax.random.PRNGKey(0)
    # Try to load existing model to continue training? 
    # Or start fresh? Let's start fresh to ensure we adapt to the new data fully.
    print("Initializing new VAE...")
    model = VAE(latent_dim=LATENT_DIM, key=key)
    
    optimizer = optax.adam(LEARNING_RATE)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    
    # 3. Training Loop
    print(f"Starting Chunked Training ({FILES_PER_CHUNK} files/chunk)...")
    
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)

    for epoch in range(EPOCHS):
        # Shuffle files every epoch so chunks are different
        random.shuffle(all_files)
        
        epoch_loss = 0
        total_batches = 0
        
        # Progress bar for the chunks
        num_chunks = (total_files + FILES_PER_CHUNK - 1) // FILES_PER_CHUNK
        
        with tqdm(total=num_chunks, desc=f"Epoch {epoch+1}/{EPOCHS}") as pbar:
            for i in range(0, total_files, FILES_PER_CHUNK):
                # A. Load Chunk
                chunk_files = all_files[i : i + FILES_PER_CHUNK]
                data_chunk = load_chunk(chunk_files)
                
                if data_chunk is None:
                    continue
                
                # B. Train on Chunk
                num_samples = data_chunk.shape[0]
                steps_in_chunk = num_samples // BATCH_SIZE
                
                # Shuffle the frames in memory
                perms = np.random.permutation(num_samples)
                data_chunk = data_chunk[perms]
                
                for j in range(steps_in_chunk):
                    # Get batch (uint8)
                    batch_uint8 = data_chunk[j*BATCH_SIZE : (j+1)*BATCH_SIZE]
                    
                    # Convert to float32/Normalize ON THE FLY (Saves RAM)
                    batch = jnp.array(batch_uint8, dtype=jnp.float32) / 255.0
                    
                    # Step
                    step_key = jax.random.fold_in(key, epoch*10000 + total_batches)
                    model, opt_state, loss = make_step(model, opt_state, batch, step_key, optimizer)
                    
                    epoch_loss += loss.item()
                    total_batches += 1
                
                # Update pbar
                current_avg = epoch_loss / total_batches if total_batches > 0 else 0
                pbar.set_postfix(loss=f"{current_avg:.2f}")
                pbar.update(1)
                
                # Free RAM
                del data_chunk

        print(f"Epoch {epoch+1} Completed. Saving checkpoint...")
        eqx.tree_serialise_leaves(MODEL_PATH, model)

    print("VAE Training Complete.")

if __name__ == "__main__":
    main()