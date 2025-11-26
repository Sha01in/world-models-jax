import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import numpy as np
import glob
import os
import sys
import argparse
import random
from src.rnn import MDNRNN
from src.config import get_config
from tqdm import tqdm

# Settings (Defaults)
BATCH_SIZE = 100
LEARNING_RATE = 1e-3
EPOCHS = 20 
MAX_SEQ_LEN = 2100 

def pad_sequence(sequences, max_len, pad_value=0.0):
    batch_size = len(sequences)
    # Check dim
    if sequences[0].ndim > 1:
        feat_dim = sequences[0].shape[1]
        padded = np.full((batch_size, max_len, feat_dim), pad_value, dtype=np.float32)
    else:
        padded = np.full((batch_size, max_len), pad_value, dtype=np.float32)
        
    mask = np.zeros((batch_size, max_len), dtype=np.float32)
    
    for i, seq in enumerate(sequences):
        length = min(len(seq), max_len)
        if sequences[0].ndim > 1:
            padded[i, :length, :] = seq[:length]
        else:
            padded[i, :length] = seq[:length]
        mask[i, :length] = 1.0
        
    return padded, mask

def get_file_paths(data_dir):
    files = glob.glob(os.path.join(data_dir, "*.npz"))
    if not files:
        print(f"\n[ERROR] No processed data found in '{data_dir}'")
        sys.exit(1)
    print(f"Found {len(files)} processed episodes in {data_dir}.")
    return files

def load_batch(files):
    all_z = []
    all_actions = []
    all_rewards = []
    all_dones = []
    
    for f in files:
        try:
            with np.load(f) as data:
                all_z.append(data['mu'])
                all_actions.append(data['actions'])
                all_rewards.append(data['rewards'])
                all_dones.append(data['dones'])
        except Exception as e:
            print(f"Error loading {f}: {e}")
            continue
            
    if not all_z:
        return None, None, None, None, None

    # Pad sequences
    X_z, mask = pad_sequence(all_z, MAX_SEQ_LEN)
    X_action, _ = pad_sequence(all_actions, MAX_SEQ_LEN)
    X_reward, _ = pad_sequence(all_rewards, MAX_SEQ_LEN)
    X_done, _ = pad_sequence(all_dones, MAX_SEQ_LEN)
    
    return X_z, X_action, X_reward, X_done, mask

def loss_fn(model, inputs, targets_z, targets_r, targets_d, mask, key):
    # Scan over sequence
    def step_fn(carry, x):
        hidden = carry
        (log_pi, mu, log_sigma, r_pred, d_pred), new_hidden = jax.vmap(model)(x, hidden)
        return new_hidden, (log_pi, mu, log_sigma, r_pred, d_pred)

    init_h = jax.vmap(lambda _: model.init_state())(jnp.arange(inputs.shape[0]))
    _, (log_pi, mu, log_sigma, r_seq, d_seq) = jax.lax.scan(step_fn, init_h, jnp.transpose(inputs, (1, 0, 2)))
    
    # Transpose back
    log_pi = jnp.transpose(log_pi, (1, 0, 2, 3))
    mu = jnp.transpose(mu, (1, 0, 2, 3))
    log_sigma = jnp.transpose(log_sigma, (1, 0, 2, 3))
    r_seq = jnp.transpose(r_seq, (1, 0, 2))
    d_seq = jnp.transpose(d_seq, (1, 0, 2))
    
    mask_seq = mask # (B, T)
    mask_expanded = jnp.expand_dims(mask_seq, -1) # (B, T, 1)
    
    # 1. MDN Loss
    y_z = jnp.expand_dims(targets_z, axis=2)
    sigma = jnp.exp(log_sigma)
    log_prob = -0.5 * (jnp.log(2 * jnp.pi) + 2 * log_sigma + ((y_z - mu) / sigma) ** 2)
    log_prob = jnp.sum(log_prob, axis=-1, keepdims=True)
    total_log_prob = jax.nn.logsumexp(log_pi + log_prob, axis=2)
    
    masked_log_prob = total_log_prob * mask_expanded
    total_valid_steps = jnp.sum(mask)
    loss_mdn = -jnp.sum(masked_log_prob) / (total_valid_steps + 1e-8)
    
    # 2. Reward Loss
    targets_r_exp = jnp.expand_dims(targets_r, -1)
    diff = r_seq - targets_r_exp
    asymmetric_weight = jnp.where(diff > 0, 5.0, 1.0)
    loss_reward = jnp.sum(asymmetric_weight * (diff ** 2) * mask_expanded) / (total_valid_steps + 1e-8)
    
    # 3. Done Loss
    targets_d_exp = jnp.expand_dims(targets_d, -1)
    bce = optax.sigmoid_binary_cross_entropy(d_seq, targets_d_exp)
    loss_done = jnp.sum(bce * mask_expanded) / (total_valid_steps + 1e-8)
    
    weighted_loss = (1.0 * loss_mdn) + (10.0 * loss_reward) + (10.0 * loss_done)
    
    return weighted_loss, (loss_mdn, loss_reward, loss_done)

@eqx.filter_jit
def make_step(model, opt_state, inputs, tz, tr, td, mask, key, optimizer):
    (loss, aux), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model, inputs, tz, tr, td, mask, key)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss, aux

def train():
    parser = argparse.ArgumentParser(description="Train MDN-RNN World Model")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--env", type=str, default="CarRacing-v3", help="Environment name")
    args = parser.parse_args()

    epochs = args.epochs
    batch_size = args.batch_size
    env_name = args.env
    
    config = get_config(env_name)
    
    # Paths
    data_dir = os.path.join("data/series", env_name)
    checkpoint_dir = os.path.join("checkpoints", env_name)
    model_path = os.path.join(checkpoint_dir, "rnn.eqx")

    # Get all files
    all_files = get_file_paths(data_dir)
    num_samples = len(all_files)
    
    key = jax.random.PRNGKey(42)
    
    # Initialize Model
    model = MDNRNN(
        latent_dim=config.latent_dim,
        action_dim=config.action_dim,
        hidden_size=config.hidden_size,
        key=key
    )
    
    optimizer = optax.adam(LEARNING_RATE)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    
    print(f"Starting RNN (Dream) training for {env_name} on {num_samples} sequences (Streaming)...")
    print(f"Config: Latent={config.latent_dim}, Hidden={config.hidden_size}, Action={config.action_dim}")
    print(f"Max Seq Len: {MAX_SEQ_LEN}")
    
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)

    steps_per_epoch = num_samples // batch_size
    
    for epoch in range(epochs):
        # Shuffle files
        random.shuffle(all_files)
        
        epoch_loss = 0
        with tqdm(range(steps_per_epoch), desc=f"Epoch {epoch+1}/{epochs}", unit="batch") as pbar:
            for i in pbar:
                # Load Batch from Disk
                batch_files = all_files[i*batch_size : (i+1)*batch_size]
                Zs, Actions, Rewards, Dones, Mask = load_batch(batch_files)
                
                if Zs is None: continue

                # Prepare Data
                inputs_z = Zs[:, :-1, :]
                inputs_a = Actions[:, :-1, :]
                targets_z = Zs[:, 1:, :]
                targets_r = Rewards[:, 1:]
                targets_d = Dones[:, 1:]
                mask_train = Mask[:, 1:]
                
                inputs = np.concatenate([inputs_z, inputs_a], axis=-1)
                
                # To JAX
                inputs = jnp.array(inputs)
                targets_z = jnp.array(targets_z)
                targets_r = jnp.array(targets_r)
                targets_d = jnp.array(targets_d, dtype=jnp.float32)
                mask_train = jnp.array(mask_train, dtype=jnp.float32)
                
                # Train Step
                key, subkey = jax.random.split(key)
                model, opt_state, loss, (l_mdn, l_rew, l_done) = make_step(
                    model, opt_state, 
                    inputs, targets_z, targets_r, targets_d, mask_train,
                    subkey, optimizer
                )
                
                epoch_loss += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.2f}", 
                                 mdn=f"{l_mdn.item():.2f}", 
                                 rew=f"{l_rew.item():.2f}", 
                                 done=f"{l_done.item():.2f}")
        
        if (epoch + 1) % 5 == 0:
            eqx.tree_serialise_leaves(model_path, model)
            
    eqx.tree_serialise_leaves(model_path, model)
    print("RNN Training Complete.")

if __name__ == "__main__":
    train()