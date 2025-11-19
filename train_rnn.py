import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import numpy as np
import glob
import os
from src.rnn import MDNRNN
from tqdm import tqdm

# Settings
DATA_DIR = "data/series/*.npz"
CHECKPOINT_DIR = "checkpoints"
MODEL_PATH = os.path.join(CHECKPOINT_DIR, "rnn.eqx")
BATCH_SIZE = 100
HIDDEN_SIZE = 256
LATENT_DIM = 32
ACTION_DIM = 3
LEARNING_RATE = 1e-3
EPOCHS = 20 

def load_dataset():
    files = glob.glob(DATA_DIR)
    print(f"Found {len(files)} processed episodes.")
    
    all_z = []
    all_actions = []
    all_rewards = []
    all_dones = []
    
    for f in files:
        with np.load(f) as data:
            all_z.append(data['mu'])
            all_actions.append(data['actions'])
            all_rewards.append(data['rewards'])
            all_dones.append(data['dones'])
            
    # Stack into arrays
    # Slice to equal lengths just in case (usually 1000)
    min_len = min([len(x) for x in all_z])
    
    X_z = np.array([x[:min_len] for x in all_z])
    X_action = np.array([x[:min_len] for x in all_actions])
    X_reward = np.array([x[:min_len] for x in all_rewards])
    X_done = np.array([x[:min_len] for x in all_dones])
    
    return X_z, X_action, X_reward, X_done

def loss_fn(model, inputs, targets_z, targets_r, targets_d, key):
    # Scan over sequence
    def step_fn(carry, x):
        hidden = carry
        # Output includes reward and done now
        (log_pi, mu, log_sigma, r_pred, d_pred), new_hidden = jax.vmap(model)(x, hidden)
        return new_hidden, (log_pi, mu, log_sigma, r_pred, d_pred)

    init_h = jax.vmap(lambda _: model.init_state())(jnp.arange(inputs.shape[0]))
    _, (log_pi, mu, log_sigma, r_seq, d_seq) = jax.lax.scan(step_fn, init_h, jnp.transpose(inputs, (1, 0, 2)))
    
    # Transpose outputs back to (Batch, Time, ...)
    log_pi = jnp.transpose(log_pi, (1, 0, 2, 3))
    mu = jnp.transpose(mu, (1, 0, 2, 3))
    log_sigma = jnp.transpose(log_sigma, (1, 0, 2, 3))
    r_seq = jnp.transpose(r_seq, (1, 0, 2)) # (B, T, 1)
    d_seq = jnp.transpose(d_seq, (1, 0, 2)) # (B, T, 1)
    
    # 1. MDN Loss (NLL)
    y_z = jnp.expand_dims(targets_z, axis=2)
    sigma = jnp.exp(log_sigma)
    log_prob = -0.5 * (jnp.log(2 * jnp.pi) + 2 * log_sigma + ((y_z - mu) / sigma) ** 2)
    log_prob = jnp.sum(log_prob, axis=-1, keepdims=True)
    total_log_prob = jax.nn.logsumexp(log_pi + log_prob, axis=2)
    loss_mdn = -jnp.mean(total_log_prob)
    
    # 2. Reward Loss (MSE)
    # r_seq is (B, T, 1), targets_r is (B, T) -> expand
    targets_r_exp = jnp.expand_dims(targets_r, -1)
    loss_reward = jnp.mean((r_seq - targets_r_exp) ** 2)
    
    # 3. Done Loss (BCE)
    targets_d_exp = jnp.expand_dims(targets_d, -1)
    loss_done = optax.sigmoid_binary_cross_entropy(d_seq, targets_d_exp).mean()
    
    # Total Loss
    return loss_mdn + loss_reward + loss_done

@eqx.filter_jit
def make_step(model, opt_state, inputs, tz, tr, td, key, optimizer):
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model, inputs, tz, tr, td, key)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss

def train():
    Zs, Actions, Rewards, Dones = load_dataset()
    
    # Prepare Inputs (t) and Targets (t+1)
    inputs_z = Zs[:, :-1, :]
    inputs_a = Actions[:, :-1, :]
    
    targets_z = Zs[:, 1:, :]
    targets_r = Rewards[:, 1:]  # Predict Next Reward
    targets_d = Dones[:, 1:]    # Predict Next Done
    
    inputs = np.concatenate([inputs_z, inputs_a], axis=-1)
    
    inputs = jnp.array(inputs)
    targets_z = jnp.array(targets_z)
    targets_r = jnp.array(targets_r)
    targets_d = jnp.array(targets_d, dtype=jnp.float32) # Ensure float for BCE
    
    num_samples = inputs.shape[0]
    key = jax.random.PRNGKey(42)
    model = MDNRNN(key=key)
    optimizer = optax.adam(LEARNING_RATE)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    
    print(f"Starting RNN (Dream) training on {num_samples} sequences...")
    
    steps_per_epoch = num_samples // BATCH_SIZE
    
    for epoch in range(EPOCHS):
        key, subkey = jax.random.split(key)
        perms = jax.random.permutation(subkey, num_samples)
        
        epoch_loss = 0
        with tqdm(range(steps_per_epoch), desc=f"Epoch {epoch+1}/{EPOCHS}", unit="batch") as pbar:
            for i in pbar:
                idx = perms[i*BATCH_SIZE : (i+1)*BATCH_SIZE]
                model, opt_state, loss = make_step(
                    model, opt_state, 
                    inputs[idx], targets_z[idx], targets_r[idx], targets_d[idx], 
                    key, optimizer
                )
                epoch_loss += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.4f}")
        
        # Save periodically
        if (epoch + 1) % 5 == 0:
            eqx.tree_serialise_leaves(MODEL_PATH, model)
            
    eqx.tree_serialise_leaves(MODEL_PATH, model)
    print("RNN Training Complete.")

if __name__ == "__main__":
    train()