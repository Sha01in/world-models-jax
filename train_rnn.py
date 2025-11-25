import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import numpy as np
import glob
import os
import sys
import argparse
from src.rnn import MDNRNN
from src.config import get_config
from tqdm import tqdm

# Settings (Defaults)
BATCH_SIZE = 100
LEARNING_RATE = 1e-3
EPOCHS = 20 

def load_dataset(data_dir):
    files = glob.glob(os.path.join(data_dir, "*.npz"))
    if not files:
        print(f"\n[ERROR] No processed data found in '{data_dir}'")
        print("You must collect and process data before training the RNN.")
        print("1. Run: python collect_data.py --env <env_name>")
        print("2. Run: python run_vae_training.py --env <env_name>")
        print("3. Run: python process_data.py --env <env_name>\n")
        sys.exit(1)
        
    print(f"Found {len(files)} processed episodes in {data_dir}.")
    
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
    if not all_z:
        print("No data loaded.")
        sys.exit(1)

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
    
    # 2. Reward Loss (Asymmetric MSE)
    # r_seq is (B, T, 1), targets_r is (B, T) -> expand
    targets_r_exp = jnp.expand_dims(targets_r, -1)
    diff = r_seq - targets_r_exp
    
    # penalize "Optimism" (Pred > Actual) significantly more than "Pessimism"
    # If diff > 0 (Pred > Actual), we multiply loss by 5.0
    # Base weight is 10.0 (from previous setup), so Optimism gets 50.0 effectively
    asymmetric_weight = jnp.where(diff > 0, 5.0, 1.0)
    
    loss_reward = jnp.mean(asymmetric_weight * (diff ** 2))
    
    # 3. Done Loss (BCE)
    targets_d_exp = jnp.expand_dims(targets_d, -1)
    loss_done = optax.sigmoid_binary_cross_entropy(d_seq, targets_d_exp).mean()
    
    # Total Loss with Weights
    # We prioritize Reward/Done accuracy to fix Sim2Real gap
    weighted_loss = (1.0 * loss_mdn) + (10.0 * loss_reward) + (10.0 * loss_done)
    
    return weighted_loss, (loss_mdn, loss_reward, loss_done)

@eqx.filter_jit
def make_step(model, opt_state, inputs, tz, tr, td, key, optimizer):
    (loss, aux), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model, inputs, tz, tr, td, key)
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

    Zs, Actions, Rewards, Dones = load_dataset(data_dir)
    
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
    
    # Initialize Model with Config
    model = MDNRNN(
        latent_dim=config.latent_dim,
        action_dim=config.action_dim,
        hidden_size=config.hidden_size,
        key=key
    )
    
    optimizer = optax.adam(LEARNING_RATE)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    
    print(f"Starting RNN (Dream) training for {env_name} on {num_samples} sequences...")
    print(f"Config: Latent={config.latent_dim}, Hidden={config.hidden_size}, Action={config.action_dim}")
    
    steps_per_epoch = num_samples // batch_size
    
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(epochs):
        key, subkey = jax.random.split(key)
        perms = jax.random.permutation(subkey, num_samples)
        
        epoch_loss = 0
        with tqdm(range(steps_per_epoch), desc=f"Epoch {epoch+1}/{epochs}", unit="batch") as pbar:
            for i in pbar:
                idx = perms[i*batch_size : (i+1)*batch_size]
                model, opt_state, loss, (l_mdn, l_rew, l_done) = make_step(
                    model, opt_state, 
                    inputs[idx], targets_z[idx], targets_r[idx], targets_d[idx], 
                    key, optimizer
                )
                epoch_loss += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.2f}", 
                                 mdn=f"{l_mdn.item():.2f}", 
                                 rew=f"{l_rew.item():.2f}", 
                                 done=f"{l_done.item():.2f}")
        
        # Save periodically
        if (epoch + 1) % 5 == 0:
            eqx.tree_serialise_leaves(model_path, model)
            
    eqx.tree_serialise_leaves(model_path, model)
    print("RNN Training Complete.")

if __name__ == "__main__":
    train()