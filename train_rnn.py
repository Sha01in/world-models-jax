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
BATCH_SIZE = 100  # Number of sequences per batch
SEQ_LEN = 999     # 1000 steps - 1 (since we predict next step)
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
    
    for f in files:
        with np.load(f) as data:
            # We need to sample Z from mu/logvar to train robustly
            # For simplicity in this version, we use the mean (mu) as the input
            # Ideally, you sample: z = mu + exp(0.5*logvar) * eps
            mu = data['mu'] 
            actions = data['actions']
            
            all_z.append(mu)
            all_actions.append(actions)
            
    # Convert to big arrays
    # Shape: (Num_Episodes, Seq_Len, Dim)
    X_z = np.array(all_z)       # (N, 1000, 32)
    X_action = np.array(all_actions) # (N, 1000, 3)
    
    return X_z, X_action

def mdn_loss_fn(model, inputs, targets, key):
    # inputs shape: (Batch, Seq_Len, 35) -> (z_t, a_t)
    # targets shape: (Batch, Seq_Len, 32) -> z_{t+1}
    
    # 1. Scan over the sequence (RNN unroll)
    def step_fn(carry, x):
        hidden = carry
        # x is (Batch, 35)
        # We vmap the model over the batch dimension
        (log_pi, mu, log_sigma), new_hidden = jax.vmap(model)(x, hidden)
        return new_hidden, (log_pi, mu, log_sigma)

    # Initial hidden state (Batch, 256)
    init_h = jax.vmap(lambda _: model.init_state())(jnp.arange(inputs.shape[0]))
    
    _, (log_pi_seq, mu_seq, log_sigma_seq) = jax.lax.scan(step_fn, init_h, jnp.transpose(inputs, (1, 0, 2)))
    
    # Scan output is (Seq_Len, Batch, ...), transpose back to (Batch, Seq_Len, ...)
    log_pi = jnp.transpose(log_pi_seq, (1, 0, 2, 3))      # (B, T, K, 1)
    mu = jnp.transpose(mu_seq, (1, 0, 2, 3))              # (B, T, K, 32)
    log_sigma = jnp.transpose(log_sigma_seq, (1, 0, 2, 3)) # (B, T, K, 32)
    
    # 2. Calculate MDN Loss (Negative Log Likelihood)
    # Target y is (B, T, 32). We need to broadcast it to K mixtures.
    y = jnp.expand_dims(targets, axis=2) # (B, T, 1, 32)
    
    # Gaussian Log Prob: -0.5 * [ log(2pi) + 2*log_sigma + (y-mu)^2 / sigma^2 ]
    # Summed over latent dimension (32) assuming diagonal covariance
    sigma = jnp.exp(log_sigma)
    log_prob_gauss = -0.5 * (jnp.log(2 * jnp.pi) + 2 * log_sigma + ((y - mu) / sigma) ** 2)
    log_prob_gauss = jnp.sum(log_prob_gauss, axis=-1, keepdims=True) # Sum over D=32 -> (B, T, K, 1)
    
    # Combine with mixture weights: log_sum_exp(log_pi + log_prob_gauss)
    # log_pi is already log_softmax'd
    total_log_prob = jax.nn.logsumexp(log_pi + log_prob_gauss, axis=2) # Sum over K -> (B, T, 1)
    
    return -jnp.mean(total_log_prob)

@eqx.filter_jit
def make_step(model, opt_state, inputs, targets, key, optimizer):
    loss, grads = eqx.filter_value_and_grad(mdn_loss_fn)(model, inputs, targets, key)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss

def train():
    # 1. Load Data
    # Z: (N, 1000, 32), Action: (N, 1000, 3)
    Zs, Actions = load_dataset()
    
    # Prepare Inputs and Targets
    # Input at t: [z_t, a_t]
    # Target at t: z_{t+1}
    
    # Slice to sequence length 999
    inputs_z = Zs[:, :-1, :]        # 0 to 998
    inputs_a = Actions[:, :-1, :]   # 0 to 998
    targets = Zs[:, 1:, :]          # 1 to 999
    
    # Concatenate inputs
    inputs = np.concatenate([inputs_z, inputs_a], axis=-1) # (N, 999, 35)
    
    # Convert to JAX
    inputs = jnp.array(inputs)
    targets = jnp.array(targets)
    
    num_samples = inputs.shape[0]
    
    # 2. Initialize Model
    key = jax.random.PRNGKey(42)
    model = MDNRNN(key=key)
    optimizer = optax.adam(LEARNING_RATE)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    
    print(f"Starting RNN training on {num_samples} sequences...")
    
    steps_per_epoch = num_samples // BATCH_SIZE
    
    for epoch in range(EPOCHS):
        key, subkey = jax.random.split(key)
        perms = jax.random.permutation(subkey, num_samples)
        
        epoch_loss = 0
        
        with tqdm(range(steps_per_epoch), desc=f"Epoch {epoch+1}/{EPOCHS}", unit="batch") as pbar:
            for i in pbar:
                idx = perms[i*BATCH_SIZE : (i+1)*BATCH_SIZE]
                batch_in = inputs[idx]
                batch_target = targets[idx]
                
                model, opt_state, loss = make_step(model, opt_state, batch_in, batch_target, key, optimizer)
                current_loss = loss.item()
                epoch_loss += current_loss
                pbar.set_postfix(loss=f"{current_loss:.4f}")
        
        print(f"Epoch {epoch+1} Avg Loss: {epoch_loss/steps_per_epoch:.4f}")
        
        # Save every 5 epochs
        if (epoch + 1) % 5 == 0:
            eqx.tree_serialise_leaves(MODEL_PATH, model)
            
    # Final Save
    eqx.tree_serialise_leaves(MODEL_PATH, model)
    print("RNN Training Complete.")

if __name__ == "__main__":
    train()