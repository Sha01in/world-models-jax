import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import numpy as np
from src.vae import VAE

# Hyperparameters
LATENT_DIM = 32
LEARNING_RATE = 1e-4
BATCH_SIZE = 128  # Increased batch size for GPU efficiency
KL_TOLERANCE = 0.5 

def loss_fn(model, batch, key):
    # batch is now float32 (0.0 to 1.0)
    recon, mu, logvar = jax.vmap(model)(batch, jax.random.split(key, batch.shape[0]))
    
    recon_loss = jnp.sum((batch - recon) ** 2, axis=(1, 2, 3)) 
    kl_loss = -0.5 * jnp.sum(1 + logvar - jnp.square(mu) - jnp.exp(logvar), axis=1)
    kl_loss = jnp.maximum(kl_loss, KL_TOLERANCE * LATENT_DIM)

    return jnp.mean(recon_loss + kl_loss)

@eqx.filter_jit
def make_step(model, opt_state, batch, key, optimizer):
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model, batch, key)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss

def train_vae(data, epochs=5):
    # data is passed in as uint8 (0-255) to save RAM
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    
    model = VAE(latent_dim=LATENT_DIM, key=subkey)
    optimizer = optax.adam(LEARNING_RATE)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    
    num_samples = data.shape[0]
    steps_per_epoch = num_samples // BATCH_SIZE
    
    print(f"Training on {num_samples} frames with batch size {BATCH_SIZE}...")
    
    for epoch in range(epochs):
        key, subkey = jax.random.split(key)
        perms = jax.random.permutation(subkey, num_samples)
        
        # Shuffling a uint8 array is fast and cheap
        data = data[perms]
        
        epoch_loss = 0
        for i in range(steps_per_epoch):
            batch_key = jax.random.fold_in(key, i)
            
            # Slice the uint8 data
            batch_uint8 = data[i*BATCH_SIZE : (i+1)*BATCH_SIZE]
            
            # CONVERSION HAPPENS HERE (On demand)
            # Convert to JAX array -> float32 -> Normalize
            batch = jnp.array(batch_uint8, dtype=jnp.float32) / 255.0
            
            model, opt_state, loss = make_step(model, opt_state, batch, batch_key, optimizer)
            epoch_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/steps_per_epoch:.4f}")
        
    return model