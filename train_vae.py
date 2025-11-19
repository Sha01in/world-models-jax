import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import numpy as np
from src.vae import VAE

# Hyperparameters
LATENT_DIM = 32
LEARNING_RATE = 1e-4
BATCH_SIZE = 32
KL_TOLERANCE = 0.5 # Threshold used in World Models paper

def loss_fn(model, batch, key):
    # batch shape: (B, 3, 64, 64)
    # We vmap over the batch dimension
    recon, mu, logvar = jax.vmap(model)(batch, jax.random.split(key, batch.shape[0]))
    
    # Reconstruction Loss (MSE)
    # Sum over all pixels
    recon_loss = jnp.sum((batch - recon) ** 2, axis=(1, 2, 3)) 
    
    # KL Divergence
    # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_loss = -0.5 * jnp.sum(1 + logvar - jnp.square(mu) - jnp.exp(logvar), axis=1)
    
    # Ensure minimum KL (free bits constraint from paper)
    kl_loss = jnp.maximum(kl_loss, KL_TOLERANCE * LATENT_DIM)

    return jnp.mean(recon_loss + kl_loss)

@eqx.filter_jit
def make_step(model, opt_state, batch, key, optimizer):
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model, batch, key)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss

def train_vae(data, epochs=5):
    """
    data: numpy array of shape (N, 3, 64, 64) normalized 0-1
    """
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    
    model = VAE(latent_dim=LATENT_DIM, key=subkey)
    optimizer = optax.adam(LEARNING_RATE)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    
    num_samples = data.shape[0]
    steps_per_epoch = num_samples // BATCH_SIZE
    
    print(f"Starting training on {num_samples} images...")
    
    for epoch in range(epochs):
        # Shuffle data
        key, subkey = jax.random.split(key)
        perms = jax.random.permutation(subkey, num_samples)
        data = data[perms]
        
        epoch_loss = 0
        for i in range(steps_per_epoch):
            batch_key = jax.random.fold_in(key, i)
            batch = data[i*BATCH_SIZE : (i+1)*BATCH_SIZE]
            
            # Convert to JAX array
            batch = jnp.array(batch)
            
            model, opt_state, loss = make_step(model, opt_state, batch, batch_key, optimizer)
            epoch_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/steps_per_epoch:.4f}")
        
    return model