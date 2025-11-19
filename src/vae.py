import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Tuple

class Encoder(eqx.Module):
    layers: list
    
    def __init__(self, key):
        keys = jax.random.split(key, 5)
        # Added padding=1 to ensure power-of-2 scaling (64->32->16->8->4)
        self.layers = [
            eqx.nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1, key=keys[0]), 
            eqx.nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, key=keys[1]), 
            eqx.nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, key=keys[2]), 
            eqx.nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, key=keys[3]), 
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = jax.nn.relu(layer(x))
        return x

class Decoder(eqx.Module):
    linear: eqx.nn.Linear
    layers: list

    def __init__(self, latent_dim, key):
        keys = jax.random.split(key, 5)
        self.linear = eqx.nn.Linear(latent_dim, 4*4*256, key=keys[0])
        
        # Added padding=1 so it mirrors the encoder (4->8->16->32->64)
        self.layers = [
            eqx.nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, key=keys[1]),
            eqx.nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, key=keys[2]),
            eqx.nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, key=keys[3]),
            eqx.nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1, key=keys[4]),
        ]

    def __call__(self, x):
        x = self.linear(x)
        x = jnp.reshape(x, (256, 4, 4)) 
        
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = jax.nn.relu(x)
            else:
                x = jax.nn.sigmoid(x) 
        return x

class VAE(eqx.Module):
    encoder: Encoder
    decoder: Decoder
    mu_head: eqx.nn.Linear
    logvar_head: eqx.nn.Linear

    def __init__(self, latent_dim=32, key=None):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.encoder = Encoder(k1)
        self.decoder = Decoder(latent_dim, k2)
        
        self.mu_head = eqx.nn.Linear(4096, latent_dim, key=k3)
        self.logvar_head = eqx.nn.Linear(4096, latent_dim, key=k4)

    def __call__(self, x, key=None) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        features = self.encoder(x)
        features = jnp.reshape(features, (-1,)) 
        
        mu = self.mu_head(features)
        logvar = self.logvar_head(features)
        
        std = jnp.exp(0.5 * logvar)
        eps = jax.random.normal(key, mu.shape)
        z = mu + std * eps
        
        recon = self.decoder(z)
        return recon, mu, logvar