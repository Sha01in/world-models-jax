"""Visual encoders with legacy and reference World Models architectures."""

import hashlib
import json
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp


ARCHITECTURES = ("current", "paper")


class Encoder(eqx.Module):
    layers: list

    def __init__(self, key, architecture="current"):
        keys = jax.random.split(key, 5)
        padding = 1 if architecture == "current" else 0
        self.layers = [
            eqx.nn.Conv2d(a, b, 4, stride=2, padding=padding, key=k)
            for a, b, k in zip((3, 32, 64, 128), (32, 64, 128, 256), keys[:4])
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = jax.nn.relu(layer(x))
        return x


class Decoder(eqx.Module):
    linear: eqx.nn.Linear
    layers: list
    initial_shape: tuple = eqx.field(static=True)

    def __init__(self, latent_dim, key, architecture="current"):
        keys = jax.random.split(key, 5)
        if architecture == "current":
            self.initial_shape = (256, 4, 4)
            width, padding, kernels = 4096, 1, (4, 4, 4, 4)
            channels = (256, 128, 64, 32)
        else:
            # Reference: dense 1024 -> 1x1x1024 -> 5 -> 13 -> 30 -> 64.
            self.initial_shape = (1024, 1, 1)
            width, padding, kernels = 1024, 0, (5, 5, 6, 6)
            channels = (1024, 128, 64, 32)
        self.linear = eqx.nn.Linear(latent_dim, width, key=keys[0])
        self.layers = [
            eqx.nn.ConvTranspose2d(a, b, size, stride=2, padding=padding, key=k)
            for a, b, size, k in zip(channels, (128, 64, 32, 3), kernels, keys[1:])
        ]

    def __call__(self, x):
        x = self.linear(x).reshape(self.initial_shape)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            x = jax.nn.relu(x) if i < len(self.layers) - 1 else jax.nn.sigmoid(x)
        return x


class VAE(eqx.Module):
    encoder: Encoder
    decoder: Decoder
    mu_head: eqx.nn.Linear
    logvar_head: eqx.nn.Linear
    architecture: str = eqx.field(static=True)

    def __init__(self, latent_dim=32, key=None, architecture="current"):
        if architecture not in ARCHITECTURES:
            raise ValueError(f"Unknown VAE architecture: {architecture}")
        self.architecture = architecture
        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.encoder = Encoder(k1, architecture)
        self.decoder = Decoder(latent_dim, k2, architecture)
        width = 4096 if architecture == "current" else 1024
        self.mu_head = eqx.nn.Linear(width, latent_dim, key=k3)
        self.logvar_head = eqx.nn.Linear(width, latent_dim, key=k4)

    def encode(self, x):
        features = self.encoder(x).reshape(-1)
        return self.mu_head(features), self.logvar_head(features)

    def __call__(self, x, key=None):
        mu, logvar = self.encode(x)
        z = mu + jnp.exp(0.5 * logvar) * jax.random.normal(key, mu.shape)
        return self.decoder(z), mu, logvar


def load_vae(path, latent_dim, key=None):
    """Old files default to the unchanged architecture; new files carry metadata."""
    path = Path(path)
    sidecar = Path(str(path) + ".json")
    metadata = json.loads(sidecar.read_text()) if sidecar.exists() else {}
    if metadata.get("latent_dim", latent_dim) != latent_dim:
        raise ValueError("VAE latent width does not match the environment")
    if "sha256" in metadata:
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != metadata["sha256"]:
            raise ValueError("VAE checkpoint does not match its architecture sidecar")
    model = VAE(
        latent_dim,
        jax.random.PRNGKey(0) if key is None else key,
        architecture=metadata.get("architecture", "current"),
    )
    return eqx.tree_deserialise_leaves(path, model)
