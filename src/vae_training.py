"""Validated VAE training and checkpoint selection, independent of the CLI."""

import json
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from src.vae_data import file_sha256, frame_indices, load_frames


def loss_components(model, batch, key, weights, kl_tolerance=0.5):
    recon, mu, logvar = jax.vmap(model)(batch, jax.random.split(key, len(batch)))
    reconstruction = jnp.sum((batch - recon) ** 2, axis=(1, 2, 3))
    raw_kl = -0.5 * jnp.sum(1 + logvar - mu**2 - jnp.exp(logvar), axis=1)
    kl = jnp.maximum(raw_kl, kl_tolerance * mu.shape[-1])
    # Padding rows never enter training gradients or validation averages.
    denominator = jnp.maximum(jnp.sum(weights), 1)

    def mean(value):
        return jnp.sum(value * weights) / denominator

    return mean(reconstruction + kl), (mean(reconstruction), mean(raw_kl))


@eqx.filter_jit
def train_step(model, opt_state, batch, weights, key, optimizer, kl_tolerance):
    (loss, metrics), gradients = eqx.filter_value_and_grad(
        loss_components, has_aux=True
    )(model, batch, key, weights, kl_tolerance)
    updates, opt_state = optimizer.update(gradients, opt_state, model)
    return eqx.apply_updates(model, updates), opt_state, loss, metrics


@eqx.filter_jit
def validation_step(model, batch, weights, key, kl_tolerance):
    return loss_components(model, batch, key, weights, kl_tolerance)


def build_frame_cache(episodes, output, limit, seed):
    """One streaming read per episode; keep the image pool on disk as uint8."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Use a fresh frame cache: {output}")
    selections = [frame_indices(e, limit, seed) for e in episodes]
    count = sum(len(indices) for indices in selections)
    output.parent.mkdir(parents=True, exist_ok=True)
    images = np.lib.format.open_memmap(
        output, mode="w+", dtype=np.uint8, shape=(count, 64, 64, 3)
    )
    cursor = 0
    for index, (episode, indices) in enumerate(zip(episodes, selections), 1):
        frames = load_frames(episode, indices)
        images[cursor : cursor + len(frames)] = frames
        cursor += len(frames)
        if index % 100 == 0:
            print(
                f"Cached {index}/{len(episodes)} episodes ({cursor} frames)", flush=True
            )
    images.flush()
    del images
    return np.load(output, mmap_mode="r")


def prepare_frame_cache(manifest, split_path, directory, train_limit, val_limit, seed):
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError("Preserve existing frame pools; use a fresh directory")
    directory.mkdir(parents=True, exist_ok=True)
    paths = (directory / "training.npy", directory / "validation.npy")
    arrays = (
        build_frame_cache(manifest["training_episodes"], paths[0], train_limit, seed),
        build_frame_cache(manifest["validation_episodes"], paths[1], val_limit, seed),
    )
    metadata = dict(
        split_sha256=file_sha256(split_path),
        frames_per_episode=train_limit,
        validation_frames_per_episode=val_limit,
        seed=seed,
        training_frames=len(arrays[0]),
        validation_frames=len(arrays[1]),
        training_sha256=file_sha256(paths[0]),
        validation_sha256=file_sha256(paths[1]),
        complete=True,
    )
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return arrays


def load_frame_cache(directory, split_path, train_limit, val_limit, seed):
    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text())
    expected = dict(
        split_sha256=file_sha256(split_path),
        frames_per_episode=train_limit,
        validation_frames_per_episode=val_limit,
        seed=seed,
        complete=True,
    )
    if any(metadata.get(k) != v for k, v in expected.items()):
        raise ValueError("Frame cache settings differ from the requested experiment")
    arrays = []
    for name in ("training", "validation"):
        path = directory / f"{name}.npy"
        if file_sha256(path) != metadata[f"{name}_sha256"]:
            raise ValueError("Frame cache content changed")
        images = np.load(path, mmap_mode="r", allow_pickle=False)
        if (
            images.shape != (metadata[f"{name}_frames"], 64, 64, 3)
            or images.dtype != np.uint8
        ):
            raise ValueError("Invalid frame cache shape or dtype")
        arrays.append(images)
    return tuple(arrays)


def batches(images, batch_size, indices=None):
    indices = np.arange(len(images)) if indices is None else indices
    for start in range(0, len(indices), batch_size):
        selected = np.asarray(images[indices[start : start + batch_size]])
        count = len(selected)
        batch = np.zeros((batch_size, 3, 64, 64), np.float32)
        batch[:count] = selected.transpose(0, 3, 1, 2).astype(np.float32) / 255
        weights = np.arange(batch_size) < count
        yield batch, weights.astype(np.float32), count


def evaluate(model, images, batch_size, seed, kl_tolerance):
    total = np.zeros(3, np.float64)
    count = 0
    key = jax.random.PRNGKey(seed)
    for index, (batch, weights, n) in enumerate(batches(images, batch_size)):
        loss, (reconstruction, kl) = validation_step(
            model, batch, weights, jax.random.fold_in(key, index), kl_tolerance
        )
        total += np.asarray([loss, reconstruction, kl], np.float64) * n
        count += n
    return dict(zip(("loss", "reconstruction", "raw_kl"), (total / count).tolist()))


def save_bundle(directory, name, model, opt_state, key, metadata):
    directory = Path(directory)
    path = directory / f"{name}.eqx"
    optimizer_path = directory / f"{name}_optimizer.eqx"
    eqx.tree_serialise_leaves(path, model)
    eqx.tree_serialise_leaves(optimizer_path, opt_state)
    rng_path = directory / f"{name}_rng.npz"
    np.savez(rng_path, key=np.asarray(key))
    manifest = dict(
        metadata,
        architecture=model.architecture,
        latent_dim=model.mu_head.out_features,
        sha256=file_sha256(path),
        optimizer_path=optimizer_path.name,
        optimizer_sha256=file_sha256(optimizer_path),
        rng_path=rng_path.name,
        rng_sha256=file_sha256(rng_path),
    )
    Path(str(path) + ".json").write_text(json.dumps(manifest, indent=2) + "\n")


def load_training_bundle(directory, name, optimizer, expected_settings):
    """Restore a fingerprinted VAE, Adam state and RNG without changing its files."""
    from src.vae import load_vae

    directory = Path(directory)
    path = directory / f"{name}.eqx"
    metadata = json.loads(Path(str(path) + ".json").read_text())
    if any(metadata.get(k) != v for k, v in expected_settings.items()):
        raise ValueError("Resume bundle settings differ from this experiment")
    optimizer_path = directory / metadata["optimizer_path"]
    rng_path = directory / metadata["rng_path"]
    for source, digest in (
        (path, metadata["sha256"]),
        (optimizer_path, metadata["optimizer_sha256"]),
        (rng_path, metadata["rng_sha256"]),
    ):
        if file_sha256(source) != digest:
            raise ValueError("Resume bundle fingerprint mismatch")
    model = load_vae(path, metadata["latent_dim"])
    template = optimizer.init(eqx.filter(model, eqx.is_array))
    state = eqx.tree_deserialise_leaves(optimizer_path, template)
    if int(state[0].count) != metadata["optimizer_steps"]:
        raise ValueError("Resume optimizer count disagrees with checkpoint")
    with np.load(rng_path, allow_pickle=False) as data:
        key = np.array(data["key"])
    if key.shape != (2,) or key.dtype != np.uint32:
        raise ValueError("Resume bundle has an invalid RNG key")
    return model, state, jnp.asarray(key), metadata


def restored_selection(history, patience, expected_best):
    """Replay validation history so continuation retains existing best/patience."""
    selection = BestCheckpoint(patience)
    for epoch, record in enumerate(history):
        if record["epoch"] != epoch:
            raise ValueError("Resume history has missing or reordered epochs")
        selection.consider(record["validation"]["loss"], epoch)
    if (
        selection.best_epoch != expected_best["epoch"]
        or selection.best_loss != expected_best["validation"]["loss"]
    ):
        raise ValueError("Resume history disagrees with best checkpoint")
    return selection


class BestCheckpoint:
    """Keep epoch zero eligible and stop after consecutive non-improvements."""

    def __init__(self, patience, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.best_epoch = None
        self.stale = 0

    def consider(self, loss, epoch):
        if not np.isfinite(loss):
            raise FloatingPointError("Nonfinite VAE validation loss")
        improved = loss < self.best_loss - self.min_delta
        if improved:
            self.best_loss, self.best_epoch, self.stale = loss, epoch, 0
        else:
            self.stale += 1
        return improved

    @property
    def should_stop(self):
        return bool(self.patience and self.stale >= self.patience)
