"""Frozen VAE means aligned with the immutable sampled-frame pools."""

import json
from pathlib import Path
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from src.vae import load_vae
from src.vae_data import file_sha256
from src.vae_training import batches, load_frame_cache


def feature_identity(frame_cache, split_path, vae_path):
    metadata_path = Path(frame_cache) / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    if not metadata["complete"] or metadata["split_sha256"] != file_sha256(split_path):
        raise ValueError(
            "Feature source frame cache has a different or incomplete split"
        )
    return dict(
        split_sha256=metadata["split_sha256"],
        vae_sha256=file_sha256(vae_path),
        frame_cache_metadata_sha256=file_sha256(metadata_path),
        training_frames=metadata["training_frames"],
        validation_frames=metadata["validation_frames"],
        frames_per_episode=metadata["frames_per_episode"],
        validation_frames_per_episode=metadata["validation_frames_per_episode"],
        seed=metadata["seed"],
    )


def load_feature_cache(directory, frame_cache, split_path, vae_path):
    """Verify means and row lineage; raw frame pools are verified by their loader."""
    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text())
    identity = feature_identity(frame_cache, split_path, vae_path)
    if not metadata.get("complete") or any(
        metadata.get(k) != v for k, v in identity.items()
    ):
        raise ValueError("Feature cache belongs to different weights, frames or split")
    means = []
    for partition in ("training", "validation"):
        path = directory / f"{partition}_mu.npy"
        if file_sha256(path) != metadata[f"{partition}_sha256"]:
            raise ValueError("Feature cache content changed")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if (
            array.shape != (identity[f"{partition}_frames"], 64)
            or array.dtype != np.float16
        ):
            raise ValueError("Feature cache has invalid shape or precision")
        means.append(array)
    return tuple(means)


def encode_frame_features(frame_cache, split_path, vae_path, directory, batch_size=128):
    """Encode sampled means only; these files are not full RNN trajectories."""
    if batch_size < 1:
        raise ValueError("Feature batch size must be positive")
    identity = feature_identity(frame_cache, split_path, vae_path)
    pools = load_frame_cache(
        frame_cache,
        split_path,
        identity["frames_per_episode"],
        identity["validation_frames_per_episode"],
        identity["seed"],
    )
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        load_feature_cache(directory, frame_cache, split_path, vae_path)
        prior = json.loads((directory / "metadata.json").read_text())
        if prior["batch_size"] != batch_size or prior["device"] != str(
            jax.devices()[0]
        ):
            raise ValueError(
                "Existing feature cache uses a different calculation shape or device"
            )
        return prior
    directory.mkdir(parents=True, exist_ok=True)
    model = load_vae(vae_path, 64)

    @eqx.filter_jit
    def encode(images):
        return jax.vmap(model.encode)(images)[0]

    metadata = dict(
        identity,
        complete=False,
        architecture=model.architecture,
        batch_size=batch_size,
        device=str(jax.devices()[0]),
        precision="float16, matching stored episode means",
        purpose="Sampled-frame feature probes only; not RNN episode data",
    )
    metadata_path = directory / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    started = time.monotonic()
    for partition, images in zip(("training", "validation"), pools):
        partial = directory / f"{partition}_mu.partial.npy"
        destination = directory / f"{partition}_mu.npy"
        means = np.lib.format.open_memmap(
            partial, mode="w+", dtype=np.float16, shape=(len(images), 64)
        )
        cursor = 0
        for index, (batch, _, count) in enumerate(batches(images, batch_size), 1):
            encoded = np.asarray(encode(jnp.asarray(batch)))[:count].astype(np.float16)
            if not np.isfinite(encoded).all():
                raise FloatingPointError("Nonfinite sampled-frame means")
            means[cursor : cursor + count] = encoded
            cursor += count
            if index % 256 == 0:
                print(
                    json.dumps(
                        dict(
                            partition=partition,
                            encoded_frames=cursor,
                            total_frames=len(images),
                        )
                    ),
                    flush=True,
                )
        means.flush()
        del means
        partial.replace(destination)
        metadata[f"{partition}_sha256"] = file_sha256(destination)
    if feature_identity(frame_cache, split_path, vae_path) != identity:
        raise ValueError("Feature sources changed during encoding")
    metadata.update(complete=True, elapsed_seconds=time.monotonic() - started)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    load_feature_cache(directory, frame_cache, split_path, vae_path)
    return metadata
