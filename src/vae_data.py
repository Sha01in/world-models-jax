"""Episode-level VAE splits and reproducible frame sampling."""

import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def transition_signature(data):
    digest = hashlib.sha256()
    for name in ("actions", "rewards", "dones"):
        value = np.asarray(data[name], np.float32)
        digest.update(value.tobytes())
    return digest.hexdigest()


def observation_header(path):
    # Inspect shape without decompressing every image in the archive.
    with zipfile.ZipFile(path) as archive, archive.open("obs.npy") as stream:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
        elif version == (2, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
        else:
            raise ValueError(f"Unsupported observation array header {version}: {path}")
    if len(shape) != 4 or shape[1:] != (64, 64, 3) or dtype != np.uint8 or fortran:
        raise ValueError(f"Expected uint8 (T,64,64,3) observations: {path}")
    if shape[0] < 1:
        raise ValueError(f"Empty episode: {path}")
    return shape


def load_split(path):
    manifest = json.loads(Path(path).read_text())
    groups = [manifest[name] for name in ("training_episodes", "validation_episodes")]
    if not all(groups):
        raise ValueError("Training and validation must both contain episodes")
    seen = set()
    for group in groups:
        for episode in group:
            # Signatures conservatively group identical actions/outcomes, even
            # when they could have arisen from different observation sequences.
            identity = episode["signature"]
            if identity in seen:
                raise ValueError("Duplicate episode or training/validation overlap")
            seen.add(identity)
    hashes = [{e["sha256"] for e in group} for group in groups]
    paths = [{str(Path(e["path"]).resolve()) for e in group} for group in groups]
    if hashes[0] & hashes[1] or paths[0] & paths[1]:
        raise ValueError("Raw archives overlap across training and validation")
    return manifest


def load_frames(episode, indices, *, verify=True):
    if verify and file_sha256(episode["path"]) != episode["sha256"]:
        raise ValueError(f"Raw episode changed since split: {episode['path']}")
    with np.load(episode["path"], allow_pickle=False) as data:
        images = data["obs"]
    if images.shape != (episode["frames"], 64, 64, 3) or images.dtype != np.uint8:
        raise ValueError(f"Unexpected observation shape: {episode['path']}")
    return images[indices]


def frame_indices(episode, limit, seed):
    count = episode["frames"]
    if not limit or count <= limit:
        return np.arange(count)
    # Each episode's sample is stable across ordering, architectures and runs.
    identity = int(episode["signature"][:16], 16)
    rng = np.random.default_rng(np.random.SeedSequence([seed, identity]))
    return np.sort(rng.choice(count, limit, replace=False))
