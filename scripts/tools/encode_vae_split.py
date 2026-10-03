"""Encode an explicit VAE split into separate RNN training/validation directories."""

import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("JAX_PLATFORMS", "cuda")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from src.vae import load_vae
from src.vae_data import file_sha256, load_split


def encode_split(split_path, vae_path, output, batch_size=128):
    manifest = load_split(split_path)
    vae_hash = file_sha256(vae_path)
    split_hash = file_sha256(split_path)
    output = Path(output)
    metadata_path = output / "metadata.json"
    identity = dict(split_sha256=split_hash, vae_sha256=vae_hash)
    if output.exists() and any(output.iterdir()):
        if not metadata_path.exists():
            raise FileExistsError("Existing output has no matching experiment identity")
        prior = json.loads(metadata_path.read_text())
        if any(prior.get(k) != v for k, v in identity.items()):
            raise ValueError("Use a fresh output for each VAE and split")
    output.mkdir(parents=True, exist_ok=True)
    model = load_vae(vae_path, 64)

    @eqx.filter_jit
    def encode(images):
        return jax.vmap(model.encode)(images)

    metadata = dict(
        identity,
        architecture=model.architecture,
        preprocessing=manifest["preprocessing"],
        complete=False,
        device=str(jax.devices()[0]),
        counts={},
    )
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    for partition in ("training", "validation"):
        destination = output / partition
        destination.mkdir(exist_ok=True)
        episodes = manifest[f"{partition}_episodes"]
        expected = {f"episode_{e['signature']}.npz" for e in episodes}
        if {p.name for p in destination.glob("*.npz")} - expected:
            raise ValueError("Unexpected encodings would contaminate the RNN split")
        for index, episode in enumerate(episodes, 1):
            path = destination / f"episode_{episode['signature']}.npz"
            if path.exists():
                with np.load(path, allow_pickle=False) as data:
                    if (
                        str(data["vae_sha256"]) != vae_hash
                        or str(data["source_sha256"]) != episode["sha256"]
                        or data["mu"].shape != (episode["frames"], 64)
                        or data["logvar"].shape != (episode["frames"], 64)
                    ):
                        raise ValueError(
                            "Existing encoding conflicts with this experiment"
                        )
                continue
            if file_sha256(episode["path"]) != episode["sha256"]:
                raise ValueError("Raw data changed since the episode split")
            with np.load(episode["path"], allow_pickle=False) as raw:
                observations = raw["obs"]
                actions, rewards, dones = (
                    raw[name].astype(np.float32)
                    for name in ("actions", "rewards", "dones")
                )
            if (
                observations.shape != (episode["frames"], 64, 64, 3)
                or not len(observations) == len(actions) == len(rewards) == len(dones)
                or np.any(dones[:-1])
            ):
                raise ValueError("Raw episode has inconsistent transition alignment")
            mus, logvars = [], []
            for start in range(0, len(observations), batch_size):
                images = observations[start : start + batch_size]
                count = len(images)
                batch = np.zeros((batch_size, 3, 64, 64), np.float32)
                batch[:count] = images.transpose(0, 3, 1, 2).astype(np.float32) / 255
                mu, logvar = map(np.asarray, encode(jnp.asarray(batch)))
                mus.append(mu[:count].astype(np.float16))
                logvars.append(logvar[:count].astype(np.float16))
            mu, logvar = np.concatenate(mus), np.concatenate(logvars)
            if not np.isfinite(mu).all() or not np.isfinite(logvar).all():
                raise FloatingPointError("Nonfinite encoded latents")
            temporary = Path(str(path) + ".partial")
            with temporary.open("wb") as stream:
                np.savez_compressed(
                    stream,
                    mu=mu,
                    logvar=logvar,
                    actions=actions,
                    rewards=rewards,
                    dones=dones,
                    vae_sha256=vae_hash,
                    source_sha256=episode["sha256"],
                    source=episode["path"],
                    split_sha256=split_hash,
                )
            temporary.replace(path)
            if index % 100 == 0:
                print(f"Encoded {partition} {index}/{len(episodes)}", flush=True)
        metadata["counts"][partition] = len(episodes)
    metadata["complete"] = True
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata), flush=True)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True)
    parser.add_argument("--vae", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--allow-cpu-smoke", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("Batch size must be positive")
    if not all(d.platform == "gpu" for d in jax.devices()) and not args.allow_cpu_smoke:
        raise RuntimeError(
            "CUDA required; CPU is allowed only for explicit smoke checks"
        )
    encode_split(args.split, args.vae, args.output_dir, args.batch_size)


if __name__ == "__main__":
    main()
