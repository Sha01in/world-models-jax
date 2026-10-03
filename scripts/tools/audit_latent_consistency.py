"""Check that legacy latent episodes were encoded by the current VAE."""

import argparse
import hashlib
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

from src.vae import VAE


def signature(data):
    digest = hashlib.sha256()
    for name in ("actions", "rewards", "dones"):
        digest.update(np.asarray(data[name], np.float32).tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--output", default="artifacts/doom_latent_consistency.json")
    args = parser.parse_args()
    raw_files = sorted(Path("data/rollouts/VizdoomTakeCover-v0").rglob("*.npz"))
    series = sorted(Path("data/series/VizdoomTakeCover-v0").glob("*.npz"))
    rng = np.random.default_rng(42)
    sampled = rng.choice(len(series), min(args.episodes, len(series)), replace=False)
    wanted = {}
    for index in sampled:
        with np.load(series[index]) as data:
            wanted.setdefault(signature(data), []).append(series[index])
    matched = {}
    for file in raw_files:
        with np.load(file) as data:
            key = signature(data)
        if key in wanted:
            matched.setdefault(key, []).append(file)

    model = eqx.tree_deserialise_leaves(
        "checkpoints/VizdoomTakeCover-v0/vae.eqx", VAE(64, jax.random.PRNGKey(0))
    )

    @jax.jit
    def encode(images):
        def one(image):
            features = model.encoder(image).reshape(-1)
            return model.mu_head(features)

        return jax.vmap(one)(images)

    checks = []
    for key, paths in wanted.items():
        raw = matched.get(key, [])
        if len(raw) != 1:
            continue
        with np.load(raw[0]) as data:
            observations = data["obs"]
        indices = [0, len(observations) // 2, len(observations) - 1]
        images = np.transpose(
            observations[indices].astype(np.float32) / 255, (0, 3, 1, 2)
        )
        current_mu = np.asarray(encode(jnp.asarray(images)))
        for path in paths:
            with np.load(path) as data:
                recorded_mu = data["mu"][indices]
            checks.append(
                {
                    "raw": str(raw[0]),
                    "series": str(path),
                    "mean_absolute_error": float(
                        np.mean(np.abs(current_mu - recorded_mu))
                    ),
                    "max_absolute_error": float(
                        np.max(np.abs(current_mu - recorded_mu))
                    ),
                }
            )
    report = {
        "sampled_series": len(sampled),
        "raw_episodes": len(raw_files),
        "unmatched_or_ambiguous_signatures": sum(
            len(paths)
            for key, paths in wanted.items()
            if len(matched.get(key, [])) != 1
        ),
        "checked_series": len(checks),
        "stale_series": sum(c["max_absolute_error"] > 1e-3 for c in checks),
        "checks": checks,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "checks"}, indent=2))


if __name__ == "__main__":
    main()
