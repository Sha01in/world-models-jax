"""Encode frozen VAE means for the fixed sampled-frame feature probes."""

import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("JAX_PLATFORMS", "cuda")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import jax

from src.vae_features import encode_frame_features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True)
    parser.add_argument("--frame-cache-dir", required=True)
    parser.add_argument("--vae", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--allow-cpu-smoke", action="store_true")
    args = parser.parse_args()
    if (
        not all(device.platform == "gpu" for device in jax.devices())
        and not args.allow_cpu_smoke
    ):
        raise RuntimeError(
            "CUDA is required; CPU is allowed only for explicit smoke tests"
        )
    metadata = encode_frame_features(
        args.frame_cache_dir, args.split, args.vae, args.output_dir, args.batch_size
    )
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
