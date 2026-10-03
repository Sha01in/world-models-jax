"""Compare synchronized RNN optimizer throughput across dependency environments."""

import argparse
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cuda")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from src.rnn import MDNRNN
from train_rnn import make_step


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=30)
    args = parser.parse_args()
    key = jax.random.PRNGKey(42)
    model = MDNRNN(64, 1, 512, key=key, factorized=True)
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-3))
    state = optimizer.init(eqx.filter(model, eqx.is_array))
    x = jax.random.normal(key, (32, 256, 65))
    z = x[..., :64]
    r = jnp.ones((32, 256))
    d = jnp.zeros((32, 256)).at[:, -1].set(1.0)
    mask = jnp.ones((32, 256))
    timings = []
    for i in range(args.steps + 5):
        started = time.perf_counter()
        model, state, loss, _ = make_step(
            model, state, x, z, r, d, mask, key, optimizer, mask, 10.0, True
        )
        loss.block_until_ready()
        if i >= 5:
            timings.append(time.perf_counter() - started)
    result = {
        "jax": jax.__version__,
        "equinox": eqx.__version__,
        "optax": optax.__version__,
        "device": str(jax.devices()[0]),
        "batch": 32,
        "sequence": 256,
        "median_seconds": float(np.median(timings)),
        "mean_seconds": float(np.mean(timings)),
        "frames_per_second": 32 * 256 / float(np.median(timings)),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
