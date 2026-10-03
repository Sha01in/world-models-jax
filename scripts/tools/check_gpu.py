"""Fail unless JAX can execute and synchronize CUDA compute, including cuDNN.

Run with the project's Linux virtualenv before launching a long training job:
    .venv/bin/python scripts/tools/check_gpu.py
"""

import os
import sys

# Fail at initialization instead of silently falling back to CPU.
os.environ.setdefault("JAX_PLATFORMS", "cuda")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np


def main():
    try:
        devices = jax.devices()
        gpu_devices = [device for device in devices if device.platform == "gpu"]
        if not gpu_devices:
            raise RuntimeError(f"CUDA GPU required; JAX found only {devices}")
        device = gpu_devices[0]
        print(f"JAX {jax.__version__}: {device.device_kind} ({device})", flush=True)
        with jax.default_device(device):
            x = jnp.ones((1024, 1024), dtype=jnp.float32)
            y = jax.jit(lambda a: a @ a)(x).block_until_ready()
            if y.device.platform != "gpu":
                raise RuntimeError(f"Matrix multiplication ran on {y.device}")
            np.testing.assert_allclose(np.asarray(y), 1024.0)

            # Exercise convolution and its backward pass, as used by the VAE.
            images = jnp.ones((2, 3, 64, 64), dtype=jnp.float32)
            weights = jnp.ones((32, 3, 4, 4), dtype=jnp.float32) / 48

            def loss(kernel):
                out = jax.lax.conv_general_dilated(
                    images,
                    kernel,
                    (2, 2),
                    "VALID",
                    dimension_numbers=("NCHW", "OIHW", "NCHW"),
                    # This smoke check expects a known FP32 result. Reduced
                    # multiplication precision can round 1/48 enough to fail it.
                    precision=jax.lax.Precision.HIGHEST,
                )
                return jnp.mean(out**2)

            value, grad = jax.jit(jax.value_and_grad(loss))(weights)
            grad.block_until_ready()
            if grad.device.platform != "gpu" or not np.isfinite(np.asarray(grad)).all():
                raise RuntimeError("GPU convolution backward pass failed")
            np.testing.assert_allclose(float(value), 1.0, rtol=1e-5)
        print("PASS: CUDA matrix multiplication and convolution backward pass.")
        return 0
    except Exception as exc:
        print(f"FAIL: GPU preflight: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
