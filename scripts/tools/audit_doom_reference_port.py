"""Verify the supplied model port against independent NumPy computations on CPU."""
# ruff: noqa: E402 -- configure imports and CPU before loading JAX.

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import jax
import jax.numpy as jnp
import numpy as np

from src.doom_reference import checked_reference_arrays, load_author_models
from tests.test_doom_reference import numpy_decode, numpy_encode, numpy_step


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("Preserve completed numeric audit")
    if jax.default_backend() != "cpu":
        raise RuntimeError("This is a CPU-only numeric audit")
    payload, manifest = checked_reference_arrays(args.reference_dir)
    vae, rnn, controller, _ = load_author_models(args.reference_dir)
    vp = [np.asarray(p, np.float32) / 10000 for p in payload["vae.json"]]
    rp = [np.asarray(p, np.float32) / 10000 for p in payload["rnn.json"]]
    rng = np.random.default_rng(83)
    errors = {}
    for index in range(3):
        image = rng.uniform(0, 1, (64, 64, 3)).astype(np.float32)
        expected_mu, expected_lv = numpy_encode(vp, image)
        actual_mu, actual_lv = vae.encode(jnp.asarray(image.transpose(2, 0, 1)))
        for label, actual, expected in (
            ("mu", actual_mu, expected_mu),
            ("logvar", actual_lv, expected_lv),
        ):
            np.testing.assert_allclose(actual, expected, rtol=3e-4, atol=3e-4)
            errors[f"encoder{index}_{label}_max_abs"] = float(
                np.max(np.abs(np.asarray(actual) - expected))
            )
    z = rng.normal(size=64).astype(np.float32)
    actual = np.asarray(vae.decoder(jnp.asarray(z))).transpose(1, 2, 0)
    expected = numpy_decode(vp, z)
    np.testing.assert_allclose(actual, expected, rtol=3e-4, atol=3e-4)
    errors["decoder_max_abs"] = float(np.max(np.abs(actual - expected)))
    numpy_hidden = (np.zeros(512, np.float32), np.zeros(512, np.float32))
    jax_hidden = rnn.init_state()
    recurrent_errors = []
    policy_errors = []
    for step in range(128):
        inputs = rng.normal(size=65).astype(np.float32)
        restart = step in (0, 43, 94)
        expected, numpy_hidden = numpy_step(rp, inputs, numpy_hidden, restart)
        prediction, jax_hidden = rnn(jnp.asarray(inputs), jax_hidden, float(restart))
        for actual, wanted in zip(jax_hidden, numpy_hidden):
            np.testing.assert_allclose(actual, wanted, rtol=5e-4, atol=5e-4)
            recurrent_errors.append(float(np.max(np.abs(np.asarray(actual) - wanted))))
        mixture = expected[1:].reshape(64, 15)
        expected_logits = mixture[:, :5]
        expected_logits = expected_logits - np.logaddexp.reduce(
            expected_logits, axis=-1, keepdims=True
        )
        for actual, wanted in (
            (prediction[0].T, expected_logits),
            (prediction[1].T, mixture[:, 5:10]),
            (prediction[2].T, mixture[:, 10:]),
            (prediction[-1], expected[:1]),
        ):
            np.testing.assert_allclose(actual, wanted, rtol=5e-4, atol=1e-3)
        expected_action = np.tanh(
            np.concatenate([inputs[:64], numpy_hidden[1], numpy_hidden[0]])
            @ np.asarray(controller)
        )
        actual_action = np.tanh(
            np.concatenate(
                [inputs[:64], np.asarray(jax_hidden[1]), np.asarray(jax_hidden[0])]
            )
            @ np.asarray(controller)
        )
        np.testing.assert_allclose(actual_action, expected_action, rtol=1e-4, atol=1e-4)
        policy_errors.append(float(abs(actual_action - expected_action)))
    errors["recurrent_state_max_abs_128_steps"] = max(recurrent_errors)
    errors["controller_action_max_abs"] = max(policy_errors)
    result = dict(
        verified_at=datetime.now(timezone.utc).isoformat(),
        device=str(jax.devices()[0]),
        reference_commit=manifest["reference_commit"],
        source_weights=manifest["files"],
        encoder_frames=3,
        recurrent_steps=128,
        independent_numpy_oracle=True,
        tolerances=dict(
            encoder_decoder=dict(rtol=3e-4, atol=3e-4),
            recurrent_state=dict(rtol=5e-4, atol=5e-4),
            recurrent_heads=dict(rtol=5e-4, atol=1e-3),
            controller_action=dict(rtol=1e-4, atol=1e-4),
        ),
        errors=errors,
        passed=True,
        limitation="NumPy equations and scatter convolutions independently check layout, gates and saved arrays. This is not a live TensorFlow runtime comparison; original Pillow version and exact paper checkpoint provenance remain unverified. Supplied weights do not reproduce our training.",
        port_source_sha256={
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                Path("src/doom_reference.py"),
                Path("tests/test_doom_reference.py"),
                Path(__file__),
            )
        },
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(passed=True, errors=errors)), flush=True)


if __name__ == "__main__":
    main()
