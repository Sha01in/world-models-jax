import json
from pathlib import Path
import tempfile
import unittest

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from src.vae import VAE
from src.vae_training import (
    load_training_bundle,
    restored_selection,
    save_bundle,
)


class TestVAEContinuation(unittest.TestCase):
    def test_resume_restores_next_adam_update_and_rng(self):
        model = VAE(4, jax.random.PRNGKey(1))
        optimizer = optax.adam(0.0001)
        parameters = eqx.filter(model, eqx.is_array)
        state = optimizer.init(parameters)
        gradient = jax.tree_util.tree_map(jnp.ones_like, parameters)
        updates, state = optimizer.update(gradient, state, parameters)
        model = eqx.apply_updates(model, updates)
        key = jax.random.PRNGKey(17)
        settings = dict(seed=73, learning_rate=0.0001)
        with tempfile.TemporaryDirectory() as temporary:
            save_bundle(
                temporary,
                "vae_last",
                model,
                state,
                key,
                dict(settings, epoch=1, optimizer_steps=1),
            )
            restored, restored_state, restored_key, _ = load_training_bundle(
                temporary,
                "vae_last",
                optimizer,
                settings,
            )
            expected_updates, expected_state = optimizer.update(
                gradient,
                state,
                eqx.filter(model, eqx.is_array),
            )
            actual_updates, actual_state = optimizer.update(
                gradient,
                restored_state,
                eqx.filter(restored, eqx.is_array),
            )
            for actual, expected in zip(
                jax.tree_util.tree_leaves((actual_updates, actual_state)),
                jax.tree_util.tree_leaves((expected_updates, expected_state)),
            ):
                np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(
                jax.random.split(restored_key), jax.random.split(key)
            )
            with self.assertRaisesRegex(ValueError, "settings differ"):
                load_training_bundle(temporary, "vae_last", optimizer, dict(seed=74))
            sidecar = Path(temporary) / "vae_last.eqx.json"
            metadata = json.loads(sidecar.read_text())
            metadata["optimizer_steps"] = 2
            sidecar.write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "optimizer count"):
                load_training_bundle(temporary, "vae_last", optimizer, settings)
            metadata["optimizer_steps"] = 1
            sidecar.write_text(json.dumps(metadata))
            with (Path(temporary) / "vae_last_optimizer.eqx").open("ab") as stream:
                stream.write(b"changed")
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                load_training_bundle(temporary, "vae_last", optimizer, settings)

    def test_resume_keeps_best_and_remaining_patience(self):
        history = [
            dict(epoch=i, validation=dict(loss=value))
            for i, value in enumerate((10, 9, 9.5, 9.4))
        ]
        best = history[1]
        selection = restored_selection(history, 3, best)
        self.assertEqual(selection.stale, 2)
        self.assertFalse(selection.should_stop)
        self.assertFalse(selection.consider(9.3, 4))
        self.assertTrue(selection.should_stop)
        self.assertEqual(selection.best_epoch, 1)
        with self.assertRaisesRegex(ValueError, "best checkpoint"):
            restored_selection(history, 3, history[-1])
        with self.assertRaisesRegex(ValueError, "missing or reordered"):
            restored_selection(history[1:], 3, best)


if __name__ == "__main__":
    unittest.main()
