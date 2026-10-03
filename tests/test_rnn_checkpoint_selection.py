"""Exercise the training loop's baseline selection and paired optimizer saves."""

import contextlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import optax

import train_rnn


class TinyModel(eqx.Module):
    weight: object
    factorized: bool = eqx.field(static=True, default=True)

    def __init__(self, **kwargs):
        self.weight = jnp.array(0.0)


class TestCheckpointSelection(unittest.TestCase):
    def exercise_loop(self, losses, expected_stop, expected_best):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            for subset in ("train", "validation"):
                directory = root / subset
                directory.mkdir()
                np.savez(
                    directory / "episode.npz",
                    mu=np.zeros((2, 1), np.float32),
                    logvar=np.zeros((2, 1), np.float32),
                    actions=np.zeros((2, 1), np.float32),
                    rewards=np.ones(2, np.float32),
                    dones=np.array([0.0, 1.0], np.float32),
                )
            output = root / "rnn.eqx"
            argv = [
                "train_rnn.py",
                "--env",
                "VizdoomTakeCover-v0",
                "--data_dir",
                str(root / "train"),
                "--validation_dir",
                str(root / "validation"),
                "--output",
                str(output),
                "--epochs",
                "10",
                "--batch_size",
                "1",
                "--early_stopping_patience",
                "3",
            ]
            values = jnp.asarray(losses)

            def controlled_loss(model, *args):
                return values[model.weight.astype(int)], (jnp.array(0.0),) * 3

            def controlled_step(model, state, x, z, r, d, mask, key, optimizer, *args):
                gradients = eqx.tree_at(lambda m: m.weight, model, jnp.array(1.0))
                _, state = optimizer.update(gradients, state, model)
                model = eqx.tree_at(lambda m: m.weight, model, model.weight + 1)
                return model, state, jnp.array(1.0), (jnp.array(0.0),) * 3

            config = SimpleNamespace(
                latent_dim=1, action_dim=1, hidden_size=2, is_doom=True
            )
            with (
                patch("sys.argv", argv),
                patch.object(train_rnn, "get_config", return_value=config),
                patch.object(train_rnn, "MDNRNN", TinyModel),
                patch.object(train_rnn, "loss_fn", controlled_loss),
                patch.object(train_rnn, "make_step", controlled_step),
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                train_rnn.train()

            for path, expected in [
                (output, expected_stop),
                (root / "rnn_best.eqx", expected_best),
            ]:
                model = eqx.tree_deserialise_leaves(path, TinyModel())
                self.assertEqual(float(model.weight), expected)
                optimizer = optax.chain(
                    optax.clip_by_global_norm(1.0), optax.adam(0.001)
                )
                state = eqx.tree_deserialise_leaves(
                    str(path) + ".opt.eqx",
                    optimizer.init(eqx.filter(model, eqx.is_array)),
                )
                self.assertEqual(int(state[1][0].count), expected)
                metadata = json.loads(Path(str(path) + ".json").read_text())
                self.assertEqual(metadata["trained_epochs"], expected)
                self.assertAlmostEqual(metadata["validation_loss"], losses[expected])
                self.assertEqual(metadata["global_step"], expected)
                self.assertIn(
                    "data_rng", json.loads(Path(str(path) + ".rng.json").read_text())
                )
            history = json.loads(Path(str(output) + ".history.json").read_text())
            self.assertEqual(history[0]["epoch"], 0)
            self.assertEqual(history[-1]["epoch"], expected_stop)

    def test_degrading_refinement_keeps_the_starting_model(self):
        self.exercise_loop([1.0, 1.2, 1.3, 1.4], expected_stop=3, expected_best=0)

    def test_improvement_resets_patience_and_saves_matching_optimizer(self):
        self.exercise_loop(
            [1.0, 0.9, 0.8, 0.85, 0.86, 0.87], expected_stop=5, expected_best=2
        )


if __name__ == "__main__":
    unittest.main()
