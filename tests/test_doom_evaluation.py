import unittest
from unittest.mock import patch
import hashlib
import json
from pathlib import Path
import tempfile

import jax.numpy as jnp
import numpy as np

from src.doom_evaluation import evaluate_parallel
from scripts.tools.select_doom_controller import main as select_controller


class TestParallelEvaluation(unittest.TestCase):
    def test_exact_seeds_rng_memory_reset_and_finished_slots(self):
        class FakeVectorEnv:
            def __init__(self, factories, **kwargs):
                self.n = len(factories)
                self.seeds = np.zeros(self.n, np.int64)
                self.steps = np.zeros(self.n, np.int64)
                self.closed = False

            def reset(self, seed, options=None):
                mask = (
                    np.ones(self.n, bool) if options is None else options["reset_mask"]
                )
                for i in np.flatnonzero(mask):
                    self.seeds[i] = seed[i]
                    self.steps[i] = 0
                return self.seeds[:, None].copy(), {}

            def step(self, actions):
                self.steps += 1
                dead = self.steps >= self.seeds % 3 + 1
                # Mimic SAME_STEP. Completed slots may continue safely, while
                # the scheduler explicitly reseeds all newly assigned games.
                self.steps[dead] = 0
                return (
                    self.seeds[:, None].copy(),
                    np.ones(self.n),
                    dead,
                    np.zeros(self.n, bool),
                    {},
                )

            def close(self):
                self.closed = True

        calls = []

        def checked_policy(obs, hidden, keys, steps):
            for i in range(len(obs)):
                if steps[i] == 0:
                    np.testing.assert_array_equal(np.asarray(hidden[0])[i], 0)
                    np.testing.assert_array_equal(np.asarray(hidden[1])[i], 0)
                    self.assertEqual(int(keys[i, 1]), int(obs[i, 0]))
                    calls.append(int(obs[i, 0]))
            return jnp.zeros((len(obs), 1)), tuple(h + 1 for h in hidden), keys

        with patch("src.doom_evaluation.AsyncVectorEnv", FakeVectorEnv):
            records = evaluate_parallel(
                checked_policy, episodes=7, seed=1000, workers=2, hidden_size=3
            )
        self.assertEqual([r["seed"] for r in records], list(range(1000, 1007)))
        self.assertEqual(sorted(calls), list(range(1000, 1007)))
        for record in records:
            length = record["seed"] % 3 + 1
            self.assertEqual(record["survival_steps"], length)
            self.assertEqual(record["score"], length)
            self.assertEqual(record["actions_left_right_wait"], [0, 0, length])

        # Recovery can schedule only unfinished, non-contiguous seeds, with the
        # reference threshold and measured death/timeout labels.
        callback_rows = []
        calls.clear()
        with patch("src.doom_evaluation.AsyncVectorEnv", FakeVectorEnv):
            recovered = evaluate_parallel(
                checked_policy,
                episodes=3,
                seed=0,
                workers=2,
                hidden_size=3,
                episode_seeds=[1004, 1010, 1033],
                action_threshold=0.3333,
                record_outcomes=True,
                on_episode=callback_rows.append,
            )
        self.assertEqual([r["seed"] for r in recovered], [1004, 1010, 1033])
        self.assertEqual(sorted(calls), [1004, 1010, 1033])
        self.assertEqual(sorted(callback_rows, key=lambda r: r["seed"]), recovered)
        self.assertTrue(all(r["terminated"] and not r["truncated"] for r in recovered))

    def test_selection_preserves_explicit_posterior_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            world = root / "world"
            world.mkdir()
            (world / "vae.eqx").write_bytes(b"vae")
            (world / "vae.eqx.json").write_text('{"architecture": "paper"}')
            (world / "rnn.eqx").write_bytes(b"rnn")
            (world / "rnn.eqx.json").write_text("{}")
            source = world / "controller_dream.npz"
            np.savez(source, params=np.zeros(4), posterior_sampling=False)
            hashes = {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (world / "vae.eqx", world / "rnn.eqx", source)
            }
            report = root / "validation.json"
            report.write_text(
                json.dumps(
                    {
                        "policy": "controller",
                        "episodes": 1,
                        "mean": 10,
                        "episodes_detail": [{"seed": 13000}],
                        "checkpoint_dir": str(world),
                        "checkpoint_sha256": hashes,
                        "inference_posterior_sampling": True,
                    }
                )
            )
            selected = root / "selected"
            with patch(
                "sys.argv",
                ["select", "--evaluations", str(report), "--output-dir", str(selected)],
            ):
                select_controller()
            with np.load(selected / "controller_dream.npz") as data:
                self.assertTrue(data["posterior_sampling"])
                self.assertFalse(data["training_posterior_sampling"])
            self.assertEqual(
                hashlib.sha256(source.read_bytes()).hexdigest(), hashes[source.name]
            )
            self.assertEqual(
                (selected / "vae.eqx.json").read_bytes(),
                (world / "vae.eqx.json").read_bytes(),
            )


if __name__ == "__main__":
    unittest.main()
