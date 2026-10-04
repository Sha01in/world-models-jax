"""Numeric inference, mixed-game scheduling and interrupted CMA recovery."""

import hashlib
import json
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np

from scripts.tools.train_doom_reference_real import run_search
from scripts.tools.audit_doom_reference_real_training import audit_search_capsule
from src.doom_real_training import (
    PopulationEvaluator,
    checked_population_records,
    make_population_policy,
)
from src.doom_reference import make_reference_policy


class ToyVAE:
    def encode(self, image):
        return jnp.repeat(image.mean(), 64), jnp.linspace(-3, -1, 64)


class ToyRNN:
    def __call__(self, inputs, hidden, restart):
        h, c = (state * (1 - restart) for state in hidden)
        next_c = c + inputs[:1] + inputs[-1:]
        return None, (h + jnp.tanh(next_c), next_c)


class FakeVectorEnv:
    def __init__(self, factories, **kwargs):
        self.n = len(factories)
        self.seeds = np.zeros(self.n, np.int64)
        self.steps = np.zeros(self.n, np.int64)
        self.closed = False

    def reset(self, seed, options=None):
        mask = np.ones(self.n, bool) if options is None else options["reset_mask"]
        for slot in np.flatnonzero(mask):
            self.seeds[slot] = seed[slot]
            self.steps[slot] = 0
        return self.seeds[:, None].copy(), {}

    def step(self, actions):
        self.steps += 1
        dead = self.steps >= self.seeds % 3 + 1
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


class SyntheticFitness:
    def __init__(self, *, interrupt_fitness=False):
        self.interrupt_fitness = interrupt_fitness

    def evaluate(self, parameters, seeds, *, completed, on_episode):
        rows = list(completed)
        done = {(row["candidate"], row["seed"]) for row in rows}
        for candidate, params in enumerate(parameters):
            for seed in seeds:
                if (candidate, seed) in done:
                    continue
                score = int(np.clip(900 + params[0] * 10000 + seed % 7, 1, 2100))
                row = dict(
                    candidate=candidate,
                    seed=seed,
                    survival_steps=score,
                    score=float(score),
                    actions_left_right_wait=[0, 0, score],
                    terminated=score < 2100,
                    truncated=score == 2100,
                )
                on_episode(row)
                rows.append(row)
                if self.interrupt_fitness and len(parameters) > 1 and len(rows) == 3:
                    raise KeyboardInterrupt(
                        "Synthetic interruption after three real records"
                    )
        return checked_population_records(rows, len(parameters), seeds, complete=True)


def fixture_settings(root, output):
    protocol = root / "protocol.json"
    protocol.write_text("{}\n")
    return dict(
        arguments=dict(
            output=str(output),
            generations=2,
            pop_size=4,
            sigma=0.005,
            seed=94,
            fitness_seed=500000,
            fitness_games=2,
            holdout_seed=510000,
            holdout_games=3,
            validate_every=1,
        ),
        protocol=str(protocol),
        protocol_sha256=hashlib.sha256(protocol.read_bytes()).hexdigest(),
        input_sha256={},
        source_sha256={},
    )


class TestRealTraining(unittest.TestCase):
    def test_dynamic_weights_preserve_original_actor_and_rng(self):
        vae, rnn = ToyVAE(), ToyRNN()
        dynamic = make_population_policy(vae, rnn)
        rng = np.random.default_rng(71)
        for workers in (4, 8):
            images = rng.integers(0, 256, (workers, 64, 64, 3), dtype=np.uint8)
            hidden = tuple(
                jnp.asarray(rng.normal(size=(workers, 512)), dtype=jnp.float32)
                for _ in range(2)
            )
            keys = jnp.stack(
                [jax.random.PRNGKey(seed) for seed in range(300, 300 + workers)]
            )
            steps = jnp.arange(workers, dtype=jnp.int32)
            controllers = jnp.asarray(
                rng.normal(0, 0.001, (workers, 1088)), dtype=jnp.float64
            )
            actual = dynamic(images, hidden, keys, steps, controllers)
            for slot in range(workers):
                original = make_reference_policy(
                    vae, rnn, controllers[slot], posterior=True
                )
                expected = original(
                    images[slot : slot + 1],
                    tuple(h[slot : slot + 1] for h in hidden),
                    keys[slot : slot + 1],
                    steps[slot : slot + 1],
                )
                for a, b in zip(
                    jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
                ):
                    np.testing.assert_array_equal(np.asarray(a)[slot], np.asarray(b)[0])
            compiled = dynamic._cache_size()
            dynamic(images, hidden, keys, steps, controllers + 0.001)
            self.assertEqual(dynamic._cache_size(), compiled)

    def test_mixed_candidates_reset_memory_rng_and_recover_exact_pairs(self):
        starts = []

        def policy(obs, memory, keys, steps, controllers):
            for slot in range(len(obs)):
                if steps[slot] == 0:
                    np.testing.assert_array_equal(np.asarray(memory[0])[slot], 0)
                    np.testing.assert_array_equal(np.asarray(memory[1])[slot], 0)
                    self.assertEqual(int(keys[slot, 1]), int(obs[slot, 0]))
                    starts.append((float(controllers[slot, 0]), int(obs[slot, 0])))
            return (
                controllers[:, :1],
                tuple(h + 1 for h in memory),
                keys + jnp.array([0, 1], dtype=jnp.uint32),
            )

        params = np.zeros((3, 1088))
        params[:, 0] = [-0.5, 0.5, 0]
        seeds = list(range(120, 127))
        with patch("src.doom_real_training.AsyncVectorEnv", FakeVectorEnv):
            with PopulationEvaluator(
                policy, lambda: None, workers=4, hidden_size=3
            ) as evaluator:
                rows = evaluator.evaluate(params, seeds)
                self.assertEqual(len(rows), 21)
                recovered = evaluator.evaluate(params, seeds, completed=rows[:5])
                self.assertEqual(rows, recovered)
                # The same workers are reused, then closed by the context manager.
                self.assertFalse(evaluator.envs.closed)
            self.assertTrue(evaluator.envs.closed)
        for row in rows:
            expected_counts = [0, 0, 0]
            expected_counts[row["candidate"]] = row["survival_steps"]
            self.assertEqual(row["actions_left_right_wait"], expected_counts)
        self.assertTrue(set((-0.5, seed) for seed in seeds) <= set(starts))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            checked_population_records([rows[0], rows[0]], 3, seeds)
        with self.assertRaisesRegex(ValueError, "outcome"):
            checked_population_records([dict(rows[0], truncated=True)], 3, seeds)

    def test_interrupted_population_matches_uninterrupted_cma_and_next_rng(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "first.npz"
            settings = fixture_settings(root, first)
            run_search(settings, SyntheticFitness(), np.zeros(1088))
            audit = audit_search_capsule(first, np.zeros(1088))
            self.assertEqual(audit["raw_training_games"], 25)
            self.assertTrue(
                audit["population_weights_optimizer_history_and_next_rng_reconstructed"]
            )
            second = root / "second.npz"
            recovered_settings = fixture_settings(root, second)
            with self.assertRaises(KeyboardInterrupt):
                run_search(
                    recovered_settings,
                    SyntheticFitness(interrupt_fitness=True),
                    np.zeros(1088),
                )
            pointer = json.loads(Path(str(second) + ".resume.json").read_text())
            self.assertEqual(pointer["phase"], "fitness")
            self.assertEqual(pointer["generation"], 0)
            partial_report = Path(
                str(second) + ".search/games/g001_fitness.partial.json"
            )
            self.assertEqual(
                len(json.loads(partial_report.read_text())["episodes_detail"]), 3
            )
            run_search(
                recovered_settings, SyntheticFitness(), np.zeros(1088), resume=True
            )
            self.assertEqual(
                audit_search_capsule(second, np.zeros(1088))["best_generation"], 2
            )
            for suffix in ("", ".last"):
                a = first if not suffix else root / "first.last.npz"
                b = second if not suffix else root / "second.last.npz"
                with np.load(a) as original, np.load(b) as recovered:
                    np.testing.assert_array_equal(
                        original["params"], recovered["params"]
                    )
                    self.assertEqual(
                        int(original["generation"]), int(recovered["generation"])
                    )
            populations = []
            for output in (first, second):
                pointer = json.loads(Path(str(output) + ".resume.json").read_text())
                self.assertTrue(pointer["complete"])
                state_path = Path(pointer["state_path"])
                self.assertEqual(
                    hashlib.sha256(state_path.read_bytes()).hexdigest(),
                    pointer["state_sha256"],
                )
                state = pickle.loads(state_path.read_bytes())
                self.assertEqual(state["optimizer"].countiter, 2)
                np.random.set_state(state["numpy_random_state"])
                populations.append(state["optimizer"].ask())
            np.testing.assert_array_equal(*populations)
            with self.assertRaisesRegex(ValueError, "Completed"):
                run_search(
                    recovered_settings, SyntheticFitness(), np.zeros(1088), resume=True
                )
            with self.assertRaises(FileExistsError):
                run_search(settings, SyntheticFitness(), np.zeros(1088))

    def test_recovery_rejects_changed_cohort_or_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "policy.npz"
            settings = fixture_settings(root, output)
            with self.assertRaises(KeyboardInterrupt):
                run_search(
                    settings, SyntheticFitness(interrupt_fitness=True), np.zeros(1088)
                )
            partial_report = Path(
                str(output) + ".search/games/g001_fitness.partial.json"
            )
            saved = json.loads(partial_report.read_text())
            saved["parameters_sha256"][0] = "changed"
            partial_report.write_text(json.dumps(saved))
            with self.assertRaisesRegex(ValueError, "different inputs"):
                run_search(settings, SyntheticFitness(), np.zeros(1088), resume=True)
            pointer = json.loads(Path(str(output) + ".resume.json").read_text())
            Path(pointer["state_path"]).write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                run_search(settings, SyntheticFitness(), np.zeros(1088), resume=True)


if __name__ == "__main__":
    unittest.main()
