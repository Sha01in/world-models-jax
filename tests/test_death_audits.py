"""Verify posterior precision and paired death uncertainty on fixed episodes."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.tools.audit_doom import audit_latents, death_counts
from scripts.tools.compare_death_audits import compare
from train_rnn import load_batch


class TestDeathAudits(unittest.TestCase):
    def test_audit_posterior_exactly_matches_training_and_avoids_float16_rounding(self):
        with tempfile.TemporaryDirectory() as temporary:
            data = dict(
                mu=np.full((3, 64), 10, np.float16),
                logvar=np.full((3, 64), -14, np.float16),
                actions=np.zeros((3, 1)),
                rewards=np.ones(3),
                dones=np.array([0, 0, 1]),
            )
            path = Path(temporary) / "episode.npz"
            np.savez(path, **data)
            expected = load_batch(
                [path], sample_posterior=True, rng=np.random.default_rng(74)
            )[0][0, :3]
            actual = audit_latents(data, np.random.default_rng(74), True)
            np.testing.assert_array_equal(actual, expected)
            self.assertEqual(actual.dtype, np.float32)
            legacy = audit_latents(data, np.random.default_rng(74), True, True)
            self.assertFalse(np.array_equal(actual, legacy))
            np.testing.assert_array_equal(audit_latents(data, None, False), data["mu"])

    def test_paired_counts_and_uncertainty_reject_mismatched_labels(self):
        protocol = dict(
            posterior_input_protocol="train_rnn_float32_inputs_and_noise",
            mean_latents_override=False,
            inference_posterior_sampling=True,
            seed=42,
            inference_batch_size=16,
        )
        reference = dict(protocol, episode_records=[])
        candidate = dict(protocol, episode_records=[])
        for i in range(3):
            for report, scores in (
                (reference, [0.6, 0.1, 0.1]),
                (candidate, [0.1, 0.1, 0.8]),
            ):
                report["episode_records"].append(
                    dict(
                        transition_signature=str(i),
                        frames=3,
                        counts=death_counts(scores, [0, 0, 1]),
                    )
                )
        paired = compare(reference, candidate, resamples=100)
        np.testing.assert_array_equal(paired["point_estimates"], [0, 0.5, 1, 0, 1, 0.5])
        np.testing.assert_array_equal(paired["intervals"][4:], [[1, 1], [0.5, 0.5]])
        self.assertEqual(paired["frames"], 9)
        changed = json.loads(json.dumps(candidate))
        changed["episode_records"][0]["counts"]["live_transitions"] += 1
        with self.assertRaisesRegex(ValueError, "labels or lengths"):
            compare(reference, changed)
        changed = json.loads(json.dumps(candidate))
        changed["episode_records"].append(changed["episode_records"][0])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            compare(reference, changed)


if __name__ == "__main__":
    unittest.main()
