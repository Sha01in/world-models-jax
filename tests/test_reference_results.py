import copy
import unittest

import numpy as np

from src.doom_reference_results import checked_survival, paired_survival_summary


def report(scores, *, own=True):
    scores = np.asarray(scores, np.int64)
    protocol = dict(
        reference_commit="fixture",
        preprocessing="RGB24",
        difficulty=4,
        episode_start_time=14,
        episode_timeout=2100,
        threshold=0.3333,
        world_precision="FP32",
        posterior_and_controller_precision="FP64",
        rng="paired seeds",
        package_versions={"fixture": "1"},
        source_sha256={},
        input_sha256={
            name: name
            for name in ("vae.json", "rnn.json", "take_cover.wad", "freedoom2.wad")
        },
        evaluation_role="test",
        inference_posterior_sampling=True,
        limitations=["CPU fixture, not real-game evidence"],
        controller_provenance=dict(own_policy_update=own, imported_public_world=True),
    )
    return dict(
        protocol=protocol,
        episodes=len(scores),
        mean=float(scores.mean()),
        std=float(scores.std()),
        deaths=len(scores),
        timeouts=0,
        episodes_detail=[
            dict(
                seed=140000 + i,
                survival_steps=int(score),
                score=float(score),
                actions_left_right_wait=[0, 0, int(score)],
                terminated=True,
                truncated=False,
            )
            for i, score in enumerate(scores)
        ],
    )


class TestReferenceResults(unittest.TestCase):
    def test_constant_paired_gain_and_target_do_not_imply_from_scratch_world(self):
        baseline = np.arange(900, 1100, 2)
        selected, control = report(baseline + 150), report(baseline, own=False)
        summary = paired_survival_summary(
            selected, control, range(140000, 140100), resamples=128
        )
        self.assertEqual(summary["selected_mean"], 1149.0)
        self.assertEqual(summary["paired_gain"], 150.0)
        self.assertEqual(summary["paired_gain95"], [150.0, 150.0])
        self.assertTrue(summary["observed1092"] and summary["own_policy_update"])
        self.assertTrue(summary["imported_public_world"])
        self.assertEqual(summary["wins"], 100)
        identical = paired_survival_summary(
            control, control, range(140000, 140100), resamples=128
        )
        self.assertEqual(identical["paired_gain95"], [0.0, 0.0])
        self.assertFalse(identical["own_policy_update"])

    def test_missing_games_wrong_rewards_and_protocol_changes_are_rejected(self):
        valid = report(np.full(100, 1200))
        for change in ("missing", "reward", "summary", "outcomes"):
            bad = copy.deepcopy(valid)
            if change == "missing":
                bad["episodes_detail"].pop()
            elif change == "reward":
                bad["episodes_detail"][3]["score"] = 1199
            elif change == "summary":
                bad["std"] = 100
            else:
                bad["episodes_detail"][3]["truncated"] = True
            with self.assertRaises(ValueError):
                checked_survival(bad, range(140000, 140100))
        for change in ("difficulty", "weights"):
            different = copy.deepcopy(valid)
            if change == "difficulty":
                different["protocol"]["difficulty"] = 5
            else:
                different["protocol"]["input_sha256"]["rnn.json"] = "different"
            with self.assertRaises(ValueError):
                paired_survival_summary(
                    valid, different, range(140000, 140100), resamples=16
                )


if __name__ == "__main__":
    unittest.main()
