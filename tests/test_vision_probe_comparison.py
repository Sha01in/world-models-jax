import unittest

import numpy as np

from scripts.tools.compare_vision_probes import compare_predictions


class TestVisionProbeComparison(unittest.TestCase):
    def test_known_paired_gains_and_frame_mismatch_rejection(self):
        targets = np.array(
            [
                [0, 1, 0, 0],
                [1, 0, 20, 20],
                [0, 1, 0, 0],
                [1, 0, 30, 30],
                [1, 0, 40, 40],
                [0, 1, 0, 0],
            ],
            dtype=np.float64,
        )
        common = dict(
            episode_signatures=np.array(["a"] * 3 + ["b"] * 3),
            frame_indices=np.tile(np.arange(3), 2),
            targets=targets,
            split_sha256=np.array("same"),
        )
        reference = dict(
            common, presence_scores=-targets[:, :2], positions=targets[:, 2:] + [2, 1]
        )
        candidate = dict(
            common,
            presence_scores=targets[:, :2],
            positions=targets[:, 2:] + [0.5, 0.25],
        )
        result = compare_predictions(reference, candidate)
        self.assertEqual(result["episodes"], 2)
        self.assertEqual(result["point_estimates"][4:6], [1, 1])
        self.assertEqual(result["point_estimates"][10:12], [1.5, 0.75])
        self.assertEqual(result["intervals"][10:12], [[1.5, 1.5], [0.75, 0.75]])
        with self.assertRaisesRegex(ValueError, "identical held-out frames"):
            compare_predictions(reference, dict(candidate, frame_indices=np.arange(6)))
        duplicates = np.zeros(6, int)
        with self.assertRaisesRegex(ValueError, "duplicated"):
            compare_predictions(
                dict(reference, frame_indices=duplicates),
                dict(candidate, frame_indices=duplicates),
            )


if __name__ == "__main__":
    unittest.main()
