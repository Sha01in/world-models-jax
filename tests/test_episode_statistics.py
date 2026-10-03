import unittest

import numpy as np

from src.episode_statistics import episode_bootstrap


class TestEpisodeBootstrap(unittest.TestCase):
    def test_replicating_correlated_frames_does_not_shrink_interval(self):
        identities = np.array([0, 0, 1, 1, 2, 2, 3, 3])
        values = np.array([0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0])
        first = episode_bootstrap(
            identities, lambda rows: np.mean(values[rows]), resamples=1000, seed=1
        )
        repeated_ids, repeated_values = np.repeat(identities, 20), np.repeat(values, 20)
        second = episode_bootstrap(
            repeated_ids,
            lambda rows: np.mean(repeated_values[rows]),
            resamples=1000,
            seed=1,
        )
        np.testing.assert_array_equal(first["intervals"], second["intervals"])
        self.assertEqual(first["episodes"], second["episodes"])
        self.assertGreater(second["frames"], first["frames"])

    def test_sampling_keeps_every_frame_in_each_selected_episode(self):
        identities = np.array([0, 0, 1, 1, 1, 2])

        def checked(rows):
            counts = np.bincount(rows, minlength=len(identities))
            for identity in np.unique(identities):
                group = counts[identities == identity]
                self.assertTrue(np.all(group == group[0]))
            return len(rows)

        result = episode_bootstrap(identities, checked, resamples=100, seed=3)
        self.assertEqual(result["valid_resamples"], [100])

    def test_paired_arrays_use_the_same_resampled_rows(self):
        identities = np.repeat(np.arange(5), 3)
        baseline = np.arange(15, dtype=float) ** 2
        candidate = baseline + 7
        result = episode_bootstrap(
            identities,
            lambda rows: np.mean(candidate[rows] - baseline[rows]),
            resamples=100,
            seed=4,
        )
        self.assertEqual(result["point_estimates"], [7])
        self.assertEqual(result["intervals"], [[7, 7]])

    def test_missing_statistic_is_reported_without_nan_json(self):
        result = episode_bootstrap([0, 1], lambda rows: [np.nan, 1], resamples=5)
        self.assertEqual(result["point_estimates"], [None, 1])
        self.assertEqual(result["intervals"], [None, [1, 1]])
        self.assertEqual(result["valid_resamples"], [0, 5])


if __name__ == "__main__":
    unittest.main()
