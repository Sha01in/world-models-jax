"""CPU checks for coherent feature gains without changing policy arithmetic."""

import unittest

import numpy as np

from src.doom_controller_gains import controller_from_log_gains


class TestControllerGains(unittest.TestCase):
    def setUp(self):
        self.initial = np.linspace(-2.0, 2.0, 1088, dtype=np.float64)

    def test_zero_point_preserves_raw_weights_and_signed_zero(self):
        self.initial[0:2] = (-0.0, 0.0)
        result = controller_from_log_gains(self.initial, np.zeros(3))
        self.assertEqual(result.tobytes(), self.initial.tobytes())
        self.assertFalse(np.shares_memory(result, self.initial))

    def test_block_order_and_original_dot_product(self):
        gains = np.asarray([0.5, 2.0, 1.25])
        result = controller_from_log_gains(self.initial, np.log(gains))
        blocks = (slice(0, 64), slice(64, 576), slice(576, 1088))
        features = np.random.default_rng(96).normal(size=1088)
        expected = 0.0
        for block, gain in zip(blocks, gains):
            np.testing.assert_allclose(result[block], self.initial[block] * gain)
            expected += gain * (features[block] @ self.initial[block])
        np.testing.assert_allclose(features @ result, expected, rtol=1e-14)
        self.assertEqual(result.dtype, np.float64)

    def test_gain_bounds_are_inclusive_and_invalid_search_points_rejected(self):
        for gain in (0.25, 4.0):
            result = controller_from_log_gains(self.initial, np.full(3, np.log(gain)))
            np.testing.assert_allclose(result, self.initial * gain)
        for point in (
            [0.0, 0.0],
            [0.0, np.nan, 0.0],
            [0.0, np.inf, 0.0],
            [2.0, 0.0, 0.0],
        ):
            with self.subTest(point=point), self.assertRaises(ValueError):
                controller_from_log_gains(self.initial, point)

    def test_invalid_initializer_and_projection_overflow_are_rejected(self):
        for weights in (
            self.initial.astype(np.float32),
            self.initial[:-1],
            np.full(1088, np.nan),
        ):
            with self.assertRaises(ValueError):
                controller_from_log_gains(weights, np.zeros(3))
        with self.assertRaisesRegex(ValueError, "non-finite weights"):
            controller_from_log_gains(np.full(1088, 1e308), np.full(3, np.log(4.0)))


if __name__ == "__main__":
    unittest.main()
