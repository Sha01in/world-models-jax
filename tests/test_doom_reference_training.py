import unittest

import jax
import jax.numpy as jnp
import numpy as np

from src.doom_reference_training import (
    make_reference_dream_engine,
    reference_start_pool,
)


class TestReferenceDream(unittest.TestCase):
    def test_start_quantization_and_invalid_shape(self):
        payload = {"initial_z.json": np.full((2, 3, 64), 2500, np.int32)}
        means, logvars = reference_start_pool(payload)
        np.testing.assert_array_equal(means, np.full((3, 64), 0.25))
        np.testing.assert_array_equal(logvars, means)
        with self.assertRaises(ValueError):
            reference_start_pool({"initial_z.json": np.zeros((2, 3, 63))})
        payload["initial_z.json"] = np.full((2, 3, 64), np.nan)
        with self.assertRaises(ValueError):
            reference_start_pool(payload)

    def test_rollouts_match_independent_scalar_state_and_raw_action_oracle(self):
        class TinyWorld:
            hidden_size = 1

            def __call__(self, inputs, hidden, restart):
                h, c = (jnp.where(restart > 0.5, 0, value) for value in hidden)
                # Almost deterministic MDN: the input latent increments by1/8.
                # Death depends on the raw policy action, not physical buttons.
                death = jnp.where((inputs[-1] > 0.3) & (h[0] >= 1), 1.0, -1.0)
                return (
                    jnp.zeros((1, 1)),
                    (inputs[:1] + 0.125).reshape(1, 1),
                    jnp.full((1, 1), -100.0),
                    jnp.zeros(1),
                    death.reshape(1),
                ), (h + 1, c + 2)

        params = np.array([[0.2, 0.5, -0.3], [-0.4, 0.075, 0.1]], np.float32)
        starts = np.array([[-1.0], [0.25]], np.float32)
        expected = []
        for weights, start in zip(params, starts):
            z, h, c = float(start[0]), 0.0, 0.0
            for tick in range(1, 8):
                raw_action = np.tanh(np.dot([z, c, h], weights))
                dead = raw_action > 0.3 and h >= 1
                z, h, c = z + 0.125, h + 1, c + 2
                if dead:
                    break
            expected.append(tick)
        self.assertEqual(expected, [2, 4])
        keys = jax.random.split(jax.random.PRNGKey(65), 2)
        actual = make_reference_dream_engine(TinyWorld(), 7)(
            jnp.asarray(params), jnp.asarray(starts), keys, 1.15
        )
        np.testing.assert_array_equal(actual, expected)

    def test_zero_death_logit_is_alive_and_cap_counts_actions(self):
        class ZeroLogitWorld:
            hidden_size = 1

            def __call__(self, inputs, hidden, restart):
                return (
                    jnp.zeros((1, 1)),
                    jnp.zeros((1, 1)),
                    jnp.full((1, 1), -100.0),
                    jnp.zeros(1),
                    jnp.zeros(1),
                ), hidden

        actual = make_reference_dream_engine(ZeroLogitWorld(), 5)(
            jnp.zeros((1, 3)),
            jnp.zeros((1, 1)),
            jax.random.split(jax.random.PRNGKey(19), 1),
            1.15,
        )
        np.testing.assert_array_equal(actual, [5])
        with self.assertRaises(ValueError):
            make_reference_dream_engine(ZeroLogitWorld(), 0)

        invalid = make_reference_dream_engine(ZeroLogitWorld(), 5)(
            jnp.full((1, 3), jnp.nan),
            jnp.zeros((1, 1)),
            jax.random.split(jax.random.PRNGKey(19), 1),
            1.15,
        )
        np.testing.assert_array_equal(invalid, [-1])


if __name__ == "__main__":
    unittest.main()
