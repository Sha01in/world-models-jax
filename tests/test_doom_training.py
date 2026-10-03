import unittest

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from src.controller import canonical_doom_action, controller_memory
from src.dream import make_dream_engine, sample_mdn, unweighted_death_probability
from src.config import get_config
from src.rnn import MDNRNN
from train_rnn import transition_targets, loss_fn, sequence_predictions
from train_rnn_packed import pack_episodes, packed_batches


class TestDoomTraining(unittest.TestCase):
    def test_stream_memory_carries_and_resets_before_episode_input(self):
        class CounterModel:
            def __call__(self, x, hidden):
                h, c = (state + 1 for state in hidden)
                return (
                    jnp.zeros((1, 1)),
                    jnp.zeros((1, 1)),
                    jnp.zeros((1, 1)),
                    h,
                    c,
                ), (h, c)

        hidden = (jnp.array([[10.0]]), jnp.array([[20.0]]))
        final, predictions = sequence_predictions(
            CounterModel(),
            jnp.zeros((1, 4, 2)),
            hidden,
            jnp.array([[False, False, True, False]]),
        )
        np.testing.assert_array_equal(predictions[3][:, 0, 0], [11, 12, 1, 2])
        np.testing.assert_array_equal(predictions[4][:, 0, 0], [21, 22, 1, 2])
        np.testing.assert_array_equal(final[0], [[2]])

    def test_packed_targets_preserve_fatal_actions_and_censored_resets(self):
        def episode(values, death):
            mu = np.asarray(values, np.float16)[:, None]
            done = np.zeros(len(mu), np.uint8)
            done[-1] = death
            return mu, np.zeros_like(mu), np.ones((len(mu), 1), np.int8), done

        packed = pack_episodes(
            [
                episode([10, 20, 30], 1),
                episode([40, 50, 60], 0),
                episode([70, 80, 90, 100], 1),
            ],
            [0, 1, 2],
        )
        first, second = list(
            packed_batches(packed, 1, 4, np.random.default_rng(3), False)
        )
        np.testing.assert_array_equal(first[3], [[0, 0, 1, 0]])
        np.testing.assert_array_equal(first[5], [[1, 1, 0, 1]])
        np.testing.assert_array_equal(second[0][0, :, 0], [50, 60, 70, 80])
        np.testing.assert_array_equal(second[3], [[0, 0, 0, 0]])
        np.testing.assert_array_equal(second[5], [[1, 0, 1, 1]])
        np.testing.assert_array_equal(second[6], [[False, False, True, False]])
        np.testing.assert_array_equal(first[1][0, -1], second[0][0, 0, :1])
        # A second stream beginning inside an episode gets no loss until reset.
        batches = list(packed_batches(packed, 2, 2, np.random.default_rng(3)))
        np.testing.assert_array_equal(batches[0][4][1], [0, 0])
        np.testing.assert_array_equal(batches[1][4][1], [1, 1])
        np.testing.assert_array_equal(batches[0][1][:, -1], batches[1][0][:, 0, :1])

    def test_sampled_death_undoes_class_weighting(self):
        actual = jnp.array([0.001, 0.01, 0.1, 0.5, 0.9])
        weighted_logits = jnp.log(actual / (1.0 - actual)) + jnp.log(10.0)
        np.testing.assert_allclose(
            unweighted_death_probability(weighted_logits, 10.0), actual, rtol=1e-6
        )

    def test_transition_outcomes_stay_with_their_action(self):
        z = np.array([[[10.0], [20.0], [30.0], [0.0]]], np.float32)
        action = np.array([[[-1.0], [0.0], [1.0], [0.0]]], np.float32)
        rewards = np.array([[1.0, 2.0, 3.0, 0.0]], np.float32)
        dones = np.array([[0.0, 0.0, 1.0, 0.0]], np.float32)
        mask = np.array([[1.0, 1.0, 1.0, 0.0]], np.float32)
        inputs, targets, r, d, valid, latent_valid = transition_targets(
            z, action, rewards, dones, mask
        )
        np.testing.assert_array_equal(inputs[0, 2], [30.0, 1.0])
        np.testing.assert_array_equal(r, rewards)
        np.testing.assert_array_equal(d, dones)
        np.testing.assert_array_equal(valid, mask)
        np.testing.assert_array_equal(latent_valid, [[1.0, 1.0, 0.0, 0.0]])
        np.testing.assert_array_equal(targets[0, :2, 0], [20.0, 30.0])

    def test_temperature_scales_variance(self):
        pi = jnp.zeros((1, 1))
        mu = jnp.zeros((1, 32))
        logsigma = jnp.zeros_like(mu)
        key = jax.random.PRNGKey(7)
        at_one = sample_mdn(pi, mu, logsigma, key, 1.0)
        at_four = sample_mdn(pi, mu, logsigma, key, 4.0)
        np.testing.assert_allclose(at_four, at_one * 2, rtol=1e-6)

    def test_factorized_mixture_and_rare_death_weighting(self):
        model = MDNRNN(2, 1, 4, key=jax.random.PRNGKey(0), factorized=True)
        model = eqx.tree_at(
            lambda m: (m.done_head.weight, m.done_head.bias),
            model,
            (
                jnp.zeros_like(model.done_head.weight),
                jnp.zeros_like(model.done_head.bias),
            ),
        )
        pi, _, _, _, _ = model(jnp.zeros(3), model.init_state())[0]
        np.testing.assert_allclose(np.exp(pi).sum(axis=0), np.ones(2), rtol=1e-6)
        x = jnp.zeros((1, 2, 3))
        z = jnp.zeros((1, 2, 2))
        r = jnp.ones((1, 2))
        d = jnp.array([[0.0, 1.0]])
        mask = jnp.ones((1, 2))
        _, (_, _, death) = loss_fn(
            model, x, z, r, d, mask, jax.random.PRNGKey(0), mask, 10.0, True
        )
        np.testing.assert_allclose(death, 5.5 * np.log(2), rtol=1e-5)

    def test_controller_uses_both_lstm_states_and_physical_actions(self):
        np.testing.assert_array_equal(
            controller_memory((jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0])), "hc"),
            [3.0, 4.0, 1.0, 2.0],
        )
        np.testing.assert_array_equal(
            canonical_doom_action(jnp.array([-0.9, -0.2, 0.2, 0.9])),
            [-1.0, 0.0, 0.0, 1.0],
        )

    def test_doom_dream_counts_terminal_step_and_ignores_reward_head(self):
        class FatalModel:
            def __call__(self, x, hidden):
                return (
                    jnp.zeros((1, 1)),
                    jnp.zeros((1, 64)),
                    jnp.zeros((1, 64)),
                    jnp.array([1000.0]),
                    jnp.array([10.0]),
                ), hidden

        cfg = get_config("VizdoomTakeCover-v0")
        engine = make_dream_engine(
            FatalModel(), lambda p, z, h, a: jnp.zeros(1), cfg, 20
        )
        scores = engine(
            jnp.zeros((2, 1)),
            jnp.zeros((2, 64)),
            jax.random.split(jax.random.PRNGKey(0), 2),
            1.15,
        )
        np.testing.assert_array_equal(scores, [1.0, 1.0])


if __name__ == "__main__":
    unittest.main()
