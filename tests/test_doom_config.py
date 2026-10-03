import unittest
import jax
import jax.numpy as jnp
import numpy as np
from unittest.mock import MagicMock

from src.config import get_config
from src.env_utils import VizDoomEnv
from src.vae import VAE
from src.rnn import MDNRNN
from src.controller import get_action


class TestDoomConfig(unittest.TestCase):
    def test_config_values(self):
        config = get_config("VizdoomTakeCover-v0")
        self.assertTrue(config.is_doom)
        self.assertEqual(config.latent_dim, 64)
        self.assertEqual(config.hidden_size, 512)
        self.assertEqual(config.action_dim, 1)

    def test_vae_shapes(self):
        config = get_config("VizdoomTakeCover-v0")
        model = VAE(latent_dim=config.latent_dim, key=jax.random.PRNGKey(0))

        # Test input shape (Batch, 3, 64, 64)
        # VAE is designed for single sample (3, 64, 64), so we vmap it for batch
        dummy_input = jnp.zeros((1, 3, 64, 64))

        # vmap the model call
        # model(x, key) -> (recon, mu, logvar)
        # We need to vmap over x (axis 0) and key (axis 0) if key is provided,
        # but here key is None in __call__ signature default, but used inside?
        # Actually VAE.__call__ takes (x, key).

        key = jax.random.PRNGKey(0)
        keys = jax.random.split(key, 1)

        recon, mu, logvar = jax.vmap(model)(dummy_input, keys)

        self.assertEqual(recon.shape, (1, 3, 64, 64))
        self.assertEqual(mu.shape, (1, 64))
        self.assertEqual(logvar.shape, (1, 64))

    def test_rnn_shapes(self):
        config = get_config("VizdoomTakeCover-v0")
        model = MDNRNN(
            latent_dim=config.latent_dim,
            action_dim=config.action_dim,
            hidden_size=config.hidden_size,
            key=jax.random.PRNGKey(0),
        )

        # Test input: z + action
        dummy_z = jnp.zeros((1, config.latent_dim))
        dummy_a = jnp.zeros((1, config.action_dim))
        dummy_input = jnp.concatenate([dummy_z, dummy_a], axis=1)

        hidden = model.init_state()
        # Expand hidden to batch size 1
        hidden = (jnp.expand_dims(hidden[0], 0), jnp.expand_dims(hidden[1], 0))

        # MDN-RNN is also single-step, so vmap for batch
        (log_pi, mu, log_sigma, reward, done), _ = jax.vmap(model)(dummy_input, hidden)

        self.assertEqual(
            mu.shape, (1, 5, config.latent_dim)
        )  # Batch, 5 Gaussians, Latent
        self.assertEqual(reward.shape, (1, 1))
        self.assertEqual(done.shape, (1, 1))

    def test_controller_shapes(self):
        config = get_config("VizdoomTakeCover-v0")
        input_dim = config.latent_dim + config.hidden_size
        output_dim = config.action_dim

        # Create dummy params
        num_params = (input_dim * output_dim) + output_dim
        params = jnp.zeros(num_params)

        z = jnp.zeros(config.latent_dim)
        h = jnp.zeros(config.hidden_size)

        action = get_action(params, z, h, action_dim=config.action_dim)

        self.assertEqual(action.shape, (config.action_dim,))

    def test_doom_wrapper(self):
        wrapper = VizDoomEnv.__new__(VizDoomEnv)
        wrapper.img_size = 64
        wrapper.max_episode_steps = 2100
        wrapper.game = MagicMock()
        wrapper.game.is_episode_finished.return_value = False
        wrapper.game.is_player_dead.return_value = False
        wrapper.game.get_state.return_value.screen_buffer = np.zeros(
            (100, 100, 3), np.uint8
        )

        # Test Reset
        obs, _ = wrapper.reset(seed=123)
        self.assertEqual(obs.shape, (64, 64, 3))
        wrapper.game.set_seed.assert_called_once_with(123)

        # Test Step with continuous action
        action = np.array([0.5])  # Should map to Right (1)
        obs, _, _, _, _ = wrapper.step(action)

        # Verify mock called with discrete action
        # Logic: val > 0.3 -> 1
        wrapper.game.make_action.assert_called_with([0, 1])

        action = np.array([-0.5])  # Should map to Left (0)
        wrapper.step(action)
        wrapper.game.make_action.assert_called_with([1, 0])
        wrapper.step(np.array([0.0]))
        wrapper.game.make_action.assert_called_with([0, 0])

        wrapper.game.is_episode_finished.return_value = True
        _, _, terminated, truncated, _ = wrapper.step(np.array([0.0]))
        self.assertFalse(terminated)
        self.assertTrue(truncated)


if __name__ == "__main__":
    unittest.main()
