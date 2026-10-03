"""Latent simulation shared by controller evolution and diagnostics."""

import jax
import jax.numpy as jnp

from src.controller import canonical_doom_action, controller_memory


def sample_mdn(log_pi, mu, log_sigma, key, temperature):
    """Reference temperature: soften mixture logits and scale variance by tau."""
    mixture_key, noise_key = jax.random.split(key)
    if log_pi.shape[-1] == 1:
        k = jax.random.categorical(mixture_key, log_pi[:, 0] / temperature)
        chosen_mu, chosen_log_sigma = mu[k], log_sigma[k]
    else:
        k = jax.random.categorical(mixture_key, log_pi.T / temperature, axis=-1)
        axes = jnp.arange(mu.shape[-1])
        chosen_mu, chosen_log_sigma = mu[k, axes], log_sigma[k, axes]
    noise = jax.random.normal(noise_key, chosen_mu.shape)
    return chosen_mu + jnp.exp(chosen_log_sigma) * jnp.sqrt(temperature) * noise


def unweighted_death_probability(logit, positive_weight):
    """Undo the odds multiplier introduced by positive-class weighted BCE."""
    return jax.nn.sigmoid(logit - jnp.log(positive_weight))


def make_dream_engine(
    rnn,
    controller_fn,
    config,
    dream_length,
    state_mode="h",
    canonical_actions=False,
    done_mode="threshold",
    done_positive_weight=1.0,
):
    @jax.jit
    def run(params_batch, start_z, keys, temperature):
        zeros = jnp.zeros((params_batch.shape[0], config.hidden_size))

        def step(carry, _):
            z, h, c, active, total, current_keys = carry
            memory = controller_memory((h, c), state_mode)
            action = jax.vmap(controller_fn, in_axes=(0, 0, 0, None))(
                params_batch, z, memory, config.action_dim
            )
            model_action = (
                canonical_doom_action(action) if canonical_actions else action
            )
            (pi, mu, log_sigma, reward, done), (next_h, next_c) = jax.vmap(rnn)(
                jnp.concatenate([z, model_action], axis=-1), (h, c)
            )
            split = jax.vmap(lambda k: jax.random.split(k, 3))(current_keys)
            next_z = jax.vmap(sample_mdn, in_axes=(0, 0, 0, 0, None))(
                pi, mu, log_sigma, split[:, 0], temperature
            )
            if done_mode == "sampled":
                dead = jax.vmap(jax.random.bernoulli)(
                    split[:, 1],
                    unweighted_death_probability(done[:, 0], done_positive_weight),
                )
            else:
                dead = done[:, 0] >= 0
            # The reference counts the terminal transition too, as the real game does.
            step_reward = jnp.ones_like(total) if config.is_doom else reward[:, 0]
            total = total + active * step_reward
            active = active * (~dead)
            return (next_z, next_h, next_c, active, total, split[:, 2]), None

        initial = (
            start_z,
            zeros,
            zeros,
            jnp.ones(params_batch.shape[0]),
            jnp.zeros(params_batch.shape[0]),
            keys,
        )

        # Evolution needs no rollout gradients, so stop once this batch is dead.
        def condition(state):
            t, carry = state
            return (t < dream_length) & jnp.any(carry[3] > 0)

        def body(state):
            t, carry = state
            return t + 1, step(carry, None)[0]

        _, carry = jax.lax.while_loop(condition, body, (0, initial))
        return carry[4]

    return run
