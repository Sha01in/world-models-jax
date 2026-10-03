"""Controller rollouts in the imported reference world, without real-game tuning."""

import jax
import jax.numpy as jnp
import numpy as np

from src.dream import sample_mdn


def reference_start_pool(payload):
    """Decode the author's quantized distribution of actual episode starts."""
    starts = np.asarray(payload["initial_z.json"], dtype=np.float64) / 10000.0
    if (
        starts.ndim != 3
        or starts.shape[0] != 2
        or starts.shape[1] < 2
        or starts.shape[2] != 64
        or not np.isfinite(starts).all()
    ):
        raise ValueError("Expected paired finite reference start means/logvars")
    return starts[0], starts[1]


def make_reference_dream_engine(rnn, dream_length=2100):
    """Port the source's z,c,h controller, raw action, restart and death timing.

    With x64 enabled and FP64 starts/weights, latent samples and control use
    FP64, while the learned recurrent computation remains FP32. JAX random
    streams differ from the original NumPy streams. A death logit strictly
    greater than zero ends the dream; the terminal transition earns one point.
    An invalid active trajectory returns -1 rather than rewarding numerical
    overflow as apparent survival.
    """
    if dream_length < 1:
        raise ValueError("Positive dream length required")

    @jax.jit
    def run(params, start_z, keys, temperature):
        batch = params.shape[0]
        zero = jnp.zeros((batch, rnn.hidden_size), dtype=jnp.float32)

        def condition(state):
            t, _, _, _, active, _, _ = state
            return (t < dream_length) & jnp.any(active)

        def step(state):
            t, z, h, c, active, total, current_keys = state
            action = jnp.tanh(jnp.sum(jnp.concatenate([z, c, h], -1) * params, -1))
            inputs = jnp.concatenate([z, action[:, None]], -1).astype(jnp.float32)
            restart = jnp.full(batch, t == 0, dtype=jnp.float32)
            predictions, (next_h, next_c) = jax.vmap(rnn)(inputs, (h, c), restart)
            pi, mu, log_sigma, _, death = predictions
            split = jax.vmap(jax.random.split)(current_keys)
            next_z = jax.vmap(sample_mdn, in_axes=(0, 0, 0, 0, None))(
                pi, mu, log_sigma, split[:, 0], temperature
            ).astype(start_z.dtype)
            finite = jnp.isfinite(action)
            for values in (z, pi, mu, log_sigma, death, next_z, next_h, next_c):
                finite &= jnp.all(jnp.isfinite(values).reshape(batch, -1), axis=-1)
            return (
                t + 1,
                next_z,
                next_h,
                next_c,
                active & finite & (death[:, 0] <= 0),
                jnp.where(active & ~finite, -1, total + active.astype(jnp.int32)),
                split[:, 1],
            )

        initial = (
            jnp.int32(0),
            start_z,
            zero,
            zero,
            jnp.ones(batch, dtype=jnp.bool_),
            jnp.zeros(batch, dtype=jnp.int32),
            keys,
        )
        return jax.lax.while_loop(condition, step, initial)[5]

    return run
