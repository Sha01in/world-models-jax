"""Batched real-game evaluation with an independent RNG stream per game seed."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from gymnasium.vector import AsyncVectorEnv, AutoresetMode

from src.controller import canonical_doom_action, controller_memory
from src.controller import get_action_linear, get_action_mlp
from src.env_utils import make_env


def make_batched_policy(
    vae,
    rnn,
    params,
    *,
    posterior,
    canonical,
    state_mode,
    controller_type,
    hidden_size,
    warmup,
):
    def one(image, hidden, episode_key, steps):
        next_key, step_key = jax.random.split(episode_key)
        features = vae.encoder(
            jnp.transpose(image.astype(jnp.float32) / 255, (2, 0, 1))
        ).reshape(-1)
        z = vae.mu_head(features)
        if posterior:
            z += jnp.exp(0.5 * vae.logvar_head(features)) * jax.random.normal(
                step_key, z.shape
            )
        memory = controller_memory(hidden, state_mode)
        if controller_type == "mlp":
            action = get_action_mlp(params, z, memory, 1, hidden_size)
        else:
            action = get_action_linear(params, z, memory, 1)
        action = jnp.where(steps < warmup, 0.0, action)
        model_action = canonical_doom_action(action) if canonical else action
        next_hidden = rnn(jnp.concatenate([z, model_action]), hidden)[1]
        return action, next_hidden, next_key

    @jax.jit
    def batch(images, hidden, keys, steps):
        # Keep single-game convolution/matmul shapes. vmap changes reduction
        # order and can move a recurrent policy across an action threshold.
        return jax.lax.map(
            lambda values: one(values[0], (values[1], values[2]), values[3], values[4]),
            (images, hidden[0], hidden[1], keys, steps),
        )

    return batch


def evaluate_parallel(policy, *, episodes, seed, workers, hidden_size):
    """Evaluate exactly the requested seeds, reseeding every assigned episode.

    SAME_STEP keeps exhausted slots safe to step while other slots finish. Their
    unassigned games are ignored. Explicit masked resets replace automatic resets
    with the next requested seed, and reset both memory states and the VAE RNG.
    """
    workers = min(workers, episodes)
    factory = partial(make_env, "VizdoomTakeCover-v0", render_mode="rgb_array")
    envs = AsyncVectorEnv(
        [factory] * workers, context="spawn", autoreset_mode=AutoresetMode.SAME_STEP
    )
    seeds = np.arange(seed, seed + workers)
    keys = jnp.stack([jax.random.PRNGKey(int(s)) for s in seeds])
    hidden = (jnp.zeros((workers, hidden_size)), jnp.zeros((workers, hidden_size)))
    steps = np.zeros(workers, np.int32)
    scores = np.zeros(workers)
    counts = np.zeros((workers, 3), np.int64)
    assigned = np.ones(workers, bool)
    next_episode = workers
    records = []
    try:
        obs, _ = envs.reset(seed=[int(s) for s in seeds])
        while len(records) < episodes:
            actions, hidden, keys = policy(obs, hidden, keys, steps)
            actions_np = np.asarray(actions)
            directions = np.where(
                actions_np[:, 0] < -0.3,
                0,
                np.where(actions_np[:, 0] > 0.3, 1, 2),
            )
            for i in np.flatnonzero(assigned):
                counts[i, directions[i]] += 1
            obs, reward, terminated, truncated, _ = envs.step(actions_np)
            steps += assigned
            scores += reward * assigned
            finished = assigned & (terminated | truncated)
            reset_mask = np.zeros(workers, bool)
            reset_seeds = [None] * workers
            for i in np.flatnonzero(finished):
                records.append(
                    {
                        "seed": int(seeds[i]),
                        "survival_steps": int(steps[i]),
                        "score": float(scores[i]),
                        "actions_left_right_wait": counts[i].tolist(),
                    }
                )
                if next_episode < episodes:
                    seeds[i] = seed + next_episode
                    next_episode += 1
                    reset_seeds[i] = int(seeds[i])
                    reset_mask[i] = True
                    keys = keys.at[i].set(jax.random.PRNGKey(int(seeds[i])))
                    hidden = tuple(h.at[i].set(0.0) for h in hidden)
                    steps[i] = 0
                    scores[i] = 0.0
                    counts[i] = 0
                else:
                    assigned[i] = False
            if reset_mask.any():
                obs, _ = envs.reset(
                    seed=reset_seeds, options={"reset_mask": reset_mask}
                )
            if finished.any():
                print(f"{len(records)}/{episodes} real games complete", flush=True)
    finally:
        envs.close()
    return sorted(records, key=lambda record: record["seed"])
