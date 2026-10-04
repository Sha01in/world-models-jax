"""Real-game fitness with fixed single-game inference and explicit game seeds."""

import hashlib

import jax
import jax.numpy as jnp
import numpy as np
from gymnasium.vector import AsyncVectorEnv, AutoresetMode

from src.doom_reference import make_reference_policy


def parameter_fingerprint(params):
    return hashlib.sha256(np.asarray(params, dtype=np.float64).tobytes()).hexdigest()


def make_population_policy(vae, rnn):
    """Pass weights as data to avoid compiling a new actor for each candidate.

    The inner actor is the original reference policy on one game. Both maps
    preserve its convolution, dot-product, recurrent and per-seed RNG shapes.
    """

    @jax.jit
    def policy(images, hidden, keys, steps, controllers):
        def one(row):
            actor = make_reference_policy(vae, rnn, row[5], posterior=True)
            action, memory, key = actor(
                row[0][None],
                (row[1][None], row[2][None]),
                row[3][None],
                row[4][None],
            )
            return action[0], (memory[0][0], memory[1][0]), key[0]

        return jax.lax.map(
            one, (images, hidden[0], hidden[1], keys, steps, controllers)
        )

    return policy


def checked_population_records(rows, candidates, seeds, *, complete=False):
    expected = {(candidate, seed) for candidate in range(candidates) for seed in seeds}
    actual = [(row["candidate"], row["seed"]) for row in rows]
    if len(set(actual)) != len(actual) or not set(actual) <= expected:
        raise ValueError("Duplicate or unexpected candidate/game pair")
    if complete and set(actual) != expected:
        raise ValueError("Real fitness cohort incomplete")
    for row in rows:
        steps = row["survival_steps"]
        counts = row["actions_left_right_wait"]
        if (
            type(steps) is not int
            or not 1 <= steps <= 2100
            or len(counts) != 3
            or any(type(count) is not int or count < 0 for count in counts)
            or sum(counts) != steps
            or row["score"] != steps
            or type(row["terminated"]) is not bool
            or type(row["truncated"]) is not bool
            or not row["terminated"] ^ row["truncated"]
        ):
            raise ValueError("Invalid real-game outcome or action counts")
    return sorted(rows, key=lambda row: (row["candidate"], row["seed"]))


class PopulationEvaluator:
    """Reuse eight game workers across populations and training holdouts.

    Each candidate sees the same explicit seeds within a generation. Work is
    assigned to free slots; only (candidate,seed) determines its controller and
    latent RNG. Completed pairs can be omitted during verified recovery.
    """

    def __init__(self, policy, env_factory, *, workers=8, hidden_size=512):
        if workers < 1 or hidden_size < 1:
            raise ValueError("Positive workers and hidden size required")
        self.policy = policy
        self.workers = workers
        self.hidden_size = hidden_size
        self.envs = AsyncVectorEnv(
            [env_factory] * workers,
            context="spawn",
            autoreset_mode=AutoresetMode.SAME_STEP,
        )

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.envs.close()

    def evaluate(self, parameters, seeds, *, completed=(), on_episode=None):
        parameters = np.asarray(parameters, dtype=np.float64)
        if (
            parameters.ndim != 2
            or parameters.shape[1] != 1088
            or len(parameters) < 1
            or not np.isfinite(parameters).all()
            or not seeds
            or len(set(seeds)) != len(seeds)
            or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds)
        ):
            raise ValueError("Finite1088-weight candidates and unique seeds required")
        rows = checked_population_records(list(completed), len(parameters), seeds)
        done = {(row["candidate"], row["seed"]) for row in rows}
        jobs = [
            (candidate, seed)
            for candidate in range(len(parameters))
            for seed in seeds
            if (candidate, seed) not in done
        ]
        if not jobs:
            return rows
        assignments = [jobs[min(i, len(jobs) - 1)] for i in range(self.workers)]
        assigned = np.arange(self.workers) < len(jobs)
        next_job = min(self.workers, len(jobs))
        keys = jnp.stack([jax.random.PRNGKey(seed) for _, seed in assignments])
        memory = (
            jnp.zeros((self.workers, self.hidden_size), dtype=jnp.float32),
            jnp.zeros((self.workers, self.hidden_size), dtype=jnp.float32),
        )
        steps = np.zeros(self.workers, np.int32)
        scores = np.zeros(self.workers)
        counts = np.zeros((self.workers, 3), np.int64)
        obs, _ = self.envs.reset(seed=[seed for _, seed in assignments])
        while assigned.any():
            controllers = jnp.asarray(
                parameters[[candidate for candidate, _ in assignments]],
                dtype=jnp.float64,
            )
            actions, memory, keys = self.policy(obs, memory, keys, steps, controllers)
            actions = np.asarray(actions)
            if actions.shape != (self.workers, 1) or not np.isfinite(actions).all():
                raise FloatingPointError("Invalid population-policy action")
            directions = np.where(
                actions[:, 0] < -0.3333,
                0,
                np.where(actions[:, 0] > 0.3333, 1, 2),
            )
            for slot in np.flatnonzero(assigned):
                counts[slot, directions[slot]] += 1
            obs, reward, terminated, truncated, _ = self.envs.step(actions)
            steps += assigned
            scores += reward * assigned
            finished = assigned & (terminated | truncated)
            reset_mask = np.zeros(self.workers, bool)
            reset_seeds = [None] * self.workers
            for slot in np.flatnonzero(finished):
                candidate, seed = assignments[slot]
                row = dict(
                    candidate=candidate,
                    seed=seed,
                    survival_steps=int(steps[slot]),
                    score=float(scores[slot]),
                    actions_left_right_wait=counts[slot].tolist(),
                    terminated=bool(terminated[slot]),
                    truncated=bool(truncated[slot]),
                )
                # Validate before accepting a row as resumable evidence.
                checked_population_records(rows + [row], len(parameters), seeds)
                rows.append(row)
                if on_episode is not None:
                    on_episode(row)
                if next_job < len(jobs):
                    assignments[slot] = jobs[next_job]
                    next_job += 1
                    reset_seeds[slot] = assignments[slot][1]
                    reset_mask[slot] = True
                    keys = keys.at[slot].set(jax.random.PRNGKey(reset_seeds[slot]))
                    memory = tuple(state.at[slot].set(0) for state in memory)
                    steps[slot] = 0
                    scores[slot] = 0
                    counts[slot] = 0
                else:
                    assigned[slot] = False
            if reset_mask.any():
                obs, _ = self.envs.reset(
                    seed=reset_seeds, options={"reset_mask": reset_mask}
                )
        return checked_population_records(rows, len(parameters), seeds, complete=True)
