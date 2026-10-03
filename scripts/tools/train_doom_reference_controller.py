"""Refine a bias-free controller on the frozen public reference world."""
# ruff: noqa: E402 -- configure CUDA and precision before model imports.

import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cuda")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")

import cma
import jax.numpy as jnp
import numpy as np

from src.doom_reference import checked_reference_arrays, load_author_models
from src.doom_reference_training import (
    make_reference_dream_engine,
    reference_start_pool,
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_policy(path, params, *, score, generation, temperature):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez(
            stream,
            params=np.asarray(params, dtype=np.float64),
            score=score,
            generation=generation,
            temperature=temperature,
            type="reference_linear",
            state_mode="ch",
            posterior_sampling=True,
            canonical_actions=False,
        )
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=91)
    parser.add_argument("--temperature", type=float, default=1.15)
    parser.add_argument("--generations", type=int, default=500)
    parser.add_argument("--pop-size", type=int, default=64)
    parser.add_argument("--rollouts", type=int, default=16)
    parser.add_argument("--candidate-batch", type=int, default=64)
    parser.add_argument("--validation-rollouts", type=int, default=64)
    parser.add_argument("--validate-every", type=int, default=10)
    parser.add_argument("--sigma", type=float, default=0.02)
    parser.add_argument("--initial", choices=("author", "random"), default="author")
    args = parser.parse_args()
    if (
        args.generations < 1
        or args.pop_size < 2
        or min(
            args.rollouts,
            args.candidate_batch,
            args.validation_rollouts,
            args.validate_every,
        )
        < 1
        or not np.isfinite([args.sigma, args.temperature]).all()
        or min(args.sigma, args.temperature) <= 0
        or not 0 <= args.seed < 2**32 - 100000
    ):
        parser.error("Positive finite settings and an unsigned seed required")
    if jax.default_backend() != "gpu":
        raise RuntimeError("CUDA required; run check_gpu.py before dispatch")
    output = Path(args.output)
    if output.suffix != ".npz":
        parser.error("Output must be a new .npz policy")
    paths = dict(
        best=output,
        last=output.with_name(output.stem + ".last.npz"),
        metadata=Path(str(output) + ".json"),
        history=Path(str(output) + ".history.jsonl"),
        optimizer=Path(str(output) + ".optimizer.pkl"),
        resume=Path(str(output) + ".resume.json"),
    )
    if any(path.exists() for path in paths.values()):
        raise FileExistsError("Preserve previous search; use a new output name")
    output.parent.mkdir(parents=True, exist_ok=True)
    _, rnn, _, manifest = load_author_models(args.reference_dir)
    payload, _ = checked_reference_arrays(args.reference_dir)
    means, logvars = reference_start_pool(payload)
    permutation = np.random.default_rng(args.seed).permutation(len(means))
    split_at = max(1, min(len(means) - 1, int(len(means) * 0.9)))
    train_indices, val_indices = permutation[:split_at], permutation[split_at:]
    pools = {
        False: (jnp.asarray(means[train_indices]), jnp.asarray(logvars[train_indices])),
        True: (jnp.asarray(means[val_indices]), jnp.asarray(logvars[val_indices])),
    }
    engine = make_reference_dream_engine(rnn)

    def evaluate(candidates, key, rollouts, *, validation=False):
        index_key, latent_key, noise_key = jax.random.split(key, 3)
        mu_pool, lv_pool = pools[validation]
        indices = jax.random.randint(index_key, (rollouts,), 0, len(mu_pool))
        starts = mu_pool[indices] + jnp.exp(lv_pool[indices] / 2) * jax.random.normal(
            latent_key, (rollouts, 64), dtype=jnp.float64
        )
        streams = jax.random.split(noise_key, rollouts)
        batch = min(args.candidate_batch, len(candidates))
        rewards = []
        for offset in range(0, len(candidates), batch):
            chunk = jnp.asarray(candidates[offset : offset + batch], dtype=jnp.float64)
            actual = len(chunk)
            padded = jnp.pad(chunk, ((0, batch - actual), (0, 0)))
            scores = np.asarray(
                engine(
                    jnp.repeat(padded, rollouts, axis=0),
                    jnp.tile(starts, (batch, 1)),
                    jnp.tile(streams, (batch, 1)),
                    args.temperature,
                )
            ).reshape(batch, rollouts)[:actual]
            if (scores < 0).any() or not np.isfinite(scores).all():
                raise FloatingPointError(
                    "Invalid active dream trajectory; inspect world"
                )
            rewards.extend(scores.mean(axis=1))
        return np.asarray(rewards)

    key = jax.random.PRNGKey(args.seed)
    key, init_key = jax.random.split(key)
    initial = np.asarray(payload["controller.json"][0], dtype=np.float64)
    if args.initial == "random":
        initial = (
            np.asarray(jax.random.normal(init_key, (1088,), dtype=jnp.float64)) * 0.01
        )
    settings = dict(
        arguments=vars(args),
        purpose="Own controller refinement with imported frozen public VAE/RNN; no own VAE/RNN training claimed",
        reference_commit=manifest["reference_commit"],
        input_sha256={
            str(Path(args.reference_dir) / name): row["sha256"]
            for name, row in manifest["files"].items()
        },
        source_sha256={
            str(path): digest(path)
            for path in (
                Path(__file__),
                Path("src/doom_reference.py"),
                Path("src/doom_reference_training.py"),
                Path("src/dream.py"),
            )
        },
        initial_policy_sha256=hashlib.sha256(initial.tobytes()).hexdigest(),
        start_train_indices=train_indices.tolist(),
        start_validation_indices=val_indices.tolist(),
        controller="1088 weights; bias-free tanh(z,c,h); raw action; restart at first step",
        dream="2100 actions maximum; death logit strictly positive; terminal action counted",
        precision="FP64 latent/control; FP32 RNN; highest matmul precision",
        random_streams="JAX; common starts/noise across candidates; fixed held-out dream streams",
        jax_version=jax.__version__,
        cma_version=cma.__version__,
        device=str(jax.devices()[0]),
        real_performance_unproven=True,
    )
    for path, fingerprint in settings["input_sha256"].items():
        if digest(path) != fingerprint:
            raise ValueError("Frozen reference inputs changed")
    paths["metadata"].write_text(json.dumps(settings, indent=2) + "\n")
    validation_key = jax.random.PRNGKey(args.seed + 100000)
    best_score = float(
        evaluate([initial], validation_key, args.validation_rollouts, validation=True)[
            0
        ]
    )
    best_generation = 0
    save_policy(
        output, initial, score=best_score, generation=0, temperature=args.temperature
    )
    baseline = dict(
        generation=0,
        dream_validation_score=best_score,
        best_generation=0,
        real_performance_unproven=True,
    )
    paths["history"].write_text(json.dumps(baseline) + "\n")
    print(json.dumps(baseline), flush=True)
    optimizer = cma.CMAEvolutionStrategy(
        initial,
        args.sigma,
        {
            "popsize": args.pop_size,
            "seed": args.seed or 1,
            "verbose": -9,
        },
    )
    for generation in range(1, args.generations + 1):
        started = time.monotonic()
        key, dream_key = jax.random.split(key)
        solutions = optimizer.ask()
        rewards = evaluate(solutions, dream_key, args.rollouts)
        optimizer.tell(solutions, -rewards)
        record = dict(
            generation=generation,
            population_mean=float(rewards.mean()),
            population_best=float(rewards.max()),
        )
        if (
            generation == 1
            or generation % args.validate_every == 0
            or generation == args.generations
        ):
            checked = np.stack([solutions[int(rewards.argmax())], optimizer.mean])
            scores = evaluate(
                checked, validation_key, args.validation_rollouts, validation=True
            )
            winner = int(scores.argmax())
            record.update(
                dream_validation_score=float(scores[winner]),
                mean_validation_score=float(scores[1]),
            )
            if scores[winner] > best_score:
                best_score, best_generation = float(scores[winner]), generation
                save_policy(
                    output,
                    checked[winner],
                    score=best_score,
                    generation=generation,
                    temperature=args.temperature,
                )
            save_policy(
                paths["last"],
                optimizer.mean,
                score=float(scores[1]),
                generation=generation,
                temperature=args.temperature,
            )
            temporary = paths["optimizer"].with_suffix(".tmp")
            with temporary.open("wb") as stream:
                pickle.dump(
                    dict(optimizer=optimizer, numpy_random_state=np.random.get_state()),
                    stream,
                )
            temporary.replace(paths["optimizer"])
            paths["resume"].write_text(
                json.dumps(
                    dict(
                        generation=generation,
                        next_key=np.asarray(key).tolist(),
                        best_generation=best_generation,
                        best_dream_score=best_score,
                        best_sha256=digest(output),
                        last_sha256=digest(paths["last"]),
                        optimizer_sha256=digest(paths["optimizer"]),
                        complete=False,
                    ),
                    indent=2,
                )
                + "\n"
            )
        record.update(
            best_dream_score=best_score,
            best_generation=best_generation,
            seconds=time.monotonic() - started,
        )
        with paths["history"].open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
    for group in ("input_sha256", "source_sha256"):
        if any(
            digest(path) != fingerprint for path, fingerprint in settings[group].items()
        ):
            raise ValueError("Frozen search inputs/source changed during execution")
    resume = json.loads(paths["resume"].read_text())
    resume["complete"] = True
    paths["resume"].write_text(json.dumps(resume, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                best_generation=best_generation,
                best_dream_score=best_score,
                real_performance_unproven=True,
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
