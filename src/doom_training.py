"""Reproducible Doom controller evolution with repeated, comparable trials."""

import glob
import hashlib
import json
import os
from pathlib import Path
import time

import cma
import jax
import jax.numpy as jnp
import numpy as np

from src.controller import get_action_linear, get_action_mlp
from src.dream import make_dream_engine
from src.jax_cma import CMA_ES
from src.rnn import load_rnn


def evolve(args, config):
    if (
        args.pop_size < 2
        or args.rollouts < 1
        or args.mini_batch_size < 1
        or args.validate_every < 1
        or args.validation_rollouts < 1
    ):
        raise ValueError(
            "Population must be at least two; rollouts and batch size must be positive"
        )
    if args.strategy == "es" and args.pop_size % 2:
        raise ValueError("Antithetic ES requires an even population")
    if (
        args.temperature <= 0
        or (args.temp_start is not None and args.temp_start <= 0)
        or (args.temp_end is not None and args.temp_end <= 0)
    ):
        raise ValueError("Temperatures must be positive")
    base = Path(args.checkpoint_dir or os.path.join("checkpoints", config.env_name))
    path = base / args.output
    path.parent.mkdir(parents=True, exist_ok=True)
    rnn_path = base / "rnn.eqx"
    rnn = load_rnn(rnn_path, config)
    sidecar = Path(str(rnn_path) + ".json")
    model_metadata = json.loads(sidecar.read_text()) if sidecar.exists() else {}
    posterior = model_metadata.get("posterior_sampling", False)
    canonical = model_metadata.get("canonical_actions", False)
    state_mode = args.state_mode or "hc"
    rng = np.random.default_rng(args.seed)
    files = sorted(
        glob.glob(
            os.path.join(args.data_dir or "data/series/" + config.env_name, "*.npz")
        )
    )
    if not files:
        raise ValueError("No latent episodes found")
    # Only actual episode starts have a corresponding all-zero memory state.
    starts, logvars = [], []
    for index in rng.choice(len(files), min(2048, len(files)), replace=False):
        with np.load(files[index]) as data:
            starts.append(data["mu"][0])
            logvars.append(data["logvar"][0])
    starts, logvars = jnp.asarray(np.stack(starts)), jnp.asarray(np.stack(logvars))
    split_at = max(1, int(len(starts) * 0.9))
    train_mu, train_logvar = starts[:split_at], logvars[:split_at]
    val_mu, val_logvar = starts[split_at:], logvars[split_at:]
    if len(val_mu) == 0:
        val_mu, val_logvar = train_mu, train_logvar

    input_dim = config.latent_dim + config.hidden_size * (
        2 if state_mode == "hc" else 1
    )
    if args.controller_type == "linear":
        num_params = input_dim + 1
        controller_fn = get_action_linear
    else:
        num_params = (input_dim + 1) * args.hidden_size + args.hidden_size + 1

        def controller_fn(p, z, h, ad):
            return get_action_mlp(p, z, h, ad, args.hidden_size)

    engine = make_dream_engine(
        rnn,
        controller_fn,
        config,
        args.dream_length,
        state_mode,
        canonical,
        args.done_mode,
        model_metadata.get("done_positive_weight", 1.0),
    )

    def evaluate(candidates, key, temperature, rollouts, validation=False):
        mus, logvars_pool = (
            (val_mu, val_logvar) if validation else (train_mu, train_logvar)
        )
        index_key, latent_key, noise_key = jax.random.split(key, 3)
        indices = jax.random.randint(index_key, (rollouts,), 0, len(mus))
        seed_zs = mus[indices]
        if posterior:
            seed_zs += jnp.exp(0.5 * logvars_pool[indices]) * jax.random.normal(
                latent_key, seed_zs.shape
            )
        # Every candidate gets the same start states and random streams this generation.
        seed_keys = jax.random.split(noise_key, rollouts)
        results = []
        batch_size = min(args.mini_batch_size, len(candidates))
        for offset in range(0, len(candidates), batch_size):
            chunk = candidates[offset : offset + batch_size]
            # Pad the final chunk so one executable handles all population sizes.
            n = len(chunk)
            padded = jnp.pad(chunk, ((0, batch_size - n), (0, 0)))
            params = jnp.repeat(padded, rollouts, axis=0)
            zs = jnp.tile(seed_zs, (batch_size, 1))
            keys = jnp.tile(seed_keys, (batch_size, 1))
            scores = engine(params, zs, keys, temperature).reshape(batch_size, rollouts)
            results.append(scores.mean(axis=-1)[:n])
        return np.asarray(jnp.concatenate(results))

    key = jax.random.PRNGKey(args.seed)
    key, init_key = jax.random.split(key)
    mean = jax.random.normal(init_key, (num_params,)) * 0.01
    if args.initial_controller:
        with np.load(args.initial_controller) as data:
            initial = data["params"]
        if initial.shape != (num_params,):
            raise ValueError(
                f"Initial controller has {initial.shape}; expected {(num_params,)}"
            )
        mean = jnp.asarray(initial)
    optimizer = state = None
    if args.strategy == "cma":
        optimizer = cma.CMAEvolutionStrategy(
            np.asarray(mean),
            0.1,
            {"popsize": args.pop_size, "seed": args.seed or 1, "verbose": -9},
        )
    elif args.strategy == "jax_cma":
        optimizer = CMA_ES(num_params, args.pop_size, sigma_init=0.1)
        state = optimizer.init(init_key)
        # Preserve the selected mean even when warm-starting.
        state = state._replace(mean=mean) if hasattr(state, "_replace") else state
        ask, tell = jax.jit(optimizer.ask), jax.jit(optimizer.tell)

    @jax.jit
    def es_update(current, eps, rewards):
        ranks = jnp.argsort(jnp.argsort(rewards)) / (args.pop_size - 1) - 0.5
        return current + 0.01 * (ranks @ eps) / (args.pop_size * 0.1)

    best_validation = -np.inf
    validation_key = jax.random.PRNGKey(args.seed + 100000)
    history_path = Path(str(path) + ".history.jsonl")
    if history_path.exists():
        raise FileExistsError(f"Use a fresh output name: {history_path} already exists")
    settings = dict(
        vars(args),
        state_mode=state_mode,
        posterior_sampling=posterior,
        canonical_actions=canonical,
        num_params=num_params,
        rnn_sha256=hashlib.sha256(rnn_path.read_bytes()).hexdigest(),
        jax_version=jax.__version__,
        device=str(jax.devices()[0]),
    )
    Path(str(path) + ".json").write_text(json.dumps(settings, indent=2) + "\n")
    print(
        f"Doom evolution: {args.strategy}, {num_params} parameters, {args.rollouts} trials/candidate; {jax.devices()}",
        flush=True,
    )
    temp_start = args.temperature if args.temp_start is None else args.temp_start
    temp_end = args.temperature if args.temp_end is None else args.temp_end
    for gen in range(args.generations):
        started = time.monotonic()
        temperature = temp_start + (temp_end - temp_start) * gen / max(
            1, args.generations - 1
        )
        key, sample_key, dream_key = jax.random.split(key, 3)
        if args.strategy == "es":
            eps_half = jax.random.normal(sample_key, (args.pop_size // 2, num_params))
            eps = jnp.concatenate([eps_half, -eps_half])
            candidates = mean + 0.1 * eps
        elif args.strategy == "cma":
            solutions = optimizer.ask()
            candidates = jnp.asarray(solutions)
        else:
            candidates, asked_state = ask(state)
        rewards = evaluate(candidates, dream_key, temperature, args.rollouts)
        if not np.isfinite(rewards).all():
            raise FloatingPointError("Non-finite dream score")
        if args.strategy == "es":
            mean = es_update(mean, eps, jnp.asarray(rewards))
        elif args.strategy == "cma":
            optimizer.tell(solutions, -rewards)
            mean = jnp.asarray(optimizer.mean)
        else:
            state = tell(asked_state, candidates, -jnp.asarray(rewards))
            mean = state.mean
        record = {
            "generation": gen + 1,
            "temperature": temperature,
            "population_mean": float(rewards.mean()),
            "population_best": float(rewards.max()),
        }
        if (
            gen == 0
            or (gen + 1) % args.validate_every == 0
            or gen + 1 == args.generations
        ):
            checked = jnp.stack([candidates[int(rewards.argmax())], mean])
            validation = evaluate(
                checked,
                validation_key,
                args.temperature,
                args.validation_rollouts,
                True,
            )
            winner = int(validation.argmax())
            record["validation_score"] = float(validation[winner])
            if validation[winner] > best_validation:
                best_validation = float(validation[winner])
                np.savez(
                    path,
                    params=np.asarray(checked[winner]),
                    score=best_validation,
                    type=args.controller_type,
                    hidden_size=args.hidden_size,
                    state_mode=state_mode,
                    posterior_sampling=posterior,
                    canonical_actions=canonical,
                    done_mode=args.done_mode,
                    generation=gen + 1,
                )
        record["best_validation_score"] = best_validation
        record["seconds"] = time.monotonic() - started
        with history_path.open("a") as f:
            f.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
    print(f"Saved {path}: dream validation score {best_validation:.2f}", flush=True)
