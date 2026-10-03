"""Evaluate a controller in real VizDoom on an explicit, reusable seed range."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from importlib.metadata import version

os.environ.setdefault("JAX_PLATFORMS", "cuda")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import jax
import jax.numpy as jnp
import numpy as np

from src.config import get_config
from src.controller import get_action_linear, get_action_mlp
from src.env_utils import make_env
from src.rnn import load_rnn
from src.vae import load_vae


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", default="checkpoints/VizdoomTakeCover-v0")
    parser.add_argument("--controller", default=None)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument(
        "--warmup", type=int, default=0, help="Legacy evaluation forced 50 no-op steps"
    )
    parser.add_argument(
        "--policy",
        choices=["controller", "random", "sweep", "still"],
        default="controller",
    )
    parser.add_argument("--output", default="artifacts/doom_evaluation.json")
    latents = parser.add_mutually_exclusive_group()
    latents.add_argument(
        "--mean-latents",
        action="store_true",
        help="Ablation: use VAE means during real-game inference",
    )
    latents.add_argument(
        "--posterior-latents",
        action="store_true",
        help="Override controller metadata to sample the VAE posterior",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel games per fused GPU inference batch",
    )
    args = parser.parse_args()
    if args.episodes < 1 or args.warmup < 0 or args.workers < 1:
        parser.error("Episodes and workers must be positive; warmup nonnegative")
    if args.workers > 1 and args.policy != "controller":
        parser.error("Parallel evaluation currently supports controller policies")
    cfg = get_config("VizdoomTakeCover-v0")
    base = Path(args.checkpoint_dir)
    controller_path = (
        Path(args.controller) if args.controller else base / "controller_dream.npz"
    )
    key = jax.random.PRNGKey(0)
    vae = rnn = params = None
    metadata = {}
    checkpoint_paths = (base / "vae.eqx", base / "rnn.eqx", controller_path)
    checkpoint_hashes = {}

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    if args.policy == "controller":
        checkpoint_hashes = {p.name: digest(p) for p in checkpoint_paths}
        vae = load_vae(base / "vae.eqx", cfg.latent_dim, key)
        rnn = load_rnn(base / "rnn.eqx", cfg)
        with np.load(controller_path) as data:
            params = jnp.asarray(data["params"])
            metadata = {k: data[k].item() for k in data.files if data[k].ndim == 0}
        controller_type = metadata.get("type", "linear")
        hidden_size = int(metadata.get("hidden_size", 64))
        state_mode = metadata.get("state_mode", "h")
        posterior = bool(metadata.get("posterior_sampling", False))
        if args.mean_latents:
            posterior = False
        elif args.posterior_latents:
            posterior = True
        canonical = bool(metadata.get("canonical_actions", False))

        @jax.jit
        def infer(image, hidden, step_key):
            features = vae.encoder(
                jnp.transpose(image.astype(jnp.float32) / 255, (2, 0, 1))
            ).reshape(-1)
            z = vae.mu_head(features)
            if posterior:
                z += jnp.exp(0.5 * vae.logvar_head(features)) * jax.random.normal(
                    step_key, z.shape
                )
            memory = (
                jnp.concatenate([hidden[1], hidden[0]])
                if state_mode == "hc"
                else hidden[0]
            )
            if controller_type == "mlp":
                action = get_action_mlp(params, z, memory, 1, hidden_size)
            else:
                action = get_action_linear(params, z, memory, 1)
            return z, action

        @jax.jit
        def update(z, action, hidden):
            if canonical:
                action = jnp.where(
                    action < -0.3, -1.0, jnp.where(action > 0.3, 1.0, 0.0)
                )
            return rnn(jnp.concatenate([z, action]), hidden)[1]

    env = (
        make_env("VizdoomTakeCover-v0", render_mode="rgb_array")
        if args.workers == 1
        else None
    )
    records = []
    started = time.monotonic()
    try:
        if args.workers > 1:
            from src.doom_evaluation import make_batched_policy, evaluate_parallel

            policy = make_batched_policy(
                vae,
                rnn,
                params,
                posterior=posterior,
                canonical=canonical,
                state_mode=state_mode,
                controller_type=controller_type,
                hidden_size=hidden_size,
                warmup=args.warmup,
            )
            records = evaluate_parallel(
                policy,
                episodes=args.episodes,
                seed=args.seed,
                workers=args.workers,
                hidden_size=cfg.hidden_size,
            )
        for episode in range(args.episodes if args.workers == 1 else 0):
            seed = args.seed + episode
            # Explicitly seed the game even with the legacy wrapper.
            env.game.set_seed(seed)
            obs, _ = env.reset(seed=seed)
            hidden = (jnp.zeros(cfg.hidden_size), jnp.zeros(cfg.hidden_size))
            ep_key = jax.random.PRNGKey(seed)
            rng = np.random.default_rng(seed)
            score = 0.0
            counts = [0, 0, 0]
            for t in range(2100):
                if args.policy == "controller":
                    ep_key, step_key = jax.random.split(ep_key)
                    z, action = infer(jnp.asarray(obs), hidden, step_key)
                    if t < args.warmup:
                        action = jnp.zeros(1)
                    hidden = update(z, action, hidden)
                    action_np = np.asarray(action)
                elif args.policy == "random":
                    action_np = np.array([rng.uniform(-1, 1)], np.float32)
                elif args.policy == "sweep":
                    action_np = np.array([1.0 if (t // 100) % 2 else -1.0], np.float32)
                else:
                    action_np = np.zeros(1, np.float32)
                v = float(action_np[0])
                counts[0 if v < -0.3 else 1 if v > 0.3 else 2] += 1
                obs, reward, terminated, truncated, _ = env.step(action_np)
                score += reward
                if terminated or truncated:
                    break
            records.append(
                {
                    "seed": seed,
                    "survival_steps": t + 1,
                    "score": score,
                    "actions_left_right_wait": counts,
                }
            )
            if (episode + 1) % 10 == 0 or episode + 1 == args.episodes:
                print(
                    f"{episode + 1}/{args.episodes}: mean={np.mean([r['survival_steps'] for r in records]):.2f}",
                    flush=True,
                )
    finally:
        if env is not None:
            env.close()
    scores = np.asarray([r["survival_steps"] for r in records], float)

    if args.policy == "controller" and checkpoint_hashes != {
        p.name: digest(p) for p in checkpoint_paths
    }:
        raise RuntimeError(
            "Checkpoints changed during evaluation; freeze them and rerun"
        )

    report = {
        "policy": args.policy,
        "episodes": len(records),
        "seed_start": args.seed,
        "warmup_steps": args.warmup,
        "mean_latents_override": args.mean_latents,
        "posterior_latents_override": args.posterior_latents,
        "inference_posterior_sampling": posterior
        if args.policy == "controller"
        else None,
        "workers": min(args.workers, args.episodes),
        "mean": float(scores.mean()),
        "std": float(scores.std()),
        "standard_error": float(scores.std(ddof=1) / np.sqrt(len(scores)))
        if len(scores) > 1
        else None,
        "min": float(scores.min()),
        "max": float(scores.max()),
        "solve_threshold": 750,
        "solved": bool(len(records) >= 100 and scores.mean() > 750),
        "elapsed_seconds": time.monotonic() - started,
        "jax_version": jax.__version__,
        "package_versions": {
            name: version(name)
            for name in (
                "jax",
                "jaxlib",
                "equinox",
                "optax",
                "vizdoom",
                "gymnasium",
                "numpy",
                "opencv-python",
            )
        },
        "device": str(jax.devices()[0]),
        "controller_metadata": metadata,
        "checkpoint_dir": str(base) if args.policy == "controller" else None,
        "controller_path": str(controller_path)
        if args.policy == "controller"
        else None,
        "checkpoint_sha256": checkpoint_hashes,
        "episodes_detail": records,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"Result: {report['mean']:.2f} +/- {report['std']:.2f}; solved={report['solved']}"
    )


if __name__ == "__main__":
    main()
