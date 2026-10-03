"""Evaluate the frozen supplied controller under the pinned legacy settings."""
# ruff: noqa: E402 -- initialize CUDA/precision before importing models.

import argparse
from datetime import datetime, timezone
from functools import partial
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cuda")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")

import jax.numpy as jnp
import numpy as np

from src.doom_evaluation import evaluate_parallel
from src.doom_reference import checked_reference_arrays, load_author_models
from src.doom_reference import make_reference_policy
from src.doom_reference_env import make_reference_env


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_records(rows, expected_seeds):
    seeds = [row["seed"] for row in rows]
    if len(seeds) != len(set(seeds)) or not set(seeds) <= set(expected_seeds):
        raise ValueError("Reference rows contain duplicate or unexpected seeds")
    for row in rows:
        steps = row["survival_steps"]
        if not 1 <= steps <= 2100:
            raise ValueError("Invalid survival length")
        if sum(row["actions_left_right_wait"]) != steps:
            raise ValueError("Action counts differ from survival")
        if not row["terminated"] ^ row["truncated"]:
            raise ValueError("Each completed game must be death or timeout")
        if row["score"] != steps:
            raise ValueError("Living reward differs from survival steps")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", required=True)
    parser.add_argument("--asset-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=120000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--difficulty", type=int, choices=(4, 5), default=4)
    parser.add_argument("--color-order", choices=("rgb", "bgr"), default="rgb")
    args = parser.parse_args()
    if args.episodes < 1 or args.workers < 1:
        parser.error("Positive episode/worker counts required")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("Preserve completed reference report")
    output.parent.mkdir(parents=True, exist_ok=True)
    partial_output = output.with_suffix(".partial.json")
    expected_seeds = list(range(args.seed, args.seed + args.episodes))
    if jax.default_backend() != "gpu":
        raise RuntimeError("CUDA required for the reference policy evaluation")
    vae, rnn, _, manifest = load_author_models(args.reference_dir)
    payload, _ = checked_reference_arrays(args.reference_dir)
    controller = jnp.asarray(payload["controller.json"][0], dtype=jnp.float64)
    inputs = {
        str(Path(args.reference_dir) / name): row["sha256"]
        for name, row in manifest["files"].items()
    }
    for name in ("take_cover.wad", "freedoom2.wad"):
        path = Path(args.asset_dir) / name
        inputs[str(path)] = digest(path)
    protocol = dict(
        policy_source="supplied public author weights; diagnostic control",
        reference_commit=manifest["reference_commit"],
        input_sha256=inputs,
        source_sha256={
            str(path): digest(path)
            for path in (
                Path("src/doom_reference.py"),
                Path("src/doom_reference_env.py"),
                Path("src/doom_evaluation.py"),
                Path(__file__),
            )
        },
        inference_posterior_sampling=True,
        controller="bias-free tanh over z,c,h; raw continuous action into RNN",
        preprocessing=f"{args.color_order.upper()}24 native640x480; crop400; legacy bytescale/Pillow bilinear/uint8 wrap",
        original_screen_format="Legacy BGR24 emitted effectiveRGB. Modern RGB24 preserves its byte order.",
        difficulty=args.difficulty,
        original_requested_difficulty=5,
        legacy_difficulty_mapping="doom-py0.0.15 clamps5 to4; current engine accepts5. Effective4 matches original launch argument.",
        episode_start_time=14,
        episode_timeout=2100,
        threshold=0.3333,
        world_precision="FP32; highest matmul/conv precision",
        posterior_and_controller_precision="FP64",
        rng="Per-game JAX key from explicit actual game seed; independent of worker assignment",
        limitations=[
            "Installed ViZDoom 1.2.4 replaces legacy doom-py engine",
            "Pillow version differs or is unverified historically",
            "JAX latent noise and explicit game seeds differ from legacy Gym RNG/seed behavior",
            "Exact reported paper checkpoint provenance unverified",
            "Supplied weights do not reproduce our training",
        ],
        package_versions={
            name: version(name)
            for name in ("jax", "jaxlib", "equinox", "vizdoom", "numpy", "pillow")
        },
    )
    if any(digest(path) != fingerprint for path, fingerprint in inputs.items()):
        raise ValueError("Reference inputs differ from pinned fingerprints")
    rows = []
    if partial_output.exists():
        saved = json.loads(partial_output.read_text())
        if saved["protocol"] != protocol or saved["expected_seeds"] != expected_seeds:
            raise ValueError("Partial report belongs to a different frozen evaluation")
        rows = saved["episodes_detail"]
        verify_records(rows, expected_seeds)
    started = time.monotonic()

    def record(row):
        rows.append(row)
        verify_records(rows, expected_seeds)
        snapshot = dict(
            protocol=protocol,
            expected_seeds=expected_seeds,
            episodes_detail=sorted(rows, key=lambda r: r["seed"]),
        )
        temporary = partial_output.with_suffix(".tmp")
        temporary.write_text(json.dumps(snapshot, indent=2) + "\n")
        temporary.replace(partial_output)
        print(
            json.dumps(
                dict(
                    completed=len(rows),
                    total=args.episodes,
                    mean=float(np.mean([r["survival_steps"] for r in rows])),
                )
            ),
            flush=True,
        )

    remaining = [
        seed for seed in expected_seeds if seed not in {r["seed"] for r in rows}
    ]
    if remaining:
        policy = make_reference_policy(vae, rnn, controller)
        evaluate_parallel(
            policy,
            episodes=len(remaining),
            seed=args.seed,
            workers=args.workers,
            hidden_size=512,
            env_factory=partial(
                make_reference_env,
                args.reference_dir,
                args.asset_dir,
                difficulty=args.difficulty,
                color_order=args.color_order,
            ),
            action_threshold=0.3333,
            record_outcomes=True,
            episode_seeds=remaining,
            on_episode=record,
        )
    verify_records(rows, expected_seeds)
    if len(rows) != args.episodes:
        raise ValueError("Reference cohort incomplete")
    if any(digest(path) != fingerprint for path, fingerprint in inputs.items()):
        raise ValueError("Frozen reference inputs changed during evaluation")
    scores = np.asarray([row["survival_steps"] for row in rows], dtype=np.float64)
    report = dict(
        measured_at=datetime.now(timezone.utc).isoformat(),
        protocol=protocol,
        policy="controller",
        episodes=len(rows),
        seed_start=args.seed,
        workers=min(args.workers, args.episodes),
        mean=float(scores.mean()),
        std=float(scores.std()),
        standard_error=float(scores.std(ddof=1) / np.sqrt(len(scores)))
        if len(scores) > 1
        else None,
        min=float(scores.min()),
        max=float(scores.max()),
        deaths=sum(row["terminated"] for row in rows),
        timeouts=sum(row["truncated"] for row in rows),
        elapsed_seconds_this_attempt=time.monotonic() - started,
        diagnostic_only=True,
        target_mean_observed=bool(len(rows) >= 100 and scores.mean() >= 1092),
        episodes_detail=sorted(rows, key=lambda row: row["seed"]),
    )
    with output.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps({key: report[key] for key in ("mean", "std", "deaths", "timeouts")}),
        flush=True,
    )


if __name__ == "__main__":
    main()
