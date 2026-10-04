"""Optimize three coherent controller gains using frozen-world real survival."""
# ruff: noqa: E402 -- initialize CUDA before importing shared training core.

import argparse
from datetime import datetime, timezone
from functools import partial
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
import traceback

os.environ.setdefault("JAX_PLATFORMS", "cuda")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import jax
import numpy as np

from scripts.tools.train_doom_reference_real import digest, verify_frozen
from src.doom_subspace_training import (
    METHOD,
    SEARCH_SPACE,
    run_subspace_search as run_search,
)
from src.doom_real_training import (
    PopulationEvaluator,
    make_population_policy,
    parameter_fingerprint,
)
from src.doom_reference import load_author_models
from src.doom_reference_env import make_reference_env
from src.doom_subspace_initializer import registered_initializer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    protocol_path = Path(args.protocol)
    protocol = json.loads(protocol_path.read_text())
    training = protocol["arguments"]
    positive = (
        "generations",
        "pop_size",
        "fitness_games",
        "holdout_games",
        "workers",
        "validate_every",
    )
    if any(type(training[key]) is not int or training[key] < 1 for key in positive):
        parser.error("Positive integer training limits required")
    if (
        training["generations"] > 4
        or not 2 <= training["pop_size"] <= 8
        or training["fitness_games"] > 16
        or training["holdout_games"] > 64
        or training["workers"] > 8
        or not np.isfinite(training["sigma"])
        or training["sigma"] <= 0
        or type(training["seed"]) is not int
        or not 1 <= training["seed"] < 2**32
        or training["temperature"] is not None
        or Path(training["output"]).suffix != ".npz"
    ):
        parser.error("Expected bounded real-survival search; no dream temperature")
    fitness_seeds = set(
        range(
            training["fitness_seed"],
            training["fitness_seed"]
            + training["generations"] * training["fitness_games"],
        )
    )
    holdout_seeds = set(
        range(
            training["holdout_seed"],
            training["holdout_seed"] + training["holdout_games"],
        )
    )
    validation_seeds = set(
        range(
            *[
                protocol["validation_seed_range"][0],
                protocol["validation_seed_range"][1] + 1,
            ]
        )
    )
    test_seeds = set(
        range(*[protocol["test_seed_range"][0], protocol["test_seed_range"][1] + 1])
    )
    groups = [fitness_seeds, holdout_seeds, validation_seeds, test_seeds]
    if any(a & b for i, a in enumerate(groups) for b in groups[i + 1 :]) or any(
        not 0 <= s < 2**32 for group in groups for s in group
    ):
        raise ValueError("Fitness/holdout/validation/test seeds must be separated")
    packages = {
        name: version(name)
        for name in ("jax", "jaxlib", "equinox", "vizdoom", "numpy", "pillow", "cma")
    }
    if packages != protocol["package_versions"]:
        raise ValueError("Runtime differs from registered versions")
    settings = dict(
        arguments=training,
        reference_commit=protocol["reference_commit"],
        input_sha256=protocol["frozen_inputs"],
        source_sha256=protocol["frozen_source"],
        package_versions=packages,
        protocol=str(protocol_path),
        protocol_sha256=digest(protocol_path),
        training_method=METHOD,
        search_space=SEARCH_SPACE,
        initializer_parameters_sha256=protocol["initializer_parameters_sha256"],
        selection="Training holdout selects mean checkpoints; fresh real validation separately chooses policies for tests",
        common_random_numbers="Every candidate sees the same registered fitness seeds in each generation; new seeds next generation",
        imported_public_world=True,
        own_world_model_training=False,
        goal_completion_unproven=True,
    )
    verify_frozen(settings)
    if (
        protocol["search_space"] != SEARCH_SPACE
        or protocol["training_method"] != METHOD
    ):
        raise ValueError("Registered three-gain space differs")
    initial = registered_initializer(protocol)
    if args.check_only:
        print(
            json.dumps(
                dict(
                    ready=True,
                    gpu_jobs_started=0,
                    initializer_parameters_sha256=parameter_fingerprint(initial),
                    fitness_games=training["fitness_games"],
                    holdout_games=training["holdout_games"],
                    optimizer_dimensions=3,
                )
            )
        )
        return
    failure = Path(str(training["output"]) + ".failed.json")
    if failure.exists():
        raise FileExistsError(
            "Diagnose and preserve the captured failure before recovery"
        )
    import fcntl

    with Path("artifacts/doom_vision_round3_vaes.lock").open("a") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if jax.default_backend() != "gpu":
            raise RuntimeError("CUDA required for real-survival training")
        vae, rnn, _, manifest = load_author_models(training["reference_dir"])
        if (
            manifest["reference_commit"] != protocol["reference_commit"]
            or parameter_fingerprint(initial)
            != protocol["initializer_parameters_sha256"]
        ):
            raise ValueError("Initializer or frozen world differs")
        try:
            with PopulationEvaluator(
                make_population_policy(vae, rnn),
                partial(
                    make_reference_env,
                    training["reference_dir"],
                    training["asset_dir"],
                    difficulty=4,
                    color_order="rgb",
                ),
                workers=training["workers"],
            ) as evaluator:
                run_search(settings, evaluator, initial, resume=args.resume)
        except BaseException:
            with failure.open("x") as stream:
                stream.write(
                    json.dumps(
                        dict(
                            measured_at=datetime.now(timezone.utc).isoformat(),
                            traceback=traceback.format_exc(),
                            protocol_sha256=settings["protocol_sha256"],
                            preserve_completed_and_partial_work=True,
                        ),
                        indent=2,
                    )
                    + "\n"
                )
            raise


if __name__ == "__main__":
    main()
