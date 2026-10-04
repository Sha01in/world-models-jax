"""Bounded real-survival controller refinement on a frozen reference world.

This deliberately uses real simulator rewards rather than dream fitness. A
separate fresh validation comparison must select any policy for reserved tests.
"""
# ruff: noqa: E402 -- initialize CUDA and precision before importing models.

import argparse
from datetime import datetime, timezone
from functools import partial
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import pickle
import sys
import traceback

os.environ.setdefault("JAX_PLATFORMS", "cuda")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")

import cma
import numpy as np

from src.doom_real_training import (
    PopulationEvaluator,
    checked_population_records,
    make_population_policy,
    parameter_fingerprint,
)
from src.doom_reference import checked_reference_arrays, load_author_models
from src.doom_reference_env import make_reference_env


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def verify_frozen(settings):
    for group in ("input_sha256", "source_sha256"):
        for path, fingerprint in settings[group].items():
            if digest(path) != fingerprint:
                raise ValueError(f"Frozen {group} changed: {path}")
    if digest(settings["protocol"]) != settings["protocol_sha256"]:
        raise ValueError("Preregistered protocol changed")


def save_policy(path, params, generation, score):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez(
            stream,
            params=np.asarray(params, dtype=np.float64),
            generation=generation,
            score=np.nan if score is None else score,
            type="reference_linear",
            state_mode="ch",
            posterior_sampling=True,
            canonical_actions=False,
            training_method="direct_real_survival_cma",
        )
    temporary.replace(path)


def real_cohort(evaluator, directory, name, parameters, seeds, settings_sha256):
    """Recover valid completed pairs; never overwrite a completed cohort."""
    output = directory / f"{name}.json"
    partial_output = directory / f"{name}.partial.json"
    expected = dict(
        settings_sha256=settings_sha256,
        parameters_sha256=[parameter_fingerprint(p) for p in parameters],
        expected_seeds=seeds,
        cohort=name,
    )
    rows = []
    if output.exists() or partial_output.exists():
        saved = json.loads((output if output.exists() else partial_output).read_text())
        if any(saved[key] != value for key, value in expected.items()):
            raise ValueError("Saved real cohort belongs to different inputs")
        rows = checked_population_records(
            saved["episodes_detail"], len(parameters), seeds, complete=output.exists()
        )
        if output.exists():
            means = [
                float(
                    np.mean([r["survival_steps"] for r in rows if r["candidate"] == i])
                )
                for i in range(len(parameters))
            ]
            if saved["candidate_means"] != means:
                raise ValueError("Saved fitness means differ from raw games")
            return np.asarray(means)

    def record(row):
        rows.append(row)
        checked_population_records(rows, len(parameters), seeds)
        atomic_json(partial_output, dict(expected, episodes_detail=rows))
        print(
            json.dumps(
                dict(
                    cohort=name, completed=len(rows), total=len(parameters) * len(seeds)
                )
            ),
            flush=True,
        )

    evaluator.evaluate(parameters, seeds, completed=rows, on_episode=record)
    rows = checked_population_records(rows, len(parameters), seeds, complete=True)
    means = [
        float(np.mean([r["survival_steps"] for r in rows if r["candidate"] == i]))
        for i in range(len(parameters))
    ]
    with output.open("x") as stream:
        stream.write(
            json.dumps(
                dict(expected, episodes_detail=rows, candidate_means=means), indent=2
            )
            + "\n"
        )
    return np.asarray(means)


def run_search(settings, evaluator, initial, *, resume=False):
    """Journal CMA before games and after tell; recover without drawing twice."""
    args = settings["arguments"]
    output = Path(args["output"])
    last = output.with_name(output.stem + ".last.npz")
    metadata = Path(str(output) + ".json")
    pointer = Path(str(output) + ".resume.json")
    history = Path(str(output) + ".history.jsonl")
    directory = Path(str(output) + ".search")
    settings_text = json.dumps(settings, indent=2) + "\n"
    settings_sha256 = hashlib.sha256(settings_text.encode()).hexdigest()
    if resume:
        saved = json.loads(pointer.read_text())
        if saved["complete"]:
            raise ValueError("Completed search must not be resumed")
        if metadata.read_text() != settings_text:
            raise ValueError("Resume settings/source/runtime differ")
        state_path = Path(saved["state_path"])
        if digest(state_path) != saved["state_sha256"]:
            raise ValueError("Saved CMA state fingerprint differs")
        state = pickle.loads(state_path.read_bytes())
        if state["settings_sha256"] != settings_sha256:
            raise ValueError("CMA state belongs to different settings")
        np.random.set_state(state["numpy_random_state"])
    else:
        if any(
            path.exists()
            for path in (output, last, metadata, pointer, history, directory)
        ):
            raise FileExistsError("Preserve previous search; use a new output")
        output.parent.mkdir(parents=True, exist_ok=True)
        directory.mkdir()
        (directory / "games").mkdir()
        metadata.write_text(settings_text)
        optimizer = cma.CMAEvolutionStrategy(
            np.asarray(initial, dtype=np.float64),
            args["sigma"],
            {"popsize": args["pop_size"], "seed": args["seed"], "verbose": -9},
        )
        state = dict(
            settings_sha256=settings_sha256,
            optimizer=optimizer,
            pending=None,
            phase="baseline",
            best_parameters=np.asarray(initial, dtype=np.float64),
            best_generation=0,
            best_score=None,
            last_score=None,
            history=[],
            numpy_random_state=np.random.get_state(),
            serial=-1,
        )

    def save():
        state["numpy_random_state"] = np.random.get_state()
        # A crash before pointer replacement leaves only an uncommitted journal
        # file. Preserve it and choose a fresh filename when recovering.
        state["serial"] += 1
        state_path = directory / f"state_{state['serial']:04d}.pkl"
        while state_path.exists():
            state["serial"] += 1
            state_path = directory / f"state_{state['serial']:04d}.pkl"
        with state_path.open("xb") as stream:
            pickle.dump(state, stream)
        save_policy(
            output,
            state["best_parameters"],
            state["best_generation"],
            state["best_score"],
        )
        save_policy(
            last,
            state["optimizer"].mean,
            state["optimizer"].countiter,
            state["last_score"],
        )
        temporary = history.with_name(history.name + ".tmp")
        temporary.write_text(
            "".join(json.dumps(row) + "\n" for row in state["history"])
        )
        temporary.replace(history)
        atomic_json(
            pointer,
            dict(
                generation=state["optimizer"].countiter,
                best_generation=state["best_generation"],
                best_real_training_holdout_score=state["best_score"],
                best_sha256=digest(output),
                last_sha256=digest(last),
                state_path=str(state_path),
                state_sha256=digest(state_path),
                phase=state["phase"],
                pending_generation=state["optimizer"].countiter + 1
                if state["pending"] is not None
                else None,
                complete=state["phase"] == "complete",
                training_method="direct_real_survival_cma",
                goal_completion_unproven=True,
            ),
        )

    if not resume:
        save()
    holdout_seeds = list(
        range(args["holdout_seed"], args["holdout_seed"] + args["holdout_games"])
    )
    while state["phase"] != "complete":
        verify_frozen(settings)
        optimizer = state["optimizer"]
        generation = optimizer.countiter
        if state["phase"] == "baseline":
            state["best_score"] = float(
                real_cohort(
                    evaluator,
                    directory / "games",
                    "g000_holdout",
                    state["best_parameters"][None],
                    holdout_seeds,
                    settings_sha256,
                )[0]
            )
            state["last_score"] = state["best_score"]
            state["history"].append(
                dict(generation=0, real_training_holdout_score=state["best_score"])
            )
            state["phase"] = "ask"
        elif state["phase"] == "ask":
            if generation == args["generations"]:
                state["phase"] = "complete"
            else:
                state["pending"] = np.asarray(optimizer.ask(), dtype=np.float64)
                state["phase"] = "fitness"
        elif state["phase"] == "fitness":
            next_generation = generation + 1
            first_seed = args["fitness_seed"] + generation * args["fitness_games"]
            seeds = list(range(first_seed, first_seed + args["fitness_games"]))
            rewards = real_cohort(
                evaluator,
                directory / "games",
                f"g{next_generation:03d}_fitness",
                state["pending"],
                seeds,
                settings_sha256,
            )
            optimizer.tell(state["pending"], -rewards)
            state["pending"] = None
            state["history"].append(
                dict(
                    generation=next_generation,
                    population_mean=float(rewards.mean()),
                    population_best=float(rewards.max()),
                    fitness_seed_range=[seeds[0], seeds[-1]],
                )
            )
            state["last_score"] = None
            state["phase"] = (
                "holdout"
                if (
                    next_generation % args["validate_every"] == 0
                    or next_generation == args["generations"]
                )
                else "ask"
            )
        elif state["phase"] == "holdout":
            score = float(
                real_cohort(
                    evaluator,
                    directory / "games",
                    f"g{generation:03d}_holdout",
                    optimizer.mean[None],
                    holdout_seeds,
                    settings_sha256,
                )[0]
            )
            state["last_score"] = score
            state["history"][-1]["real_training_holdout_score"] = score
            if score > state["best_score"]:
                state["best_score"] = score
                state["best_parameters"] = optimizer.mean.copy()
                state["best_generation"] = generation
            state["phase"] = "ask"
        else:
            raise ValueError("Unknown journal phase")
        save()
        print(
            json.dumps(
                dict(
                    generation=state["optimizer"].countiter,
                    phase=state["phase"],
                    best_generation=state["best_generation"],
                    best_real_training_holdout_score=state["best_score"],
                    goal_completion_unproven=True,
                )
            ),
            flush=True,
        )
    return json.loads(pointer.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--resume", action="store_true")
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
        training["generations"] > 16
        or not 2 <= training["pop_size"] <= 16
        or training["fitness_games"] > 4
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
        training_method="direct_real_survival_cma",
        selection="Training holdout selects mean checkpoints; fresh real validation separately chooses policies for tests",
        common_random_numbers="Every candidate sees the same explicit four fitness seeds in each generation; new seeds next generation",
        imported_public_world=True,
        own_world_model_training=False,
        goal_completion_unproven=True,
    )
    verify_frozen(settings)
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
        payload, _ = checked_reference_arrays(training["reference_dir"])
        initial = np.asarray(payload["controller.json"][0], dtype=np.float64)
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
