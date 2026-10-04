"""Independently reconstruct a completed direct-real CMA search on CPU."""
# ruff: noqa: E402 -- force CPU before importing the training helpers.

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path
import pickle
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cma
import numpy as np

from scripts.tools.train_doom_reference_real import digest, verify_frozen
from src.doom_real_training import checked_population_records, parameter_fingerprint


def audit_search_capsule(output, initial):
    output = Path(output)
    metadata = Path(str(output) + ".json")
    settings = json.loads(metadata.read_text())
    verify_frozen(settings)
    args = settings["arguments"]
    resume_path = Path(str(output) + ".resume.json")
    resume = json.loads(resume_path.read_text())
    if not resume["complete"] or resume["generation"] != args["generations"]:
        raise ValueError("Completed bounded search required")
    state_path = Path(resume["state_path"])
    if digest(state_path) != resume["state_sha256"]:
        raise ValueError("Final optimizer/RNG state hash differs")
    state = pickle.loads(state_path.read_bytes())
    settings_hash = digest(metadata)
    if (
        state["settings_sha256"] != settings_hash
        or state["phase"] != "complete"
        or state["pending"] is not None
    ):
        raise ValueError("Incomplete or mismatched final CMA state")
    count = 0
    files = {str(path): digest(path) for path in (metadata, resume_path, state_path)}

    def cohort(name, parameters, seeds):
        nonlocal count
        path = Path(str(output) + ".search/games") / f"{name}.json"
        report = json.loads(path.read_text())
        if (
            report["settings_sha256"] != settings_hash
            or report["parameters_sha256"]
            != [parameter_fingerprint(p) for p in parameters]
            or report["expected_seeds"] != seeds
            or report["cohort"] != name
        ):
            raise ValueError("Raw game cohort differs from reconstructed weights/seeds")
        rows = checked_population_records(
            report["episodes_detail"], len(parameters), seeds, complete=True
        )
        means = [
            float(np.mean([r["survival_steps"] for r in rows if r["candidate"] == i]))
            for i in range(len(parameters))
        ]
        if report["candidate_means"] != means:
            raise ValueError("Fitness summary differs from complete raw games")
        count += len(rows)
        files[str(path)] = digest(path)
        return np.asarray(means)

    holdouts = list(
        range(args["holdout_seed"], args["holdout_seed"] + args["holdout_games"])
    )
    best_params = np.asarray(initial, dtype=np.float64)
    best_score = float(cohort("g000_holdout", best_params[None], holdouts)[0])
    best_generation = 0
    expected_history = [dict(generation=0, real_training_holdout_score=best_score)]
    saved_rng = np.random.get_state()
    try:
        optimizer = cma.CMAEvolutionStrategy(
            initial,
            args["sigma"],
            {"popsize": args["pop_size"], "seed": args["seed"], "verbose": -9},
        )
        final_score = None
        for generation in range(1, args["generations"] + 1):
            population = np.asarray(optimizer.ask(), dtype=np.float64)
            first_seed = args["fitness_seed"] + (generation - 1) * args["fitness_games"]
            seeds = list(range(first_seed, first_seed + args["fitness_games"]))
            rewards = cohort(f"g{generation:03d}_fitness", population, seeds)
            optimizer.tell(population, -rewards)
            row = dict(
                generation=generation,
                population_mean=float(rewards.mean()),
                population_best=float(rewards.max()),
                fitness_seed_range=[seeds[0], seeds[-1]],
            )
            if (
                generation % args["validate_every"] == 0
                or generation == args["generations"]
            ):
                score = float(
                    cohort(
                        f"g{generation:03d}_holdout", optimizer.mean[None], holdouts
                    )[0]
                )
                row["real_training_holdout_score"] = score
                final_score = score
                if score > best_score:
                    best_score, best_generation, best_params = (
                        score,
                        generation,
                        optimizer.mean.copy(),
                    )
            expected_history.append(row)
        history_path = Path(str(output) + ".history.jsonl")
        if [
            json.loads(row) for row in history_path.read_text().splitlines()
        ] != expected_history or state["history"] != expected_history:
            raise ValueError("History differs from independently reconstructed games")
        files[str(history_path)] = digest(history_path)
        if state["optimizer"].countiter != args["generations"]:
            raise ValueError("Optimizer generation count differs")
        np.testing.assert_array_equal(state["optimizer"].mean, optimizer.mean)
        np.testing.assert_array_equal(state["best_parameters"], best_params)
        for key, expected in (
            ("best_generation", best_generation),
            ("best_score", best_score),
            ("last_score", final_score),
        ):
            if state[key] != expected:
                raise ValueError(f"Saved {key} differs")
        next_population = np.asarray(optimizer.ask())
        np.random.set_state(state["numpy_random_state"])
        np.testing.assert_array_equal(
            np.asarray(state["optimizer"].ask()), next_population
        )
        for path, weights, generation, score, key in (
            (output, best_params, best_generation, best_score, "best_sha256"),
            (
                output.with_name(output.stem + ".last.npz"),
                optimizer.mean,
                args["generations"],
                final_score,
                "last_sha256",
            ),
        ):
            if digest(path) != resume[key]:
                raise ValueError("Saved policy hash differs")
            with np.load(path, allow_pickle=False) as archive:
                np.testing.assert_array_equal(archive["params"], weights)
                flags = dict(
                    type="reference_linear",
                    state_mode="ch",
                    training_method="direct_real_survival_cma",
                )
                if (
                    archive["params"].dtype != np.float64
                    or int(archive["generation"]) != generation
                    or float(archive["score"]) != score
                    or any(str(archive[k]) != v for k, v in flags.items())
                    or not bool(archive["posterior_sampling"])
                    or bool(archive["canonical_actions"])
                ):
                    raise ValueError(
                        "Policy architecture/provenance/score flags differ"
                    )
            files[str(path)] = digest(path)
        if (
            resume["best_generation"] != best_generation
            or resume["best_real_training_holdout_score"] != best_score
        ):
            raise ValueError("Resume selection differs")
    finally:
        np.random.set_state(saved_rng)
    return dict(
        checked_at=datetime.now(timezone.utc).isoformat(),
        generations=args["generations"],
        raw_training_games=count,
        best_generation=best_generation,
        best_real_training_holdout_score=best_score,
        population_weights_optimizer_history_and_next_rng_reconstructed=True,
        all_raw_game_pairs_verified=True,
        files=files,
        goal_completion_unproven=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    protocol = json.loads(Path(args.protocol).read_text())
    metadata = json.loads(
        Path(str(protocol["arguments"]["output"]) + ".json").read_text()
    )
    if (
        metadata["arguments"] != protocol["arguments"]
        or metadata["protocol_sha256"] != digest(args.protocol)
        or metadata["source_sha256"] != protocol["frozen_source"]
        or metadata["input_sha256"] != protocol["frozen_inputs"]
    ):
        raise ValueError("Completed training metadata differs from registration")
    for path, fingerprint in {
        **protocol["frozen_inputs"],
        **protocol["frozen_source"],
    }.items():
        if digest(path) != fingerprint:
            raise ValueError(f"Registered fingerprint changed: {path}")
    if {name: version(name) for name in protocol["package_versions"]} != protocol[
        "package_versions"
    ]:
        raise ValueError("Runtime differs from registration")
    selection = json.loads(Path(protocol["initializer_selection_frozen"]).read_text())
    audit = json.loads(Path(protocol["initializer_selection_cpu_audit"]).read_text())
    if (
        selection["selected"]["controller"] is not None
        or audit["selected"] != selection["selected"]
        or not audit["validation_selection_recomputed"]
        or not audit["all_validation_records_verified"]
    ):
        raise ValueError(
            "Initializer is not the verified validation-selected public control"
        )
    path = Path(protocol["arguments"]["reference_dir"]) / "controller.json"
    initial = np.asarray(json.loads(path.read_text())[0], dtype=np.float64)
    if parameter_fingerprint(initial) != protocol["initializer_parameters_sha256"]:
        raise ValueError("Public initializer changed")
    result = audit_search_capsule(protocol["arguments"]["output"], initial)
    result.update(
        protocol_sha256=digest(args.protocol),
        public_initializer_selected_from_complete_validation_only=True,
        canonical_preserved=True,
        imported_public_world=True,
        own_world_model_training=False,
        current_package_versions=protocol["package_versions"],
    )
    with Path(args.output).open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "generations",
                    "raw_training_games",
                    "best_generation",
                    "best_real_training_holdout_score",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
