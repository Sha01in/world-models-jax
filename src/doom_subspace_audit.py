"""Independent CPU reconstruction of projected controllers and bounded CMA."""

from datetime import datetime, timezone
import json
from pathlib import Path
import pickle

import cma
import numpy as np

from scripts.tools.train_doom_reference_real import digest, verify_frozen
from src.doom_controller_gains import (
    FEATURE_BLOCK_SIZES,
    MIN_GAIN,
    MAX_GAIN,
    controller_from_log_gains,
)
from src.doom_real_training import checked_population_records, parameter_fingerprint
from src.doom_subspace_training import METHOD, SEARCH_SPACE


def mean_log_gains(optimizer):
    """Reconstruct bounded mean separately from the producer helper."""
    return np.asarray(optimizer.to_phenotype(optimizer.mean), dtype=np.float64)


def audit_subspace_capsule(output, initial):
    output = Path(output)
    metadata = Path(str(output) + ".json")
    settings = json.loads(metadata.read_text())
    verify_frozen(settings)
    controller_from_log_gains(initial, np.zeros(3))
    if settings["initializer_parameters_sha256"] != parameter_fingerprint(initial):
        raise ValueError("Frozen subspace initializer differs")
    if (
        settings["training_method"] != METHOD
        or settings["search_space"] != SEARCH_SPACE
    ):
        raise ValueError("Registered subspace mapping differs")
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
    best_gains = np.zeros(3)
    best_params = controller_from_log_gains(initial, best_gains)
    best_score = float(cohort("g000_holdout", best_params[None], holdouts)[0])
    best_generation = 0
    expected_history = [dict(generation=0, real_training_holdout_score=best_score)]
    saved_rng = np.random.get_state()
    try:
        optimizer = cma.CMAEvolutionStrategy(
            np.zeros(3),
            args["sigma"],
            {
                "popsize": args["pop_size"],
                "seed": args["seed"],
                "verbose": -9,
                "bounds": [float(np.log(MIN_GAIN)), float(np.log(MAX_GAIN))],
            },
        )
        final_score = None
        for generation in range(1, args["generations"] + 1):
            population = np.asarray(optimizer.ask(), dtype=np.float64)
            first_seed = args["fitness_seed"] + (generation - 1) * args["fitness_games"]
            seeds = list(range(first_seed, first_seed + args["fitness_games"]))
            rewards = cohort(
                f"g{generation:03d}_fitness",
                np.stack(
                    [controller_from_log_gains(initial, point) for point in population]
                ),
                seeds,
            )
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
                        f"g{generation:03d}_holdout",
                        controller_from_log_gains(initial, mean_log_gains(optimizer))[
                            None
                        ],
                        holdouts,
                    )[0]
                )
                row["real_training_holdout_score"] = score
                final_score = score
                if score > best_score:
                    best_score, best_generation = score, generation
                    best_gains = mean_log_gains(optimizer)
                    best_params = controller_from_log_gains(initial, best_gains)
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
        np.testing.assert_array_equal(state["best_log_gains"], best_gains)
        for key in ("sigma", "countiter", "countevals"):
            if getattr(state["optimizer"], key) != getattr(optimizer, key):
                raise ValueError("Saved optimizer adaptation/count differs")
        np.testing.assert_array_equal(state["optimizer"].sm.C, optimizer.sm.C)
        np.testing.assert_array_equal(state["optimizer"].pc, optimizer.pc)
        np.testing.assert_array_equal(
            state["optimizer"].adapt_sigma.ps, optimizer.adapt_sigma.ps
        )
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
        for path, weights, gains, generation, score, key in (
            (
                output,
                best_params,
                best_gains,
                best_generation,
                best_score,
                "best_sha256",
            ),
            (
                output.with_name(output.stem + ".last.npz"),
                controller_from_log_gains(initial, mean_log_gains(optimizer)),
                mean_log_gains(optimizer),
                args["generations"],
                final_score,
                "last_sha256",
            ),
        ):
            if digest(path) != resume[key]:
                raise ValueError("Saved policy hash differs")
            with np.load(path, allow_pickle=False) as archive:
                np.testing.assert_array_equal(archive["params"], weights)
                np.testing.assert_array_equal(archive["log_gains"], gains)
                np.testing.assert_array_equal(
                    archive["feature_block_sizes"], FEATURE_BLOCK_SIZES
                )
                if str(
                    archive["initializer_parameters_sha256"]
                ) != parameter_fingerprint(initial):
                    raise ValueError("Projected checkpoint initializer differs")
                flags = dict(
                    type="reference_linear",
                    state_mode="ch",
                    training_method=METHOD,
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
        projected1088_weights_and_three_dimensional_optimizer_reconstructed=True,
        training_method=METHOD,
        search_space=SEARCH_SPACE,
        initializer_parameters_sha256=parameter_fingerprint(initial),
        files=files,
        goal_completion_unproven=True,
    )
