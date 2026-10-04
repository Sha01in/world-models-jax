"""Journal a bounded three-gain CMA search using original real-game inference."""

import hashlib
import json
from pathlib import Path
import pickle

import cma
import numpy as np

from scripts.tools.train_doom_reference_real import (
    atomic_json,
    digest,
    real_cohort,
    verify_frozen,
)
from src.doom_controller_gains import (
    FEATURE_BLOCK_SIZES,
    MIN_GAIN,
    MAX_GAIN,
    controller_from_log_gains,
)
from src.doom_real_training import parameter_fingerprint

METHOD = "direct_real_survival_subspace_cma"
SEARCH_SPACE = dict(
    kind="log_feature_gains",
    blocks=list(FEATURE_BLOCK_SIZES),
    gain_bounds=[MIN_GAIN, MAX_GAIN],
    dimensions=3,
)


def mean_log_gains(optimizer):
    """Convert CMA's genotype mean to its bounded phenotype before projecting."""
    return np.asarray(optimizer.to_phenotype(optimizer.mean), dtype=np.float64)


def save_subspace_policy(path, initial, log_gains, generation, score):
    params = controller_from_log_gains(initial, log_gains)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez(
            stream,
            params=params,
            log_gains=np.asarray(log_gains, dtype=np.float64),
            feature_block_sizes=np.asarray(FEATURE_BLOCK_SIZES),
            initializer_parameters_sha256=parameter_fingerprint(initial),
            generation=generation,
            score=np.nan if score is None else score,
            type="reference_linear",
            state_mode="ch",
            posterior_sampling=True,
            canonical_actions=False,
            training_method=METHOD,
        )
    temporary.replace(path)


def run_subspace_search(settings, evaluator, initial, *, resume=False):
    """Journal CMA before games and after tell; recover without drawing twice."""
    initial = np.asarray(initial)
    controller_from_log_gains(initial, np.zeros(3))
    if settings["initializer_parameters_sha256"] != parameter_fingerprint(initial):
        raise ValueError("Frozen subspace initializer differs")
    if (
        settings["training_method"] != METHOD
        or settings["search_space"] != SEARCH_SPACE
    ):
        raise ValueError("Registered subspace mapping differs")
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
            np.zeros(3),
            args["sigma"],
            {
                "popsize": args["pop_size"],
                "seed": args["seed"],
                "verbose": -9,
                "bounds": [float(np.log(MIN_GAIN)), float(np.log(MAX_GAIN))],
            },
        )
        state = dict(
            settings_sha256=settings_sha256,
            optimizer=optimizer,
            pending=None,
            phase="baseline",
            best_log_gains=np.zeros(3),
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
        save_subspace_policy(
            output,
            initial,
            state["best_log_gains"],
            state["best_generation"],
            state["best_score"],
        )
        save_subspace_policy(
            last,
            initial,
            mean_log_gains(state["optimizer"]),
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
                training_method=METHOD,
                search_space=SEARCH_SPACE,
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
                    controller_from_log_gains(initial, state["best_log_gains"])[None],
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
                np.stack(
                    [
                        controller_from_log_gains(initial, point)
                        for point in state["pending"]
                    ]
                ),
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
                    controller_from_log_gains(initial, mean_log_gains(optimizer))[None],
                    holdout_seeds,
                    settings_sha256,
                )[0]
            )
            state["last_score"] = score
            state["history"][-1]["real_training_holdout_score"] = score
            if score > state["best_score"]:
                state["best_score"] = score
                state["best_log_gains"] = mean_log_gains(optimizer)
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
