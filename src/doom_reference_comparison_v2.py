"""Fresh real-validation gates for direct-real controller refinement.

This module never starts an environment. Selection uses complete validation
records; reserved reports are read only after an immutable choice is audited.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.tools.summarize_doom_reference_results import read_checked_report
from src.doom_reference_results import checked_survival, paired_survival_summary
from src.doom_reference_selection import reference_validation_decision
from src.episode_statistics import episode_bootstrap


WORLD_FIELDS = (
    "reference_commit",
    "preprocessing",
    "difficulty",
    "episode_start_time",
    "episode_timeout",
    "threshold",
    "world_precision",
    "posterior_and_controller_precision",
    "rng",
    "package_versions",
    "source_sha256",
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def recorded_seed_audit(
    proposed,
    *,
    exclude=(),
    artifact_directory="artifacts",
    checkpoint_directory="checkpoints",
):
    """Include nested training/evaluation records when checking fresh seeds."""
    from artifacts.audit_vision_round3_real_seeds import check_recorded_real_seeds

    artifact_directory = Path(artifact_directory).resolve()
    exclusions = {Path(path).resolve() for path in exclude}
    result = check_recorded_real_seeds(
        proposed, exclude=exclude, directory=artifact_directory
    )
    nested = []
    for directory in (artifact_directory, Path(checkpoint_directory).resolve()):
        for path in directory.rglob("*.json"):
            if path.parent == artifact_directory or path.resolve() in exclusions:
                continue
            value = read(path)
            rows = value.get("episodes_detail") if isinstance(value, dict) else None
            if rows:
                used = {int(row["seed"]) for row in rows}
                nested.append(
                    dict(
                        path=str(path),
                        recorded_game_pairs=len(rows),
                        seed_min=min(used),
                        seed_max=max(used),
                    )
                )
                for name, wanted in proposed.items():
                    result["overlap"][name] = sorted(
                        set(result["overlap"][name]) | (set(wanted) & used)
                    )
    result["nested_sources"] = nested
    return result


def identity(params):
    return hashlib.sha256(np.asarray(params, dtype=np.float64).tobytes()).hexdigest()


def verify_protocol(protocol):
    for group in ("frozen_inputs", "frozen_source"):
        if any(
            digest(path) != fingerprint for path, fingerprint in protocol[group].items()
        ):
            raise ValueError(f"Registered {group} changed")
    if protocol["validation_games"] != 80 or protocol["test_games"] != 100:
        raise ValueError("Complete80-game validation and100-game test required")
    for name, count in (("validation", 80), ("test", 100)):
        first, last = protocol[name + "_seed_range"]
        if last - first + 1 != count:
            raise ValueError("Registered cohort range/count differs")


def archive_info(path, protocol, *, direct=False):
    path = Path(path)
    base = (
        path.with_name(path.name.removesuffix(".last.npz") + ".npz")
        if path.name.endswith(".last.npz")
        else path
    )
    metadata_path = Path(str(base) + ".json")
    resume_path = Path(str(base) + ".resume.json")
    metadata, resume = read(metadata_path), read(resume_path)
    key = "best_sha256" if path == base else "last_sha256"
    expected_generation = (
        resume["best_generation"] if path == base else resume["generation"]
    )
    if not resume["complete"] or digest(path) != resume[key]:
        raise ValueError("Complete frozen controller search required")
    if (
        metadata["reference_commit"] != protocol["reference_commit"]
        or metadata["source_sha256"]["src/doom_reference.py"]
        != protocol["frozen_source"]["src/doom_reference.py"]
    ):
        raise ValueError("Controller world/conversion differs")
    for name in ("vae.json", "rnn.json"):
        trained = str(Path(metadata["arguments"]["reference_dir"]) / name)
        expected = str(Path(protocol["arguments"]["reference_dir"]) / name)
        if metadata["input_sha256"][trained] != protocol["frozen_inputs"][expected]:
            raise ValueError("Controller latent world differs")
    with np.load(path, allow_pickle=False) as archive:
        params = archive["params"].copy()
        generation = int(archive["generation"])
        if (
            params.shape != (1088,)
            or params.dtype != np.float64
            or not np.isfinite(params).all()
            or generation != expected_generation
            or str(archive["type"]) != "reference_linear"
            or str(archive["state_mode"]) != "ch"
            or not bool(archive["posterior_sampling"])
            or bool(archive["canonical_actions"])
        ):
            raise ValueError("Controller weights/generation/architecture flags differ")
        if direct and (
            str(archive["training_method"]) != "direct_real_survival_cma"
            or metadata["training_method"] != "direct_real_survival_cma"
            or metadata["arguments"]["temperature"] is not None
            or resume["generation"] != protocol["arguments"]["generations"]
        ):
            raise ValueError("Candidate is not the registered direct-real refinement")
    if direct:
        if (
            metadata["arguments"] != protocol["arguments"]
            or metadata["input_sha256"] != protocol["frozen_inputs"]
            or metadata["source_sha256"] != protocol["frozen_source"]
            or digest(resume["state_path"]) != resume["state_sha256"]
        ):
            raise ValueError("Direct-real capsule changed")
    return dict(
        params=params,
        generation=generation,
        temperature=metadata["arguments"]["temperature"],
        files={str(p): digest(p) for p in (path, metadata_path, resume_path)},
    )


def failed_identities(protocol):
    failures = set()
    for path, row in protocol["previous_failed_candidates"].items():
        if (
            digest(path) != row["checkpoint_sha256"]
            or digest(row["source_frozen_selection"])
            != row["source_frozen_selection_sha256"]
        ):
            raise ValueError("Previously failed policy evidence changed")
        with np.load(path, allow_pickle=False) as archive:
            if identity(archive["params"]) != row["parameters_sha256"]:
                raise ValueError("Previously failed raw policy changed")
        failures.add(row["parameters_sha256"])
    return failures


def public_parameters(protocol):
    path = Path(protocol["arguments"]["reference_dir"]) / "controller.json"
    params = np.asarray(read(path)[0], dtype=np.float64)
    if (
        params.shape != (1088,)
        or not np.isfinite(params).all()
        or identity(params) != protocol["public_parameters_sha256"]
    ):
        raise ValueError("Public control raw parameters changed")
    return params


def validation_cases(protocol, directory, prefix):
    """Register controls first, then distinct eligible best/final candidates."""
    verify_protocol(protocol)
    failures = failed_identities(protocol)
    best = Path(protocol["arguments"]["output"])
    definitions = [
        (row["name"], row["controller"], "control")
        for row in protocol["validation_controls"]
    ]
    definitions.extend(
        (
            ("val_direct_best", str(best), "candidate"),
            (
                "val_direct_last",
                str(best.with_name(best.stem + ".last.npz")),
                "candidate",
            ),
        )
    )
    cases, seen = [], set()
    for name, actor, kind in definitions:
        if actor is None:
            params, generation = public_parameters(protocol), 0
        else:
            info = archive_info(actor, protocol, direct=kind == "candidate")
            params, generation = info["params"], info["generation"]
            if kind == "control":
                control = next(
                    row
                    for row in protocol["validation_controls"]
                    if row["controller"] == actor
                )
                if digest(actor) != control["controller_sha256"]:
                    raise ValueError("Registered control changed")
        fingerprint = identity(params)
        if fingerprint in seen or (kind == "candidate" and fingerprint in failures):
            continue
        if kind == "candidate" and generation < 1:
            raise ValueError("Unupdated policy cannot be a new candidate")
        seen.add(fingerprint)
        cases.append(
            dict(
                name=name,
                controller=actor,
                kind=kind,
                inference="posterior",
                parameters_sha256=fingerprint,
                seed=protocol["validation_seed_range"][0],
                episodes=80,
                role="validation",
                report=str(Path(directory) / f"{prefix}_{name}.json"),
            )
        )
    if (
        len(cases) > protocol["maximum_validation_policies"]
        or len(cases) < 2
        or [row["controller"] for row in cases if row["kind"] == "control"]
        != [row["controller"] for row in protocol["validation_controls"]]
    ):
        raise ValueError(
            "Both distinct registered controls and bounded candidates required"
        )
    return cases


def checked_report(protocol, case):
    report = read_checked_report(case["report"], case, role=case["role"])
    check_report_header(protocol, case, report)
    if report["workers"] != 8:
        raise ValueError("Expected eight real-game workers")
    return report


def check_report_header(protocol, case, report):
    template_path = protocol["environment_template_report"]
    if digest(template_path) != protocol["frozen_inputs"][template_path]:
        raise ValueError("Reference environment template changed")
    template = read(template_path)["protocol"]
    recorded = report["protocol"]
    if any(recorded[key] != template[key] for key in WORLD_FIELDS) or any(
        recorded["input_sha256"].get(path) != sha
        for path, sha in template["input_sha256"].items()
    ):
        raise ValueError("Real evaluation world/environment/runtime differs")
    provenance = recorded["controller_provenance"]
    if not provenance["imported_public_world"]:
        raise ValueError("Expected unchanged imported public world")
    if case["controller"] is None:
        params = public_parameters(protocol)
        if provenance["own_policy_update"]:
            raise ValueError("Public control was mislabeled as a new policy")
    else:
        direct = case["kind"] == "candidate"
        info = archive_info(case["controller"], protocol, direct=direct)
        params = info["params"]
        if (
            provenance["generation"] != info["generation"]
            or provenance["dream_temperature"] != info["temperature"]
            or not provenance["own_policy_update"]
            or any(
                recorded["input_sha256"].get(path) != sha
                for path, sha in info["files"].items()
            )
        ):
            raise ValueError("Recorded controller provenance differs")
    if identity(params) != case["parameters_sha256"]:
        raise ValueError("Recorded raw policy identity differs")


def checked_partial(protocol, case, path):
    snapshot = read(path)
    expected = list(range(case["seed"], case["seed"] + case["episodes"]))
    if snapshot["expected_seeds"] != expected:
        raise ValueError("Partial report belongs to a different seed cohort")
    rows = snapshot["episodes_detail"]
    seeds = [row["seed"] for row in rows]
    if seeds != sorted(set(seeds)) or not set(seeds) <= set(expected):
        raise ValueError("Partial report contains duplicate/unexpected games")
    check_report_header(protocol, case, snapshot)
    for group in ("input_sha256", "source_sha256"):
        if any(
            digest(path) != sha for path, sha in snapshot["protocol"][group].items()
        ):
            raise ValueError("Partial report fingerprints changed")
    if rows:
        scores = np.asarray([row["survival_steps"] for row in rows], dtype=np.float64)
        mini = dict(
            protocol=snapshot["protocol"],
            episodes_detail=rows,
            episodes=len(rows),
            mean=float(scores.mean()),
            std=float(scores.std()),
            deaths=sum(row["terminated"] for row in rows),
            timeouts=sum(row["truncated"] for row in rows),
        )
        checked_survival(mini, seeds, role=case["role"])
    if snapshot["protocol"]["controller_provenance"].get("controller") != case[
        "controller"
    ] or snapshot["protocol"]["inference_posterior_sampling"] != (
        case["inference"] == "posterior"
    ):
        raise ValueError("Partial report actor/inference differs")
    return snapshot


def select_complete_validation(protocol, cases):
    reports = [checked_report(protocol, case) for case in cases]
    decision = reference_validation_decision(cases, reports)
    controls = [i for i, case in enumerate(cases) if case["kind"] == "control"]
    control = max(controls, key=lambda index: reports[index]["mean"])
    decision["paired_control"] = cases[control]
    return decision, reports


def independent_frozen_selection(protocol, frozen, *, resamples=50000):
    """Reconstruct eligibility and select by raw means, without the selector."""
    verify_protocol(protocol)
    cases = frozen["validation_tasks"]
    controls = protocol["validation_controls"]
    if [case["controller"] for case in cases[: len(controls)]] != [
        row["controller"] for row in controls
    ] or any(case["kind"] != "control" for case in cases[: len(controls)]):
        raise ValueError("Frozen control order differs")
    control_ids = set()
    for row in controls:
        params = (
            public_parameters(protocol)
            if row["controller"] is None
            else archive_info(row["controller"], protocol)["params"]
        )
        control_ids.add(identity(params))
    failures = failed_identities(protocol)
    best = Path(protocol["arguments"]["output"])
    expected_new, seen = [], set(control_ids)
    for actor in (best, best.with_name(best.stem + ".last.npz")):
        info = archive_info(actor, protocol, direct=True)
        fingerprint = identity(info["params"])
        if fingerprint in seen or fingerprint in failures:
            continue
        if info["generation"] < 1:
            raise ValueError("Frozen candidate was not updated")
        expected_new.append((str(actor), fingerprint))
        seen.add(fingerprint)
    actual_new = [
        (row["controller"], row["parameters_sha256"]) for row in cases[len(controls) :]
    ]
    if (
        actual_new != expected_new
        or any(row["kind"] != "candidate" for row in cases[len(controls) :])
        or len(cases) > protocol["maximum_validation_policies"]
    ):
        raise ValueError("Frozen candidate eligibility is incomplete or altered")
    reports, values = [], []
    for case in cases:
        if (
            case["seed"] != protocol["validation_seed_range"][0]
            or case["episodes"] != 80
            or case["role"] != "validation"
            or case["inference"] != "posterior"
            or digest(case["report"])
            != frozen["validation_report_sha256"][case["report"]]
        ):
            raise ValueError("Frozen complete validation cohort/hash differs")
        report = checked_report(protocol, case)
        reports.append(report)
        values.append(
            checked_survival(
                report, range(case["seed"], case["seed"] + 80), role="validation"
            )
        )
    means = [float(scores.mean()) for scores in values]
    winner = max(range(len(cases)), key=lambda index: means[index])
    control = max(range(len(controls)), key=lambda index: means[index])
    new_won = cases[winner]["kind"] == "candidate"
    if (
        frozen["selected"] != cases[winner]
        or frozen["paired_control"] != cases[control]
        or frozen["new_policy_won"] != new_won
        or frozen["validation_scores"]
        != {case["name"]: mean for case, mean in zip(cases, means, strict=True)}
    ):
        raise ValueError("Frozen choice differs from complete raw validation")
    statistics = episode_bootstrap(
        np.arange(80),
        lambda indices: [
            *[scores[indices].mean() for scores in values],
            *[(scores[indices] - values[control][indices]).mean() for scores in values],
        ],
        resamples=resamples,
        seed=74181,
    )
    summaries = [
        dict(
            name=case["name"],
            mean=means[i],
            std=float(values[i].std()),
            mean95=statistics["intervals"][i],
            minus_paired_control=means[i] - means[control],
            paired_gain95=statistics["intervals"][i + len(cases)],
            deaths=reports[i]["deaths"],
            timeouts=reports[i]["timeouts"],
        )
        for i, case in enumerate(cases)
    ]
    return dict(
        selected=cases[winner],
        paired_control=cases[control],
        new_policy_won=new_won,
        validation=summaries,
        validation_selection_recomputed=True,
        all_validation_records_verified=True,
        complete_candidate_eligibility_reconstructed=True,
        goal_completion_unproven=True,
    )


def test_cases(protocol, frozen, directory, prefix):
    if not frozen["new_policy_won"]:
        return []
    cases = []
    for name, source in (
        ("test_selected", frozen["selected"]),
        ("test_control", frozen["paired_control"]),
    ):
        cases.append(
            dict(
                source,
                name=name,
                seed=protocol["test_seed_range"][0],
                episodes=100,
                role="test",
                report=str(Path(directory) / f"{prefix}_{name}.json"),
            )
        )
    return cases


def completed_pair(protocol, frozen, cases, *, resamples=50000):
    expected = test_cases(
        protocol,
        frozen,
        Path(cases[0]["report"]).parent,
        Path(cases[0]["report"]).name.removesuffix("_test_selected.json"),
    )
    if cases != expected or len(cases) != 2:
        raise ValueError("Reserved pair differs from frozen validation choice")
    reports = [checked_report(protocol, case) for case in cases]
    return paired_survival_summary(
        reports[0],
        reports[1],
        range(protocol["test_seed_range"][0], protocol["test_seed_range"][1] + 1),
        seed=74182,
        resamples=resamples,
    )
