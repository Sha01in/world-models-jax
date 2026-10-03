"""CPU-only audit of validation selection and the reserved real-game comparison."""

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.doom_reference_results import checked_survival, paired_survival_summary  # noqa: E402


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_checked_report(path, task, *, role):
    report = json.loads(Path(path).read_text())
    seeds = range(task["seed"], task["seed"] + task["episodes"])
    checked_survival(report, seeds, role=role)
    protocol = report["protocol"]
    if (
        protocol["inference_posterior_sampling"] != (task["inference"] == "posterior")
        or protocol["controller_provenance"].get("controller") != task["controller"]
    ):
        raise ValueError("Recorded inference/controller differs from frozen task")
    for group in ("input_sha256", "source_sha256"):
        if any(digest(path) != sha for path, sha in protocol[group].items()):
            raise ValueError(f"Frozen report fingerprints differ: {group}")
    return report


def audit_result(result_path, *, resamples=50000, protocol_path=None):
    result_path = Path(result_path)
    result = json.loads(result_path.read_text())
    freeze_path = Path(result["frozen_selection"])
    if digest(freeze_path) != result["frozen_selection_sha256"]:
        raise ValueError("Frozen selection fingerprint differs")
    freeze = json.loads(freeze_path.read_text())
    tasks = freeze["validation_tasks"]
    expected_test_range = [140000, 140099]
    control_controller = None
    controlled_protocol = None
    if protocol_path is not None:
        protocol_path = Path(protocol_path)
        controlled_protocol = json.loads(protocol_path.read_text())
        if freeze["protocol_sha256"] != digest(protocol_path) or result[
            "protocol_sha256"
        ] != digest(protocol_path):
            raise ValueError("Preregistered comparison protocol changed")
        for group in ("frozen_inputs", "frozen_training_source"):
            if any(digest(p) != sha for p, sha in controlled_protocol[group].items()):
                raise ValueError("Preregistered world/training source changed")
        expected_test_range = controlled_protocol["test_seed_range"]
        if expected_test_range[1] - expected_test_range[0] + 1 != 100:
            raise ValueError("A complete100-game reserved test is required")
        control_controller = controlled_protocol["previous_selected_policy"]
        if (
            digest(control_controller)
            != controlled_protocol["previous_selected_policy_sha256"]
        ):
            raise ValueError("Preregistered paired control changed")
        new_best = Path(controlled_protocol["arguments"]["output"])
        new_last = new_best.with_name(new_best.stem + ".last.npz")
        allowed_candidates = {str(new_best), str(new_last)}
        allowed_controls = {None, control_controller}
        val_first, val_last = controlled_protocol["validation_seed_range"]
        if val_last - val_first + 1 != controlled_protocol["validation_games"]:
            raise ValueError("Preregistered validation count differs from cohort")
        identities = set()
        for task in tasks:
            if (
                task["seed"] != val_first
                or task["episodes"] != controlled_protocol["validation_games"]
                or task["inference"] != "posterior"
            ):
                raise ValueError(
                    "Validation differs from preregistered cohort/inference"
                )
            actor = task["controller"]
            if actor not in allowed_controls | allowed_candidates:
                raise ValueError("Validation actor is outside preregistered candidates")
            expected_kind = "control" if actor in allowed_controls else "candidate"
            if task["kind"] != expected_kind:
                raise ValueError("Validation policy role differs from protocol")
            if actor is None:
                public = (
                    Path(controlled_protocol["arguments"]["reference_dir"])
                    / "controller.json"
                )
                params = np.asarray(json.loads(public.read_text())[0], np.float64)
            else:
                with np.load(actor, allow_pickle=False) as checkpoint:
                    params = np.asarray(checkpoint["params"], np.float64)
            identity = hashlib.sha256(params.tobytes()).hexdigest()
            if identity != task["parameters_sha256"] or identity in identities:
                raise ValueError("Policy identity differs or was not deduplicated")
            identities.add(identity)
        if {
            t["controller"] for t in tasks if t["kind"] == "control"
        } != allowed_controls:
            raise ValueError("Both preregistered controls must be validated")
        control_indices = [i for i, t in enumerate(tasks) if t["kind"] == "control"]
        if control_indices != list(range(len(control_indices))):
            raise ValueError("Preregistered controls must precede candidates for ties")
    validation = []
    for task in tasks:
        path = task["report"]
        if digest(path) != freeze["validation_report_sha256"][path]:
            raise ValueError("Validation report changed after policy selection")
        report = read_checked_report(path, task, role="validation")
        if report["mean"] != freeze["validation_scores"][task["name"]]:
            raise ValueError("Frozen selection score differs from actual validation")
        validation.append(report)
    if not validation:
        raise ValueError("No validation evidence")
    winner = max(range(len(tasks)), key=lambda index: validation[index]["mean"])
    selected_task = tasks[winner]
    if selected_task != freeze["selected"] or selected_task != result["selected"]:
        raise ValueError("Selected policy is not the validation winner")
    for group, report_group in (
        ("frozen_inputs", "input_sha256"),
        ("frozen_source", "source_sha256"),
    ):
        if freeze[group] != validation[winner]["protocol"][report_group]:
            raise ValueError("Selection fingerprints differ from winning validation")
    if (
        freeze["selected_controller_provenance"]
        != validation[winner]["protocol"]["controller_provenance"]
    ):
        raise ValueError("Selected policy provenance differs")
    if freeze["test_seed_range"] != expected_test_range:
        raise ValueError("Unexpected reserved test cohort")
    highest_control_required = controlled_protocol is not None and (
        "highest validation control"
        in controlled_protocol.get("paired_test_control", "")
    )
    if highest_control_required:
        control_indices = [i for i, t in enumerate(tasks) if t["kind"] == "control"]
        paired_index = max(control_indices, key=lambda i: validation[i]["mean"])
        if freeze.get("paired_control") != tasks[paired_index]:
            raise ValueError("Paired control is not the highest validation control")
        control_controller = tasks[paired_index]["controller"]
    elif "paired_control" in freeze:
        raise ValueError("Protocol does not declare validation-selected paired control")
    if controlled_protocol is not None:
        provenance = validation[winner]["protocol"]["controller_provenance"]
        if (
            selected_task["kind"] != "candidate"
            or not provenance["own_policy_update"]
            or provenance["generation"] <= 0
            or provenance["dream_temperature"]
            != controlled_protocol["arguments"]["temperature"]
            or not freeze["new_policy_won"]
        ):
            raise ValueError(
                "Reserved tests require an updated candidate validation winner"
            )
        if datetime.fromisoformat(controlled_protocol["preregistered_at"]) > min(
            datetime.fromisoformat(r["measured_at"]) for r in validation
        ):
            raise ValueError("Comparison protocol postdates validation")
    first_test_seed = expected_test_range[0]
    test_task = dict(selected_task, seed=first_test_seed, episodes=100)
    control_task = dict(
        controller=control_controller,
        inference="posterior",
        seed=first_test_seed,
        episodes=100,
    )
    selected_path, control_path = result["selected_test"], result["control_test"]
    for path, key in (
        (selected_path, "selected_test_sha256"),
        (control_path, "control_test_sha256"),
    ):
        if digest(path) != result[key]:
            raise ValueError("Reserved report fingerprint differs")
    selected = read_checked_report(selected_path, test_task, role="test")
    control = read_checked_report(control_path, control_task, role="test")
    if selected["protocol"]["input_sha256"] != freeze["frozen_inputs"]:
        raise ValueError("Selected test inputs differ from frozen validation policy")
    if selected["protocol"]["source_sha256"] != freeze["frozen_source"]:
        raise ValueError("Selected test source differs from frozen validation")
    frozen_at = datetime.fromisoformat(freeze["frozen_at"])
    if any(
        datetime.fromisoformat(report["measured_at"]) > frozen_at
        for report in validation
    ):
        raise ValueError("Validation report postdates the frozen selection")
    if any(
        datetime.fromisoformat(report["measured_at"]) < frozen_at
        for report in (selected, control)
    ):
        raise ValueError("Reserved report predates frozen selection")
    summary = paired_survival_summary(
        selected,
        control,
        range(first_test_seed, first_test_seed + 100),
        resamples=resamples,
    )
    if (
        result["test_mean"] != summary["selected_mean"]
        or not np.isclose(
            result["test_std"], summary["selected_std"], rtol=1e-12, atol=1e-9
        )
        or result["own_policy_update"] != summary["own_policy_update"]
    ):
        raise ValueError("Supervisor result differs from actual test/provenance")
    return dict(
        **summary,
        result_path=str(result_path),
        result_sha256=digest(result_path),
        selected_report=selected_path,
        selected_report_sha256=digest(selected_path),
        control_report=control_path,
        control_report_sha256=digest(control_path),
        frozen_selection=str(freeze_path),
        frozen_selection_sha256=digest(freeze_path),
        validation=[
            dict(
                name=t["name"],
                mean=r["mean"],
                std=r["std"],
                inference=t["inference"],
                controller=t["controller"],
            )
            for t, r in zip(tasks, validation)
        ],
        selection_recomputed_from_validation_only=True,
        reserved_seed_cohort_verified=True,
        policy_provenance=selected["protocol"]["controller_provenance"],
        protocol_limitations=selected["protocol"]["limitations"],
        completion_still_requires_review=True,
        preregistered_protocol=str(protocol_path)
        if protocol_path is not None
        else None,
        paired_control=freeze.get("paired_control"),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result",
        default="artifacts/doom_reference_round5_real_evaluation_result.json",
    )
    parser.add_argument(
        "--output", default="artifacts/doom_reference_round5_paired_comparison.json"
    )
    parser.add_argument("--protocol", help="Preregistered controlled comparison JSON")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("Preserve existing paired analysis")
    analysis = audit_result(args.result, protocol_path=args.protocol)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        stream.write(json.dumps(analysis, indent=2, allow_nan=False) + "\n")
    print(json.dumps(analysis, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
