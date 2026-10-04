"""CPU reconstruction of direct-real validation choice and paired test results."""
# ruff: noqa: E402 -- force CPU and initialize the workspace before imports.

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.doom_reference_comparison_v2 import (
    completed_pair,
    digest,
    independent_frozen_selection,
    read,
    recorded_seed_audit,
    test_cases,
    verify_protocol,
)


def frozen_choice_audit(protocol_path, freeze_path, *, resamples=50000):
    protocol, frozen = read(protocol_path), read(freeze_path)
    verify_protocol(protocol)
    if (
        frozen["protocol_sha256"] != digest(protocol_path)
        or frozen["test_seed_range"] != protocol["test_seed_range"]
        or digest(frozen["training_audit"]) != frozen["training_audit_sha256"]
    ):
        raise ValueError("Frozen protocol/training audit differs")
    training = read(frozen["training_audit"])
    if (
        training["protocol_sha256"] != digest(protocol_path)
        or not training[
            "population_weights_optimizer_history_and_next_rng_reconstructed"
        ]
        or not training["all_raw_game_pairs_verified"]
        or training["raw_training_games"]
        != protocol["maximum_training_fitness_games"]
        + protocol["maximum_training_holdout_games"]
        or any(digest(path) != sha for path, sha in training["files"].items())
    ):
        raise ValueError(
            "Completed independently reconstructed training capsule changed"
        )
    if any(
        digest(path) != sha for path, sha in frozen["selection_source_sha256"].items()
    ):
        raise ValueError("Frozen real selection/audit source changed")
    if {name: version(name) for name in protocol["package_versions"]} != protocol[
        "package_versions"
    ]:
        raise ValueError("Current runtime differs from registration")
    result = independent_frozen_selection(protocol, frozen, resamples=resamples)
    result.update(
        checked_at=datetime.now(timezone.utc).isoformat(),
        frozen_selection=str(freeze_path),
        frozen_selection_sha256=digest(freeze_path),
        protocol_sha256=digest(protocol_path),
        training_audit_sha256=digest(frozen["training_audit"]),
        parameters_world_source_runtime_and_canonical_verified=True,
        training_method="direct_real_survival_cma",
        imported_public_world=True,
        own_world_model_training=False,
        uncertainty="Whole-game paired validation bootstrap with fixed policies; excludes training and sequential selection uncertainty",
    )
    return result


def real_result_audit(protocol_path, result_path, *, resamples=50000):
    protocol, result = read(protocol_path), read(result_path)
    freeze_path = Path(result["frozen_selection"])
    frozen = read(freeze_path)
    if (
        result["protocol_sha256"] != digest(protocol_path)
        or result["frozen_selection_sha256"] != digest(freeze_path)
        or result["selection_audit_sha256"] != digest(result["selection_audit"])
    ):
        raise ValueError("Real result protocol/freeze/selection-audit hash changed")
    selection = frozen_choice_audit(protocol_path, freeze_path, resamples=resamples)
    previous = read(result["selection_audit"])
    for key in (
        "selected",
        "paired_control",
        "new_policy_won",
        "all_validation_records_verified",
        "complete_candidate_eligibility_reconstructed",
        "frozen_selection_sha256",
        "protocol_sha256",
    ):
        if previous[key] != selection[key]:
            raise ValueError("Prior independent selection audit differs")
    expected = test_cases(
        protocol, frozen, frozen["report_directory"], frozen["report_prefix"]
    )
    if (
        result["selected"] != frozen["selected"]
        or result["paired_control"] != frozen["paired_control"]
        or result["new_policy_won"] != frozen["new_policy_won"]
        or result["test_tasks"] != expected
        or result["reserved_test_consumed"] != bool(expected)
    ):
        raise ValueError("Reserved result does not follow the frozen validation choice")
    output = dict(
        checked_at=datetime.now(timezone.utc).isoformat(),
        result_sha256=digest(result_path),
        frozen_selection_sha256=digest(freeze_path),
        protocol_sha256=digest(protocol_path),
        validation=selection["validation"],
        selected=frozen["selected"],
        paired_control=frozen["paired_control"],
        new_policy_won=frozen["new_policy_won"],
        all_validation_records_verified=True,
        training_method="direct_real_survival_cma",
        imported_public_world=True,
        own_world_model_training=False,
        canonical_preserved=True,
        goal_completion_unproven=True,
        actual_process_session_exit_review_still_required=True,
    )
    if expected:
        output["paired_test"] = completed_pair(
            protocol, frozen, expected, resamples=resamples
        )
        output["all200_reserved_raw_records_verified"] = True
        output["reserved_test_consumed"] = True
        output["observed1092"] = output["paired_test"]["observed1092"]
        output["test_report_sha256"] = {
            case["report"]: digest(case["report"]) for case in expected
        }
        frozen_time = datetime.fromisoformat(frozen["frozen_at"])
        if any(
            datetime.fromisoformat(read(case["report"])["measured_at"]) <= frozen_time
            for case in expected
        ):
            raise ValueError("Reserved report predates its frozen validation choice")
    else:
        for name in ("test_selected", "test_control"):
            report = (
                Path(frozen["report_directory"])
                / f"{frozen['report_prefix']}_{name}.json"
            )
            if any(
                path.exists()
                for path in (
                    report,
                    report.with_suffix(".partial.json"),
                    report.with_suffix(".failed.json"),
                )
            ):
                raise ValueError(
                    "Retained-control comparison must not consume reserved games"
                )
        first, last = protocol["test_seed_range"]
        audit = recorded_seed_audit(dict(test=list(range(first, last + 1))))
        if audit["overlap"]["test"]:
            raise ValueError("Reserved seeds were used by another recorded run")
        output.update(
            reserved_test_consumed=False,
            observed1092=False,
            no_reserved_consumption_verified=True,
        )
    return output


def write_audit(path, value):
    path = Path(path)
    if path.exists():
        saved = read(path)
        if any(
            saved[key] != item for key, item in value.items() if key != "checked_at"
        ):
            raise ValueError("Preserved audit differs from current reconstruction")
        return
    with path.open("x") as stream:
        stream.write(json.dumps(value, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--freeze")
    mode.add_argument("--result")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    value = (
        frozen_choice_audit(args.protocol, args.freeze)
        if args.freeze
        else real_result_audit(args.protocol, args.result)
    )
    write_audit(args.output, value)
    print(
        json.dumps(
            {
                key: value[key]
                for key in ("selected", "new_policy_won", "goal_completion_unproven")
            }
        )
    )


if __name__ == "__main__":
    main()
