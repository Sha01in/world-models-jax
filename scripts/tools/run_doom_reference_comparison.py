"""Serial fresh validation, independent choice audit and conditional paired test."""
# ruff: noqa: E402 -- initialize workspace and CPU-only supervisor first.

import argparse
from datetime import datetime, timezone
import fcntl
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.doom_reference_comparison import (
    archive_info,
    checked_partial,
    checked_report,
    digest,
    identity,
    public_parameters,
    recorded_seed_audit,
    read,
    select_complete_validation,
    test_cases,
    validation_cases,
    verify_protocol,
)


def write_new(path, value):
    with Path(path).open("x") as stream:
        stream.write(json.dumps(value, indent=2) + "\n")


def comparison(
    protocol_path,
    directory,
    prefix,
    training_audit,
    *,
    evaluate,
    audit_choice,
    audit_result,
    source_hashes,
):
    """The independent choice callback must succeed before any reserved job."""
    protocol = read(protocol_path)
    cases = validation_cases(protocol, directory, prefix)
    registered_path = Path(directory) / f"{prefix}_registered_validation_cases.json"
    registration = dict(
        protocol_sha256=digest(protocol_path),
        training_audit_sha256=digest(training_audit),
        validation_tasks=cases,
    )
    if registered_path.exists():
        if read(registered_path) != registration:
            raise ValueError("Completed search/case registration changed")
    else:
        write_new(registered_path, registration)
    for case in cases:
        evaluate(protocol, case)
    decision, reports = select_complete_validation(protocol, cases)
    freeze_path = Path(directory) / f"{prefix}_frozen_selection.json"
    proposal = dict(
        protocol_sha256=digest(protocol_path),
        training_audit=str(training_audit),
        training_audit_sha256=digest(training_audit),
        validation_tasks=cases,
        validation_report_sha256={
            case["report"]: digest(case["report"]) for case in cases
        },
        validation_scores={
            case["name"]: report["mean"]
            for case, report in zip(cases, reports, strict=True)
        },
        selected=decision["selected"],
        paired_control=decision["paired_control"],
        new_policy_won=decision["new_policy_won"],
        test_seed_range=protocol["test_seed_range"],
        selection_source_sha256=source_hashes,
        report_directory=str(directory),
        report_prefix=prefix,
        training_method="direct_real_survival_cma",
        imported_public_world=True,
        goal_completion_unproven=True,
    )
    if freeze_path.exists():
        frozen = read(freeze_path)
        if any(frozen[key] != value for key, value in proposal.items()):
            raise ValueError("Frozen validation choice or provenance changed")
    else:
        frozen = dict(proposal, frozen_at=datetime.now(timezone.utc).isoformat())
        write_new(freeze_path, frozen)
    audit_path = Path(directory) / f"{prefix}_frozen_selection_cpu_audit.json"
    audit_choice(protocol_path, freeze_path, audit_path)
    verified = read(audit_path)
    if (
        verified["frozen_selection_sha256"] != digest(freeze_path)
        or verified["protocol_sha256"] != digest(protocol_path)
        or verified["selected"] != frozen["selected"]
        or verified["paired_control"] != frozen["paired_control"]
        or verified["new_policy_won"] != frozen["new_policy_won"]
        or not verified["all_validation_records_verified"]
        or not verified["complete_candidate_eligibility_reconstructed"]
    ):
        raise ValueError(
            "Independent selection audit does not verify the frozen choice"
        )
    reserved = test_cases(protocol, frozen, directory, prefix)
    for case in reserved:
        evaluate(protocol, case)
    result_path = Path(directory) / f"{prefix}_real_evaluation_result.json"
    result = dict(
        protocol=str(protocol_path),
        protocol_sha256=digest(protocol_path),
        frozen_selection=str(freeze_path),
        frozen_selection_sha256=digest(freeze_path),
        selection_audit=str(audit_path),
        selection_audit_sha256=digest(audit_path),
        selected=frozen["selected"],
        paired_control=frozen["paired_control"],
        new_policy_won=frozen["new_policy_won"],
        test_tasks=reserved,
        reserved_test_consumed=bool(reserved),
        imported_public_world=True,
        training_method="direct_real_survival_cma",
        goal_completion_unproven=True,
    )
    if result_path.exists():
        if read(result_path) != result:
            raise ValueError("Completed real result differs from frozen choice")
    else:
        write_new(result_path, result)
    audit_result(
        protocol_path,
        result_path,
        Path(directory) / f"{prefix}_real_result_cpu_audit.json",
    )
    return result


def current_training_ready(protocol):
    state = read("artifacts/task_state.json")
    missing = []
    if (
        not state.get("reference_round13_training_supervisor_exit_verified")
        or state.get("reference_round13_training_session_exit_code") != 0
    ):
        missing.append("actual training supervisor/session exit is not verified")
    for key in (
        "reference_round13_training_supervisor_pid",
        "reference_round13_training_pid",
    ):
        pid = state.get(key)
        if pid and Path(f"/proc/{pid}").exists():
            missing.append(f"{key} {pid} is still live")
    audit = Path("artifacts/doom_reference_round13_training_cpu_audit.json")
    if not audit.exists():
        missing.append("independent completed training audit missing")
    if missing:
        return None, missing
    verified = read(audit)
    if (
        verified["protocol_sha256"]
        != digest("artifacts/doom_reference_round13_direct_real_protocol.json")
        or not verified[
            "population_weights_optimizer_history_and_next_rng_reconstructed"
        ]
        or not verified["all_raw_game_pairs_verified"]
        or verified["raw_training_games"]
        != protocol["maximum_training_fitness_games"]
        + protocol["maximum_training_holdout_games"]
        or any(digest(path) != sha for path, sha in verified["files"].items())
    ):
        raise ValueError("Independent training audit/current capsule differs")
    return audit, []


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--directory", default="artifacts")
    parser.add_argument("--prefix", default="doom_reference_round13")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    protocol_path = Path(args.protocol).resolve()
    protocol = read(protocol_path)
    verify_protocol(protocol)
    if {name: version(name) for name in protocol["package_versions"]} != protocol[
        "package_versions"
    ]:
        raise ValueError("Current runtime differs from registration")
    training_audit, deferred = current_training_ready(protocol)
    if deferred:
        print(json.dumps(dict(ready=False, deferred=deferred, gpu_jobs_started=0)))
        if not args.check_only:
            raise RuntimeError(
                "Finish and verify the existing training supervisor first"
            )
        return
    directory = Path(args.directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    preparation = read(
        "artifacts/doom_reference_round13_real_workflow_preparation_cpu.json"
    )
    if (
        args.prefix != preparation["report_prefix"]
        or str(directory) != preparation["report_directory"]
    ):
        raise ValueError("Real report names/directory differ from verified workflow")
    if preparation["protocol_sha256"] != digest(protocol_path) or any(
        digest(path) != sha for path, sha in preparation["source_sha256"].items()
    ):
        raise ValueError("Verified real workflow source/protocol changed")
    if args.check_only:
        print(
            json.dumps(
                dict(ready=True, training_audit=str(training_audit), gpu_jobs_started=0)
            )
        )
        return
    if Path(directory / f"{args.prefix}_real_evaluation_result.json").exists():
        raise FileExistsError("Completed comparison requires review, not redispatch")

    def update(**changes):
        path = Path("artifacts/task_state.json")
        state = read(path)
        state.update(changes)
        temporary = path.with_name(path.name + ".real.tmp")
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(path)

    def run(command, stage, *, gpu):
        log = directory / f"{args.prefix}_{stage}.log"
        index = 0
        while log.exists():
            index += 1
            log = directory / f"{args.prefix}_{stage}_retry{index}.log"
        Path(directory / f"{args.prefix}_real_evaluations.status").write_text(
            stage + "\n"
        )
        with log.open("x") as stream:
            child = subprocess.Popen(
                [sys.executable, *command],
                stdout=stream,
                stderr=subprocess.STDOUT,
                env=dict(
                    os.environ,
                    JAX_PLATFORMS="cuda" if gpu else "cpu",
                    XLA_PYTHON_CLIENT_PREALLOCATE="false",
                    OPENBLAS_NUM_THREADS="4",
                    OMP_NUM_THREADS="1",
                    PYTHONUNBUFFERED="1",
                ),
            )
            update(
                active_reference_child_pid=child.pid,
                active_reference_log=str(log),
                reference_round13_stage=stage,
            )
            print(
                json.dumps(dict(stage=stage, child_pid=child.pid, log=str(log))),
                flush=True,
            )
            code = child.wait()
        update(active_reference_child_pid=None)
        if code:
            raise RuntimeError(
                f"{stage} exited{code}; preserve reports and diagnose its log"
            )

    def evaluate(current_protocol, case):
        verify_protocol(current_protocol)
        params = (
            public_parameters(current_protocol)
            if case["controller"] is None
            else archive_info(
                case["controller"], current_protocol, direct=case["kind"] == "candidate"
            )["params"]
        )
        if identity(params) != case["parameters_sha256"]:
            raise ValueError("Frozen actor changed before GPU dispatch")
        if {
            name: version(name) for name in current_protocol["package_versions"]
        } != current_protocol["package_versions"]:
            raise ValueError("Registered runtime changed before GPU dispatch")
        output = Path(case["report"])
        if output.exists():
            checked_report(current_protocol, case)
            return
        if output.with_suffix(".failed.json").exists():
            raise RuntimeError("Captured failure requires diagnosis before recovery")
        if output.with_suffix(".partial.json").exists():
            checked_partial(current_protocol, case, output.with_suffix(".partial.json"))
        run(["scripts/tools/check_gpu.py"], case["name"] + "_gpu_preflight", gpu=True)
        command = [
            "scripts/tools/evaluate_doom_reference.py",
            "--reference-dir",
            current_protocol["arguments"]["reference_dir"],
            "--asset-dir",
            current_protocol["arguments"]["asset_dir"],
            "--output",
            case["report"],
            "--episodes",
            str(case["episodes"]),
            "--seed",
            str(case["seed"]),
            "--workers",
            "8",
            "--inference",
            "posterior",
            "--role",
            case["role"],
        ]
        if case["controller"] is not None:
            command.extend(["--controller", case["controller"]])
        run(command, case["name"], gpu=True)
        checked_report(current_protocol, case)

    def audit_choice(current_protocol, freeze, output):
        run(
            [
                "scripts/tools/audit_doom_reference_comparison.py",
                "--protocol",
                str(current_protocol),
                "--freeze",
                str(freeze),
                "--output",
                str(output),
            ],
            "frozen_selection_cpu_audit",
            gpu=False,
        )

    def audit_result(current_protocol, result, output):
        run(
            [
                "scripts/tools/audit_doom_reference_comparison.py",
                "--protocol",
                str(current_protocol),
                "--result",
                str(result),
                "--output",
                str(output),
            ],
            "real_result_cpu_audit",
            gpu=False,
        )

    with Path("artifacts/doom_vision_round3_vaes.lock").open("a") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cases = validation_cases(protocol, directory, args.prefix)
        exclusions = []
        for case in cases:
            output = Path(case["report"])
            for path in (output, output.with_suffix(".partial.json")):
                if path.exists():
                    if path == output:
                        checked_report(protocol, case)
                    else:
                        checked_partial(protocol, case, path)
                    exclusions.append(path)
        freeze_path = directory / f"{args.prefix}_frozen_selection.json"
        if freeze_path.exists():
            # Only a verified immutable choice can exempt its own reserved
            # records for recovery. Audit it again before scheduling any GPU.
            choice = read(freeze_path)
            audit_choice(
                protocol_path,
                freeze_path,
                directory / f"{args.prefix}_frozen_selection_cpu_audit.json",
            )
            for case in test_cases(protocol, choice, directory, args.prefix):
                output = Path(case["report"])
                for path in (output, output.with_suffix(".partial.json")):
                    if path.exists():
                        checked_report(
                            protocol, case
                        ) if path == output else checked_partial(protocol, case, path)
                        exclusions.append(path)
        proposed = {
            name: list(range(first, last + 1))
            for name, (first, last) in dict(
                validation=protocol["validation_seed_range"],
                test=protocol["test_seed_range"],
            ).items()
        }
        seeds = recorded_seed_audit(proposed, exclude=exclusions)
        if any(seeds["overlap"].values()):
            raise ValueError("Fresh validation/test seeds overlap another recorded run")
        update(
            active_reference_pid=os.getpid(),
            reference_round13_real_supervisor_pid=os.getpid(),
            reference_round13_stage="real_validation_dispatch",
        )
        try:
            result = comparison(
                protocol_path,
                directory,
                args.prefix,
                training_audit,
                evaluate=evaluate,
                audit_choice=audit_choice,
                audit_result=audit_result,
                source_hashes=preparation["source_sha256"],
            )
            update(
                active_reference_pid=None,
                active_reference_child_pid=None,
                reference_round13_real_result=str(
                    directory / f"{args.prefix}_real_evaluation_result.json"
                ),
                reference_round13_stage="real_comparison_complete_awaiting_exit_review",
                reference_round13_new_policy_won=result["new_policy_won"],
                additional_gpu_work_queued=False,
            )
            print(json.dumps(result), flush=True)
        except BaseException:
            update(
                active_reference_pid=None,
                active_reference_child_pid=None,
                reference_round13_stage="real_comparison_failed_inspect_preserve",
                additional_gpu_work_queued=False,
            )
            raise


if __name__ == "__main__":
    main()
