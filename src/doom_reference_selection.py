"""Validation-only selection for a controlled reference-policy comparison."""

from src.doom_reference_results import checked_survival


def reference_validation_decision(tasks, reports):
    """Prefer earlier controls on ties and test only a genuine new-policy winner."""
    if not tasks or len(tasks) != len(reports):
        raise ValueError("Complete matching validation tasks/reports required")
    cohort = (tasks[0]["seed"], tasks[0]["episodes"], tasks[0]["inference"])
    identities = set()
    controls = []
    for index, (task, report) in enumerate(zip(tasks, reports)):
        if (task["seed"], task["episodes"], task["inference"]) != cohort:
            raise ValueError("Validation policies must share seeds and inference")
        checked_survival(
            report,
            range(task["seed"], task["seed"] + task["episodes"]),
            role="validation",
        )
        protocol = report["protocol"]
        if protocol["controller_provenance"].get("controller") != task[
            "controller"
        ] or protocol["inference_posterior_sampling"] != (
            task["inference"] == "posterior"
        ):
            raise ValueError("Validation actor/inference differs from task")
        identity = (task["parameters_sha256"], task["inference"])
        if identity in identities:
            raise ValueError(
                "Identical policies must be deduplicated before evaluation"
            )
        identities.add(identity)
        if task["kind"] == "control":
            controls.append(index)
        elif task["kind"] != "candidate":
            raise ValueError("Unknown policy role")
    if not controls or controls != list(range(len(controls))):
        raise ValueError("Controls must precede candidates for deterministic ties")
    winner = max(range(len(reports)), key=lambda index: reports[index]["mean"])
    provenance = reports[winner]["protocol"]["controller_provenance"]
    new_won = bool(
        tasks[winner]["kind"] == "candidate"
        and provenance["own_policy_update"]
        and provenance.get("generation", 0) > 0
    )
    return dict(
        winner=winner,
        selected=tasks[winner],
        new_policy_won=new_won,
        selected_validation_mean=reports[winner]["mean"],
        best_control_mean=max(reports[i]["mean"] for i in controls),
        reason="new updated policy won validation"
        if new_won
        else "control retained or candidate was not updated; no new reserved test",
    )
