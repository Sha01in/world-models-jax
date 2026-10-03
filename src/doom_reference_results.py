"""Validate real-game survival reports and quantify fixed-policy uncertainty."""

import numpy as np

from src.episode_statistics import episode_bootstrap


def checked_survival(report, expected_seeds, *, role="test"):
    """Require complete, unique, unit-reward games in the requested seed order."""
    rows = report["episodes_detail"]
    seeds = [row["seed"] for row in rows]
    expected = list(expected_seeds)
    if seeds != expected or len(set(seeds)) != len(seeds) or not seeds:
        raise ValueError("Report seed cohort incomplete, duplicated or reordered")
    if report["episodes"] != len(rows) or report["protocol"]["evaluation_role"] != role:
        raise ValueError("Report count or evaluation role differs")
    for row in rows:
        steps, counts = row["survival_steps"], row["actions_left_right_wait"]
        if (
            not isinstance(steps, int)
            or isinstance(steps, bool)
            or not 1 <= steps <= 2100
            or row["score"] != steps
            or len(counts) != 3
            or any(
                not isinstance(c, int) or isinstance(c, bool) or c < 0 for c in counts
            )
            or sum(counts) != steps
            or type(row["terminated"]) is not bool
            or type(row["truncated"]) is not bool
            or not (row["terminated"] ^ row["truncated"])
        ):
            raise ValueError(f"Invalid real-game survival record: {row}")
    values = np.asarray([row["survival_steps"] for row in rows], np.float64)
    if (
        report["mean"] != float(values.mean())
        or not np.isclose(report["std"], values.std(), rtol=1e-12, atol=1e-9)
        or report["deaths"] != sum(row["terminated"] for row in rows)
        or report["timeouts"] != sum(row["truncated"] for row in rows)
    ):
        raise ValueError("Reported mean/SD/outcome counts differ from actual games")
    return values


def paired_survival_summary(
    selected, control, expected_seeds, *, resamples=50000, seed=74140
):
    """Use the same resampled games for both policies and their difference."""
    expected_seeds = list(expected_seeds)
    scores = checked_survival(selected, expected_seeds)
    baseline = checked_survival(control, expected_seeds)
    first, second = selected["protocol"], control["protocol"]
    shared = (
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
    if any(first[key] != second[key] for key in shared):
        raise ValueError(
            "Paired reports differ in world/environment/numerical protocol"
        )
    for name in ("vae.json", "rnn.json", "take_cover.wad", "freedoom2.wad"):
        fingerprints = []
        for protocol in (first, second):
            matching = [
                value
                for path, value in protocol["input_sha256"].items()
                if path.replace("\\", "/").rsplit("/", 1)[-1] == name
            ]
            if len(matching) != 1:
                raise ValueError(f"Missing or ambiguous paired input: {name}")
            fingerprints.append(matching[0])
        if fingerprints[0] != fingerprints[1]:
            raise ValueError(f"Paired input differs: {name}")
    differences = scores - baseline
    uncertainty = episode_bootstrap(
        np.asarray(list(expected_seeds)),
        lambda index: [
            scores[index].mean(),
            baseline[index].mean(),
            differences[index].mean(),
        ],
        resamples=resamples,
        seed=seed,
    )
    uncertainty["score_rows"] = uncertainty.pop("frames")
    uncertainty["unit"] = "whole seeded real games; paired scores resampled together"
    uncertainty["method"] = "percentile bootstrap, controller and world held fixed"
    return dict(
        episodes=len(scores),
        selected_mean=float(scores.mean()),
        selected_std=float(scores.std()),
        control_mean=float(baseline.mean()),
        control_std=float(baseline.std()),
        paired_gain=float(differences.mean()),
        selected_mean95=uncertainty["intervals"][0],
        control_mean95=uncertainty["intervals"][1],
        paired_gain95=uncertainty["intervals"][2],
        wins=int((differences > 0).sum()),
        losses=int((differences < 0).sum()),
        ties=int((differences == 0).sum()),
        bootstrap=uncertainty,
        observed1092=bool(len(scores) >= 100 and scores.mean() >= 1092),
        own_policy_update=bool(first["controller_provenance"]["own_policy_update"]),
        imported_public_world=bool(
            first["controller_provenance"]["imported_public_world"]
        ),
        limitation="Whole-game sampling uncertainty for fixed policies; excludes training variability, selection uncertainty and differences from the historical paper runtime.",
    )
