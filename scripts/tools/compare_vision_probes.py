"""Paired feature-probe differences on identical held-out episode frames."""

import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from scripts.tools.probe_doom_vision import auc
from src.episode_statistics import episode_bootstrap
from src.vae_data import file_sha256


def load_predictions(report_path):
    report = json.loads(Path(report_path).read_text())
    predictions = Path(report["predictions"])
    if file_sha256(predictions) != report["predictions_sha256"]:
        raise ValueError("Probe predictions do not match their report")
    with np.load(predictions, allow_pickle=False) as source:
        arrays = {name: source[name] for name in source.files}
    if (
        str(arrays["split_sha256"]) != report["split_sha256"]
        or str(arrays["vae_sha256"]) != report["vae_sha256"]
    ):
        raise ValueError("Prediction provenance disagrees with probe report")
    return report, arrays


def compare_predictions(reference, candidate):
    for name in ("episode_signatures", "frame_indices", "targets", "split_sha256"):
        if not np.array_equal(reference[name], candidate[name]):
            raise ValueError(
                "Paired probes must use identical held-out frames and targets"
            )
    identities = list(
        zip(
            reference["episode_signatures"].tolist(),
            reference["frame_indices"].tolist(),
        )
    )
    if len(identities) != len(set(identities)):
        raise ValueError("Probe frame identities are duplicated")

    def statistics(rows):
        labels = reference["targets"][rows]
        positive = labels[:, 0].astype(bool)
        areas, errors = [], []
        for data in (reference, candidate):
            areas.append(
                np.array(
                    [
                        np.nan
                        if (
                            value := auc(
                                labels[:, index], data["presence_scores"][rows, index]
                            )
                        )
                        is None
                        else value
                        for index in range(2)
                    ]
                )
            )
            errors.append(
                np.abs(data["positions"][rows][positive] - labels[positive, 2:]).mean(
                    axis=0
                )
                if positive.any()
                else np.full(2, np.nan)
            )
        return np.concatenate(
            [
                areas[0],
                areas[1],
                areas[1] - areas[0],
                errors[0],
                errors[1],
                errors[0] - errors[1],
            ]
        )

    result = episode_bootstrap(reference["episode_signatures"], statistics)
    result["metric_order"] = [
        "reference_larger_auc",
        "reference_small_auc",
        "candidate_larger_auc",
        "candidate_small_auc",
        "larger_auc_gain",
        "small_auc_gain",
        "reference_horizontal_mae",
        "reference_vertical_mae",
        "candidate_horizontal_mae",
        "candidate_vertical_mae",
        "horizontal_mae_reduction",
        "vertical_mae_reduction",
    ]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("Preserve previous comparisons; use a fresh output")
    baseline, baseline_arrays = load_predictions(args.reference)
    candidate, candidate_arrays = load_predictions(args.candidate)
    report = dict(
        reference=str(Path(args.reference).resolve()),
        candidate=str(Path(args.candidate).resolve()),
        reference_vae_sha256=baseline["vae_sha256"],
        candidate_vae_sha256=candidate["vae_sha256"],
        paired_comparison=compare_predictions(baseline_arrays, candidate_arrays),
        limitation="Color proxies are not verified projectile labels; fitted probes and VAEs are held fixed. Feature gains do not establish real survival or include training variability.",
    )
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
