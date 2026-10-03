"""Compare fixed visual audit frames with paired whole-episode uncertainty."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from src.episode_statistics import episode_bootstrap
from src.vae_data import file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("Preserve previous comparisons; use a fresh output")
    directories = [Path(args.reference), Path(args.candidate)]
    reports = [json.loads((d / "report.json").read_text()) for d in directories]
    records = [r["records"] for r in reports]
    identities = [[(r["signature"], r["frame"]) for r in rows] for rows in records]
    if (
        identities[0] != identities[1]
        or reports[0]["split_sha256"] != reports[1]["split_sha256"]
        or len(set(identities[0])) != len(identities[0])
    ):
        raise ValueError("Audits must use identical, unique held-out frames")
    with (
        np.load(directories[0] / "frames.npz") as left,
        np.load(directories[1] / "frames.npz") as right,
    ):
        if not np.array_equal(left["original"], right["original"]):
            raise ValueError("Audit raw pixels differ")
    errors = [
        np.array([[r["mean_pixel_mse"], r["posterior_pixel_mse"]] for r in rows])
        for rows in records
    ]

    def statistics(rows):
        baseline, candidate = (error[rows].mean(axis=0) for error in errors)
        return np.concatenate(
            [baseline, candidate, baseline - candidate, 1 - candidate / baseline]
        )

    comparison = episode_bootstrap(
        [identity[0] for identity in identities[0]], statistics
    )
    comparison["method"] = "paired percentile bootstrap, fitted VAEs held fixed"
    comparison["metric_order"] = [
        "reference_mean_mse",
        "reference_posterior_mse",
        "candidate_mean_mse",
        "candidate_posterior_mse",
        "mean_mse_reduction",
        "posterior_mse_reduction",
        "mean_mse_relative_reduction",
        "posterior_mse_relative_reduction",
    ]
    report = dict(
        reference=reports[0]["vae"],
        candidate=reports[1]["vae"],
        reference_vae_sha256=reports[0]["vae_sha256"],
        candidate_vae_sha256=reports[1]["vae_sha256"],
        audit_fingerprints=[file_sha256(d / "report.json") for d in directories],
        paired_comparison=comparison,
        limitation="Small fatal-window sample and whole-image pixel error; not verified projectile detection or real survival. Training variability is not included.",
    )
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
