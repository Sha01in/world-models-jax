"""Compare detection counts on matched whole episodes, retaining temporal dependence."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np

from src.episode_statistics import episode_bootstrap
from src.vae_data import file_sha256


def compare(reference, candidate, *, resamples=5000, seed=73502):
    for field in (
        "posterior_input_protocol",
        "mean_latents_override",
        "inference_posterior_sampling",
        "seed",
        "inference_batch_size",
    ):
        if reference[field] != candidate[field]:
            raise ValueError(f"Audit protocol mismatch: {field}")
    rows = []
    for report in (reference, candidate):
        records = report["episode_records"]
        identities = [r["transition_signature"] for r in records]
        if len(identities) != len(set(identities)):
            raise ValueError("Duplicate audit episode identity")
        rows.append({r["transition_signature"]: r for r in records})
    if set(rows[0]) != set(rows[1]) or not rows[0]:
        raise ValueError("Audits cover different episodes")
    if [r["transition_signature"] for r in reference["episode_records"]] != [
        r["transition_signature"] for r in candidate["episode_records"]
    ]:
        raise ValueError(
            "Audit sampling order differs; posterior RNG pairing is unverified"
        )
    names = sorted(rows[0])
    counts = []
    fields = (
        "detected_deaths",
        "fatal_transitions",
        "false_deaths",
        "live_transitions",
    )
    for report in rows:
        values = np.asarray(
            [[report[name]["counts"][k] for k in fields] for name in names], np.int64
        )
        if (
            np.any(values < 0)
            or np.any(values[:, 0] > values[:, 1])
            or np.any(values[:, 2] > values[:, 3])
        ):
            raise ValueError("Invalid detection counts")
        if any(
            report[name]["frames"] != values[i, 1] + values[i, 3]
            for i, name in enumerate(names)
        ):
            raise ValueError("Episode labels or lengths differ")
        counts.append(values)
    if not np.array_equal(counts[0][:, [1, 3]], counts[1][:, [1, 3]]) or any(
        rows[0][name]["frames"] != rows[1][name]["frames"] for name in names
    ):
        raise ValueError("Episode labels or lengths differ")

    def statistic(indices):
        rates = []
        for values in counts:
            tp, positive, fp, negative = values[indices].sum(axis=0)
            rates.extend(
                (
                    tp / positive if positive else np.nan,
                    fp / negative if negative else np.nan,
                )
            )
        return [*rates, rates[2] - rates[0], rates[1] - rates[3]]

    result = episode_bootstrap(names, statistic, resamples=resamples, seed=seed)
    result["method"] = "paired percentile bootstrap, fitted RNNs held fixed"
    result["frames"] = sum(rows[0][name]["frames"] for name in names)
    result["metric_order"] = [
        "reference_recall",
        "reference_false_death_rate",
        "candidate_recall",
        "candidate_false_death_rate",
        "recall_gain",
        "false_death_rate_reduction",
    ]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if Path(args.output).exists():
        raise FileExistsError("Preserve existing comparison")
    reports = [
        json.loads(Path(p).read_text()) for p in (args.reference, args.candidate)
    ]
    report = dict(
        reference=args.reference,
        candidate=args.candidate,
        source_report_sha256=[file_sha256(p) for p in (args.reference, args.candidate)],
        paired_comparison=compare(*reports),
        limitation="Weighted-BCE detection scores are not calibrated probabilities; fixed-model holdout uncertainty is not real-game performance or training variability.",
    )
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
