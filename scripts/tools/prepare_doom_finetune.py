"""Combine fresh failures with historical training data, preserving held-out data."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", required=True)
    parser.add_argument("--new-data-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--historical-episodes", type=int, default=2560)
    parser.add_argument("--new-validation-fraction", type=float, default=0.1)
    parser.add_argument(
        "--integrity-report", default="artifacts/doom_series_integrity.json"
    )
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    if args.historical_episodes < 1 or not 0 < args.new_validation_fraction < 1:
        parser.error(
            "Historical count must be positive and validation fraction between zero and one"
        )
    output = Path(args.output_root)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a fresh output root to preserve the split")
    base = json.loads(Path(args.base_manifest).read_text())
    excluded = set()
    if Path(args.integrity_report).exists():
        for group in json.loads(Path(args.integrity_report).read_text())[
            "duplicate_paths"
        ]:
            excluded.update(str(Path(p).resolve()) for p in group[1:])
    historical = sorted(
        {Path(p).resolve() for p in base["training_files"]}
        - {Path(p) for p in excluded}
    )
    fresh = sorted(Path(args.new_data_dir).resolve().glob("*.npz"))
    if len(fresh) < 2:
        raise ValueError("At least two fresh episodes are needed")
    rng = np.random.default_rng(args.seed)
    selected = [
        historical[i]
        for i in rng.choice(
            len(historical),
            min(args.historical_episodes, len(historical)),
            replace=False,
        )
    ]
    rng.shuffle(fresh)
    n_val = min(len(fresh) - 1, max(1, int(len(fresh) * args.new_validation_fraction)))
    train_new, val_new = fresh[n_val:], fresh[:n_val]
    validation_old = sorted({Path(p).resolve() for p in base["validation_files"]})
    train_paths = selected + train_new
    validation_paths = validation_old + val_new
    if set(train_paths).intersection(validation_paths):
        raise ValueError("Training and validation share sources")
    for subset, paths in (("train", train_paths), ("validation", validation_paths)):
        directory = output / subset
        directory.mkdir(parents=True)
        for index, path in enumerate(paths):
            if not path.is_file():
                raise FileNotFoundError(path)
            (directory / f"episode_{index:05d}.npz").symlink_to(path)
    manifest = {
        "base_manifest": args.base_manifest,
        "seed": args.seed,
        "historical_training_episodes": len(selected),
        "new_training_episodes": len(train_new),
        "historical_validation_episodes": len(validation_old),
        "new_validation_episodes": len(val_new),
        "training_sources": [str(p) for p in train_paths],
        "validation_sources": [str(p) for p in validation_paths],
    }
    (output / "split.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in manifest.items() if not k.endswith("sources")}, indent=2
        )
    )


if __name__ == "__main__":
    main()
