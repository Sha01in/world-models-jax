"""Check exact duplicate latent episodes and train/validation leakage."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/series/VizdoomTakeCover-v0")
    parser.add_argument("--manifest", required=True, help="RNN JSON sidecar")
    parser.add_argument("--output", default="artifacts/doom_series_integrity.json")
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    training = {str(Path(f).resolve()) for f in manifest["training_files"]}
    validation = {str(Path(f).resolve()) for f in manifest["validation_files"]}
    groups = {}
    internal_terminal = []
    inconsistent_lengths = []
    frames = 0
    files = sorted(Path(args.data_dir).glob("*.npz"))
    for path in files:
        digest = hashlib.sha256()
        with np.load(path) as episode:
            for name in ("mu", "logvar", "actions", "rewards", "dones"):
                value = np.ascontiguousarray(episode[name])
                digest.update(name.encode())
                digest.update(str(value.shape).encode())
                digest.update(str(value.dtype).encode())
                digest.update(value.tobytes())
            n = len(episode["mu"])
            frames += n
            if any(
                len(episode[name]) != n
                for name in ("logvar", "actions", "rewards", "dones")
            ):
                inconsistent_lengths.append(str(path))
            if np.any(episode["dones"][:-1]):
                internal_terminal.append(str(path))
        groups.setdefault(digest.hexdigest(), []).append(str(path.resolve()))
    duplicates = [paths for paths in groups.values() if len(paths) > 1]
    leakage = [
        paths
        for paths in duplicates
        if training.intersection(paths) and validation.intersection(paths)
    ]
    report = {
        "episodes": len(files),
        "frames": frames,
        "unique_episodes": len(groups),
        "duplicate_groups": len(duplicates),
        "duplicate_extra_episodes": sum(len(paths) - 1 for paths in duplicates),
        "train_validation_duplicate_groups": len(leakage),
        "internal_terminal_episodes": internal_terminal,
        "inconsistent_length_episodes": inconsistent_lengths,
        "duplicate_paths": duplicates,
        "leakage_paths": leakage,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in report.items() if not k.endswith("paths")}, indent=2
        )
    )


if __name__ == "__main__":
    main()
