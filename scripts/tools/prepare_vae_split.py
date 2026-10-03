"""Map a verified RNN episode split back to immutable raw VAE training data."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from src.vae_data import file_sha256, observation_header, transition_signature


def prepare(series_manifest, raw_root, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Preserve existing splits; use a new output path")
    source = json.loads(Path(series_manifest).read_text())
    wanted = defaultdict(list)
    for partition in ("training", "validation"):
        for path in source[f"{partition}_sources"]:
            with np.load(path, allow_pickle=False) as data:
                wanted[transition_signature(data)].append((partition, path))
    overlap = [s for s, entries in wanted.items() if len({p for p, _ in entries}) > 1]
    if overlap:
        raise ValueError(f"{len(overlap)} transition groups cross the source split")
    matched = defaultdict(list)
    files = sorted(Path(raw_root).rglob("*.npz"))
    for index, path in enumerate(files, 1):
        with np.load(path, allow_pickle=False) as data:
            signature = transition_signature(data)
        if signature in wanted:
            matched[signature].append(path)
        if index % 1000 == 0:
            print(f"Indexed {index}/{len(files)} raw episodes", flush=True)
    missing = set(wanted) - set(matched)
    if missing:
        raise ValueError(f"Cannot recover {len(missing)} raw episode groups")
    result = {
        "schema_version": 1,
        "series_manifest": str(Path(series_manifest).resolve()),
        "series_manifest_sha256": file_sha256(series_manifest),
        "raw_root": str(Path(raw_root).resolve()),
        "preprocessing": "existing_full_frame_rgb64_uint8",
        "training_episodes": [],
        "validation_episodes": [],
        "excluded_ambiguous_groups": [],
    }
    for index, signature in enumerate(sorted(wanted), 1):
        paths = matched[signature]
        # Multiple archives can be byte-identical aliases. Different raw files
        # with the same actions/outcomes cannot be identified safely from legacy
        # latent files; exclude them rather than silently guessing the source.
        hashes = {file_sha256(path) for path in paths}
        if len(hashes) > 1:
            result["excluded_ambiguous_groups"].append(
                {"signature": signature, "raw_paths": [str(p) for p in paths]}
            )
            continue
        path = paths[0]
        shape = observation_header(path)
        with np.load(path, allow_pickle=False) as data:
            deaths = np.asarray(data["dones"], bool)
            terminated = bool(deaths[-1])
        partition = wanted[signature][0][0]
        result[f"{partition}_episodes"].append(
            {
                "path": str(path.resolve()),
                "sha256": next(iter(hashes)),
                "signature": signature,
                "frames": shape[0],
                "physical_death": terminated,
                "series_sources": [p for _, p in wanted[signature]],
            }
        )
        if index % 500 == 0:
            print(f"Fingerprinting {index}/{len(wanted)} selected episodes", flush=True)
    result["counts"] = {
        partition: {
            "episodes": len(result[f"{partition}_episodes"]),
            "frames": sum(e["frames"] for e in result[f"{partition}_episodes"]),
            "fatal": sum(e["physical_death"] for e in result[f"{partition}_episodes"]),
        }
        for partition in ("training", "validation")
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["counts"], indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--series-manifest", required=True)
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    prepare(args.series_manifest, args.raw_root, args.output)


if __name__ == "__main__":
    main()
