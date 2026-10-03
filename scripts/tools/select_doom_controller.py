"""Freeze the policy selected on a common real-game validation seed set."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluations", nargs="+", required=True)
    parser.add_argument(
        "--checkpoint-dir", default=None, help="Fallback for older reports"
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    reports = [(Path(p), json.loads(Path(p).read_text())) for p in args.evaluations]
    seed_sets = {
        tuple(r["seed"] for r in report["episodes_detail"]) for _, report in reports
    }
    if len(seed_sets) != 1 or any(
        report["policy"] != "controller" for _, report in reports
    ):
        raise ValueError("Compare controller evaluations on the same validation seeds")
    selected_path, selected = max(reports, key=lambda pair: pair[1]["mean"])
    source_base = selected.get("checkpoint_dir") or args.checkpoint_dir
    if source_base is None:
        raise ValueError(
            "The selected report needs checkpoint_dir or a fallback argument"
        )
    base = Path(source_base)
    controller_names = set(selected["checkpoint_sha256"]) - {"vae.eqx", "rnn.eqx"}
    if len(controller_names) != 1:
        raise ValueError("Expected exactly one controller fingerprint")
    source = base / next(iter(controller_names))
    paths = (base / "vae.eqx", base / "rnn.eqx", source)
    if {p.name: digest(p) for p in paths} != selected["checkpoint_sha256"]:
        raise ValueError("Source checkpoints changed since validation")
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            "Use a fresh output directory to preserve the previous selection"
        )
    with np.load(source) as data:
        payload = {name: data[name].copy() for name in data.files}
    payload["training_posterior_sampling"] = payload.get(
        "training_posterior_sampling", payload.get("posterior_sampling", False)
    )
    if selected.get("mean_latents_override", False):
        payload["posterior_sampling"] = False
    if "inference_posterior_sampling" in selected:
        payload["posterior_sampling"] = selected["inference_posterior_sampling"]
    payload["selected_validation_mean"] = selected["mean"]
    payload["selection_episodes"] = selected["episodes"]
    output.mkdir(parents=True, exist_ok=True)
    for path in (base / "vae.eqx", base / "rnn.eqx", base / "rnn.eqx.json"):
        shutil.copy2(path, output / path.name)
    vae_sidecar = base / "vae.eqx.json"
    if vae_sidecar.exists():
        shutil.copy2(vae_sidecar, output / vae_sidecar.name)
    destination = output / "controller_dream.npz"
    np.savez(destination, **payload)
    source_settings = Path(str(source) + ".json")
    provenance = {
        "selection_rule": "Maximum mean on common real-game validation seeds",
        "validation_seeds": list(next(iter(seed_sets))),
        "candidate_means": {str(p): r["mean"] for p, r in reports},
        "selected_evaluation": str(selected_path),
        "source_controller": str(source),
        "source_checkpoint_sha256": selected["checkpoint_sha256"],
        "inference_uses_vae_means": not bool(payload.get("posterior_sampling", False)),
        "training_settings": json.loads(source_settings.read_text())
        if source_settings.exists()
        else {},
        "selected_controller_sha256": digest(destination),
    }
    Path(str(destination) + ".json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Selected {source}: validation mean {selected['mean']:.2f}; saved {output}")


if __name__ == "__main__":
    main()
