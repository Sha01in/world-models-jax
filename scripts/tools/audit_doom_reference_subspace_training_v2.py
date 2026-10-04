"""CPU replay of a warm-start direct-real search and its selected initializer."""
# ruff: noqa: E402 -- select CPU before importing the unchanged replay core.

import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.doom_subspace_audit import audit_subspace_capsule as audit_search_capsule
from scripts.tools.train_doom_reference_real import digest
from src.doom_subspace_initializer_v2 import registered_initializer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    protocol = json.loads(Path(args.protocol).read_text())
    metadata = json.loads(
        Path(str(protocol["arguments"]["output"]) + ".json").read_text()
    )
    if (
        metadata["arguments"] != protocol["arguments"]
        or metadata["protocol_sha256"] != digest(args.protocol)
        or metadata["source_sha256"] != protocol["frozen_source"]
        or metadata["input_sha256"] != protocol["frozen_inputs"]
    ):
        raise ValueError("Completed training metadata differs from registration")
    for path, fingerprint in {
        **protocol["frozen_inputs"],
        **protocol["frozen_source"],
    }.items():
        if digest(path) != fingerprint:
            raise ValueError(f"Registered fingerprint changed: {path}")
    if {name: version(name) for name in protocol["package_versions"]} != protocol[
        "package_versions"
    ]:
        raise ValueError("Runtime differs from registration")
    initial = registered_initializer(protocol)
    result = audit_search_capsule(protocol["arguments"]["output"], initial)
    result.update(
        protocol_sha256=digest(args.protocol),
        initializer_selected_from_complete_validation_only=True,
        initializer_parameters_sha256=protocol["initializer_parameters_sha256"],
        canonical_preserved=True,
        projected_parent_initializer_reconstructed=True,
        initializer_parent_protocol_sha256=digest(
            protocol["initializer_parent_protocol"]
        ),
        imported_public_world=True,
        own_world_model_training=False,
        current_package_versions=protocol["package_versions"],
    )
    with Path(args.output).open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "generations",
                    "raw_training_games",
                    "best_generation",
                    "best_real_training_holdout_score",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
