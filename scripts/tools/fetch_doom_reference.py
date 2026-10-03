"""Fetch immutable public reference weights and source for the VizDoom audit."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.request

REFERENCE_COMMIT = "fd982b9691a941b52c6addbde29bc801ca6202c8"
BASE = f"https://raw.githubusercontent.com/hardmaru/WorldModelsExperiments/{REFERENCE_COMMIT}"
FILES = {
    "vae.json": "doomrnn/tf_models/vae.json",
    "rnn.json": "doomrnn/tf_models/rnn.json",
    "initial_z.json": "doomrnn/tf_models/initial_z.json",
    "controller.json": "doomrnn/log/doomrnn.cma.16.64.best.json",
    "source/doomrnn.py": "doomrnn/doomrnn.py",
    "source/doomreal.py": "doomrnn/doomreal.py",
    "source/model.py": "doomrnn/model.py",
    "source/config.py": "doomrnn/config.py",
    "source/extract.py": "doomrnn/extract.py",
    "source/rnn_train.py": "doomrnn/rnn_train.py",
    "source/vae_train.py": "doomrnn/vae_train.py",
    "source/README.md": "doomrnn/README.md",
}
AUXILIARY = {
    "legacy/scipy_pilutil.py": "https://raw.githubusercontent.com/scipy/scipy/v1.1.0/scipy/misc/pilutil.py",
    "legacy/tensorflow_rnn_cell.py": "https://raw.githubusercontent.com/tensorflow/tensorflow/v1.8.0/tensorflow/contrib/rnn/python/ops/rnn_cell.py",
    "legacy/doom_assets_tree.json": "https://api.github.com/repos/ppaquette/gym-doom/git/trees/60ff576?recursive=1",
    "legacy/doom_take_cover.py": "https://raw.githubusercontent.com/ppaquette/gym-doom/60ff576/ppaquette_gym_doom/doom_take_cover.py",
}


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fetch(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "download_manifest.json"
    manifest = (
        json.loads(manifest_path.read_text())
        if manifest_path.exists()
        else dict(
            reference_commit=REFERENCE_COMMIT,
            started_at=datetime.now(timezone.utc).isoformat(),
            complete=False,
            files={},
            purpose="Diagnostic reference control; supplied weights alone do not reproduce our training or prove the paper's reported checkpoint provenance.",
        )
    )
    if manifest["reference_commit"] != REFERENCE_COMMIT:
        raise ValueError("Reference revision differs; use a separate output directory")
    manifest["complete"] = False
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    urls = {local: f"{BASE}/{remote}" for local, remote in FILES.items()}
    urls.update(AUXILIARY)
    for local, url in urls.items():
        path = output / local
        if path.exists():
            saved = manifest["files"].get(local)
            if not saved or saved["url"] != url or saved["sha256"] != sha256(path):
                raise ValueError(
                    f"Preserve unexpected or modified reference file: {path}"
                )
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".download")
        request = urllib.request.Request(
            url, headers={"User-Agent": "WorldModels-reference-audit"}
        )
        with (
            urllib.request.urlopen(request, timeout=60) as response,
            temporary.open("wb") as stream,
        ):
            while chunk := response.read(1024 * 1024):
                stream.write(chunk)
        temporary.rename(path)
        manifest["files"][local] = dict(
            url=url, bytes=path.stat().st_size, sha256=sha256(path)
        )
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps(dict(file=local, **manifest["files"][local])), flush=True)
    manifest["complete"] = True
    manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    fetch(args.output_dir)


if __name__ == "__main__":
    main()
