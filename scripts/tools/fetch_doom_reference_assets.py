"""Inspect the pinned doom-py distribution without installing legacy bindings."""

import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import urllib.request
import zipfile

ARCHIVE_SHA256 = "2878cff5fc4b898040664b4a951ae55aeeaef691e3ef38a44fee85799e9b8e75"
FREEDOOM_URL = (
    "https://github.com/freedoom/freedoom/releases/download/v0.10.1/freedoom-0.10.1.zip"
)


def fetch_freedoom(output):
    """Version from doom-py 0.0.15's download recipe; never install packages."""
    archive = output / "freedoom-0.10.1.zip"
    record = output / "freedoom_manifest.json"
    previous = json.loads(record.read_text()) if record.exists() else None
    if not archive.exists():
        with urllib.request.urlopen(FREEDOOM_URL, timeout=60) as response:
            data = response.read()
        with archive.open("xb") as stream:
            stream.write(data)
    fingerprint = hashlib.sha256(archive.read_bytes()).hexdigest()
    if previous and previous["archive_sha256"] != fingerprint:
        raise ValueError("Existing Freedoom archive changed")
    member = "freedoom-0.10.1/freedoom2.wad"
    with zipfile.ZipFile(archive) as bundle:
        data = bundle.read(member)
    path = output / "freedoom2.wad"
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError("Preserve changed local Freedoom WAD")
    else:
        with path.open("xb") as stream:
            stream.write(data)
    manifest = dict(
        archive_url=FREEDOOM_URL,
        archive_sha256=fingerprint,
        member=member,
        wad_sha256=hashlib.sha256(data).hexdigest(),
        wad_bytes=len(data),
        provenance="Version explicitly named in the pinned doom-py download recipe. Release bytes fingerprinted here; no historical installation attestation.",
    )
    if previous and previous != manifest:
        raise ValueError("Freedoom provenance changed")
    if not record.exists():
        record.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest), flush=True)


def fetch(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    metadata_url = "https://pypi.org/pypi/doom-py/0.0.15/json"
    with urllib.request.urlopen(metadata_url, timeout=60) as response:
        metadata = json.load(response)
    source = next(row for row in metadata["urls"] if row["packagetype"] == "sdist")
    if source["digests"]["sha256"] != ARCHIVE_SHA256:
        raise ValueError("Pinned doom-py archive fingerprint differs")
    archive = output / "doom-py-0.0.15.tar.gz"
    if not archive.exists():
        with urllib.request.urlopen(source["url"], timeout=60) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != ARCHIVE_SHA256:
            raise ValueError("Downloaded archive fingerprint differs")
        with archive.open("xb") as stream:
            stream.write(data)
    if hashlib.sha256(archive.read_bytes()).hexdigest() != ARCHIVE_SHA256:
        raise ValueError("Existing archive changed")
    files = {}
    with tarfile.open(archive, "r:gz") as bundle:
        candidates = [
            member
            for member in bundle.getmembers()
            if member.isfile()
            and (
                member.name.endswith("/scenarios/take_cover.wad")
                or member.name.endswith("/scenarios/freedoom2.wad")
                or member.name.endswith("/download_freedoom.sh")
                or member.name.endswith("/doom_py/__init__.py")
            )
        ]
        for member in candidates:
            # Read selected members; never extract archive paths or execute code.
            data = bundle.extractfile(member).read()
            path = output / Path(member.name).name
            if path.exists():
                if path.read_bytes() != data:
                    raise ValueError(f"Preserve changed local asset: {path}")
            else:
                with path.open("xb") as stream:
                    stream.write(data)
            files[path.name] = dict(
                member=member.name,
                bytes=len(data),
                sha256=hashlib.sha256(data).hexdigest(),
            )
    manifest = dict(
        metadata_url=metadata_url,
        archive_url=source["url"],
        archive_sha256=ARCHIVE_SHA256,
        files=files,
        purpose="Read-only legacy asset comparison, not a doom-py installation",
    )
    manifest_path = output / "asset_manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if any(previous[key] != manifest[key] for key in manifest if key != "files"):
            raise ValueError("Preserve changed asset manifest")
        if any(files.get(name) != row for name, row in previous["files"].items()):
            raise ValueError("Previously recorded assets differ")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest), flush=True)
    fetch_freedoom(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    fetch(args.output_dir)


if __name__ == "__main__":
    main()
