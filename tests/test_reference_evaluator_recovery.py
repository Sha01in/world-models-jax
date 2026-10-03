"""Isolated CPU checks for reference policy provenance and failed-row recovery."""

import os
from pathlib import Path
import subprocess
import sys
import unittest


class TestReferenceEvaluatorRecovery(unittest.TestCase):
    def test_policy_world_mismatch_and_failed_record_are_preserved(self):
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--fixture"],
            cwd=Path(__file__).resolve().parents[1],
            env=dict(os.environ, JAX_PLATFORMS="cpu", OPENBLAS_NUM_THREADS="4"),
            text=True,
            capture_output=True,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn(
            "PASS: world identity and failing real record recovery", result.stdout
        )


def fixture():
    import hashlib
    import json
    import runpy
    import tempfile
    from unittest.mock import patch

    import numpy as np

    namespace = runpy.run_path("scripts/tools/evaluate_doom_reference.py")
    entry = namespace["main"]
    functions = entry.__globals__
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        manifest = dict(reference_commit="cpu_fixture", files={})
        for name in ("vae.json", "rnn.json", "controller.json"):
            path = root / name
            path.write_text("CPU fixture data; not public model weights\n")
            manifest["files"][name] = dict(
                sha256=hashlib.sha256(path.read_bytes()).hexdigest()
            )
        for name in ("take_cover.wad", "freedoom2.wad"):
            (root / name).write_bytes(b"CPU fixture assets")
        controller = root / "policy.npz"
        for path, generation in ((controller, 1), (root / "policy.last.npz", 2)):
            np.savez(
                path,
                params=np.full(1088, generation / 100.0),
                generation=generation,
                type="reference_linear",
                state_mode="ch",
                posterior_sampling=True,
                canonical_actions=False,
                temperature=1.15,
            )
        settings = dict(
            reference_commit="cpu_fixture",
            arguments=dict(reference_dir=str(root), temperature=1.15),
            input_sha256={
                str(root / name): row["sha256"]
                for name, row in manifest["files"].items()
            },
            source_sha256={
                "src/doom_reference.py": namespace["digest"]("src/doom_reference.py")
            },
        )
        settings_path = Path(str(controller) + ".json")
        settings_path.write_text(json.dumps(settings))
        resume_path = Path(str(controller) + ".resume.json")
        resume_path.write_text(
            json.dumps(
                dict(
                    complete=True,
                    best_generation=1,
                    generation=2,
                    best_sha256=namespace["digest"](controller),
                    last_sha256=namespace["digest"](root / "policy.last.npz"),
                )
            )
        )
        for name in ("policy.npz", "policy.last.npz"):
            params, evidence, _ = namespace["load_controller"](root / name, manifest)
            assert params.shape == (1088,) and evidence["own_policy_update"]
        settings["input_sha256"][str(root / "rnn.json")] = "0" * 64
        settings_path.write_text(json.dumps(settings))
        try:
            namespace["load_controller"](controller, manifest)
            raise AssertionError("Mismatched recurrent world was accepted")
        except ValueError as error:
            assert "different latent world" in str(error)
        settings["input_sha256"][str(root / "rnn.json")] = manifest["files"][
            "rnn.json"
        ]["sha256"]
        settings_path.write_text(json.dumps(settings))
        payload = {"controller.json": [np.zeros(1088).tolist()]}
        functions["load_author_models"] = lambda _: (None, None, None, manifest)
        functions["checked_reference_arrays"] = lambda _: (payload, manifest)
        functions["make_reference_policy"] = lambda *a, **kw: None
        good = dict(
            seed=123000,
            survival_steps=4,
            score=4.0,
            actions_left_right_wait=[0, 0, 4],
            terminated=True,
            truncated=False,
        )
        bad = dict(
            seed=123001,
            survival_steps=5,
            score=0.0,
            actions_left_right_wait=[0, 0, 5],
            terminated=True,
            truncated=False,
        )

        def failed_games(*args, **kwargs):
            kwargs["on_episode"](good)
            kwargs["on_episode"](bad)

        functions["evaluate_parallel"] = failed_games
        output = root / "real.json"
        arguments = [
            "fixture",
            "--reference-dir",
            str(root),
            "--asset-dir",
            str(root),
            "--output",
            str(output),
            "--episodes",
            "2",
            "--seed",
            "123000",
            "--controller",
            str(controller),
            "--inference",
            "mean",
            "--role",
            "validation",
        ]
        with (
            patch("jax.default_backend", return_value="gpu"),
            patch.object(sys, "argv", arguments),
        ):
            try:
                entry()
                raise AssertionError("Invalid reward was accepted")
            except ValueError as error:
                assert "123001" in str(error)
            saved = json.loads(output.with_suffix(".partial.json").read_text())
            failure = json.loads(output.with_suffix(".failed.json").read_text())
            assert saved["episodes_detail"] == [good]
            assert failure["offending_record"] == bad and failure["actual_rows"] == [
                good,
                bad,
            ]
            assert not failure["protocol"]["inference_posterior_sampling"]
            assert failure["protocol"]["evaluation_role"] == "validation"
            try:
                entry()
                raise AssertionError("Captured failure was blindly resumed")
            except FileExistsError:
                pass
            assert not output.exists()
        print("PASS: world identity and failing real record recovery")


if __name__ == "__main__":
    if "--fixture" in sys.argv:
        fixture()
    else:
        unittest.main()
