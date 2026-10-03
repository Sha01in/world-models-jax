"""An isolated CPU fixture checks real CMA checkpoint and RNG persistence."""

import os
from pathlib import Path
import subprocess
import sys
import unittest


class TestReferenceControllerCheckpoint(unittest.TestCase):
    def test_fixture_keeps_baseline_and_restores_next_cma_population(self):
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--fixture"],
            cwd=Path(__file__).resolve().parents[1],
            env=dict(os.environ, JAX_PLATFORMS="cpu", OPENBLAS_NUM_THREADS="4"),
            text=True,
            capture_output=True,
            timeout=90,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("PASS: CPU fixture; CMA/RNG continuation verified", result.stdout)


def fixture():
    import hashlib
    import json
    import pickle
    import runpy
    import tempfile
    from unittest.mock import patch

    import jax.numpy as jnp
    import numpy as np

    namespace = runpy.run_path("scripts/tools/train_doom_reference_controller.py")
    entry = namespace["main"]
    globals_dict = entry.__globals__

    class TinyWorld:
        hidden_size = 512

        def __call__(self, inputs, hidden, restart):
            h, c = (jnp.where(restart > 0.5, 0, value) for value in hidden)
            return (
                jnp.zeros((1, 64)),
                jnp.zeros((1, 64)),
                jnp.full((1, 64), -100.0),
                jnp.zeros(1),
                jnp.where(h[:1] >= 1, 1.0, -1.0),
            ), (h + 1, c + 2)

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "fixture.json"
        source.write_text("CPU checkpoint fixture; not pretrained world weights\n")
        payload = {
            "controller.json": [np.zeros(1088).tolist()],
            "initial_z.json": np.zeros((2, 10, 64)).tolist(),
        }
        manifest = dict(
            reference_commit="cpu_fixture",
            files={
                "fixture.json": dict(
                    sha256=hashlib.sha256(source.read_bytes()).hexdigest()
                )
            },
        )
        globals_dict["load_author_models"] = lambda _: (
            None,
            TinyWorld(),
            None,
            manifest,
        )
        globals_dict["checked_reference_arrays"] = lambda _: (payload, manifest)
        output = root / "policy.npz"
        # Only the CLI's dispatch guard is mocked. JAX devices remain CPU and
        # the fixture's metadata explicitly records that. No real GPU run.
        with (
            patch("jax.default_backend", return_value="gpu"),
            patch.object(
                sys,
                "argv",
                [
                    "fixture",
                    "--reference-dir",
                    str(root),
                    "--output",
                    str(output),
                    "--generations",
                    "2",
                    "--pop-size",
                    "4",
                    "--rollouts",
                    "2",
                    "--candidate-batch",
                    "4",
                    "--validation-rollouts",
                    "2",
                    "--validate-every",
                    "1",
                ],
            ),
        ):
            entry()
        resume = json.loads(Path(str(output) + ".resume.json").read_text())
        assert resume["complete"] and resume["generation"] == 2
        assert resume["best_generation"] == 0  # No improved fitness in this world.
        with np.load(output) as data:
            np.testing.assert_array_equal(data["params"], np.zeros(1088))
        optimizer_bytes = Path(str(output) + ".optimizer.pkl").read_bytes()
        assert hashlib.sha256(optimizer_bytes).hexdigest() == resume["optimizer_sha256"]
        populations = []
        for _ in range(2):
            saved = pickle.loads(optimizer_bytes)
            assert saved["optimizer"].countiter == 2
            with np.load(root / "policy.last.npz") as data:
                np.testing.assert_array_equal(data["params"], saved["optimizer"].mean)
            np.random.set_state(saved["numpy_random_state"])
            populations.append(saved["optimizer"].ask())
        np.testing.assert_array_equal(*populations)
        print("PASS: CPU fixture; CMA/RNG continuation verified")


if __name__ == "__main__":
    if "--fixture" in sys.argv:
        fixture()
    else:
        unittest.main()
