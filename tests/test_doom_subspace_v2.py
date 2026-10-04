"""Projected parent provenance and64-game search recovery on CPU fixtures."""

from contextlib import redirect_stdout
from importlib.metadata import version
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts.tools import train_doom_reference_subspace_v2 as trainer
from scripts.tools import run_doom_reference_subspace_comparison_v2 as supervisor
from scripts.tools.run_doom_reference_subspace_comparison_v2 import comparison
from src.doom_subspace_audit import audit_subspace_capsule
from src.doom_subspace_comparison import digest, identity
from src.doom_subspace_initializer_v2 import registered_initializer
from src.doom_subspace_training import METHOD, run_subspace_search
from tests.test_doom_real_training import SyntheticFitness
from tests.test_doom_subspace_workflow import SubspaceFixture, settings_for


def limits(output):
    return dict(
        generations=4,
        pop_size=8,
        fitness_games=64,
        holdout_games=64,
        workers=8,
        validate_every=4,
        sigma=0.25,
        seed=97,
        temperature=None,
        output=str(output),
    )


def parent_fixture(root):
    parent = SubspaceFixture(root)
    parent.protocol["training_method"] = METHOD
    parent.protocol_path.write_text(json.dumps(parent.protocol))
    audit = json.loads(parent.training_audit.read_text())
    audit["protocol_sha256"] = digest(parent.protocol_path)
    parent.training_audit.write_text(json.dumps(audit))
    result = parent.run()
    selection = str(root / f"{parent.prefix}_frozen_selection.json")
    audit_path = str(root / f"{parent.prefix}_frozen_selection_cpu_audit.json")
    child = dict(
        parent.protocol,
        initializer_selection_frozen=selection,
        initializer_selection_cpu_audit=audit_path,
        initializer_parent_protocol=str(parent.protocol_path),
        initializer_controller=result["selected"]["controller"],
        initializer_parameters_sha256=result["selected"]["parameters_sha256"],
    )
    child["frozen_inputs"] = dict(parent.protocol["frozen_inputs"])
    for path in (selection, audit_path, str(parent.protocol_path)):
        child["frozen_inputs"][path] = digest(path)
    return parent, result, child


class TestProjectedParent(unittest.TestCase):
    def test_complete_projected_parent_and_no_reserved_outcomes_needed(self):
        with tempfile.TemporaryDirectory() as directory:
            parent, result, child = parent_fixture(Path(directory))
            with np.load(child["initializer_controller"], allow_pickle=False) as a:
                expected = a["params"].copy()
            np.testing.assert_array_equal(registered_initializer(child), expected)
            for case in result["test_tasks"]:
                Path(case["report"]).write_text("not a test result")
            np.testing.assert_array_equal(registered_initializer(child), expected)
            self.assertEqual(identity(expected), child["initializer_parameters_sha256"])
            self.assertTrue(parent.events)

    def test_incomplete_parent_validation_or_foreign_choice_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            _, result, child = parent_fixture(Path(directory))
            foreign = dict(child, initializer_parameters_sha256="0" * 64)
            with self.assertRaisesRegex(ValueError, "validation-selected"):
                registered_initializer(foreign)
            f = Path(result["selected"]["report"])
            value = json.loads(f.read_text())
            value["episodes_detail"].pop()
            f.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "cohort/hash"):
                registered_initializer(child)

    def test_forged_projected_weights_are_rejected_after_checksum_update(self):
        with tempfile.TemporaryDirectory() as directory:
            parent, _, child = parent_fixture(Path(directory))
            actor = Path(child["initializer_controller"])
            with np.load(actor, allow_pickle=False) as a:
                values = {k: a[k].copy() for k in a.files}
            values["log_gains"][0] += 0.01
            with actor.open("wb") as f:
                np.savez(f, **values)
            pointer = Path(str(parent.best) + ".resume.json")
            value = json.loads(pointer.read_text())
            key = "best_sha256" if actor == parent.best else "last_sha256"
            value[key] = digest(actor)
            pointer.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "Projected"):
                registered_initializer(child)

    def test_actual_cpu_cli_accepts64_without_loading_gpu_models(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent, _, child = parent_fixture(root)
            child["arguments"] = dict(
                limits(root / "child.npz"),
                reference_dir=parent.arguments["reference_dir"],
                asset_dir="unused_cpu_fixture",
                fitness_seed=560000,
                holdout_seed=570000,
            )
            child.update(
                training_method=METHOD,
                validation_seed_range=[130820, 130899],
                test_seed_range=[250000, 250099],
                package_versions={
                    n: version(n)
                    for n in (
                        "jax",
                        "jaxlib",
                        "equinox",
                        "vizdoom",
                        "numpy",
                        "pillow",
                        "cma",
                    )
                },
            )
            path = root / "child_protocol.json"
            path.write_text(json.dumps(child))
            output = io.StringIO()
            with (
                patch("sys.argv", ["train", "--protocol", str(path), "--check-only"]),
                patch.object(
                    trainer,
                    "load_author_models",
                    side_effect=AssertionError("GPU model loading"),
                ),
                redirect_stdout(output),
            ):
                trainer.main()
            result = json.loads(output.getvalue())
            self.assertTrue(result["ready"])
            self.assertEqual(result["fitness_games"], 64)
            self.assertEqual(result["gpu_jobs_started"], 0)


class TestLargerFitness(unittest.TestCase):
    def test_limits_preserve_bounded_compute_and_no_dream_temperature(self):
        args = limits(Path("controller.npz"))
        trainer.validate_training_limits(args)
        for key, value in (
            ("fitness_games", 65),
            ("generations", 5),
            ("pop_size", 9),
            ("workers", 9),
            ("holdout_games", 65),
            ("sigma", np.nan),
            ("temperature", 1.15),
            ("seed", True),
        ):
            with self.subTest(key=key), self.assertRaises(ValueError):
                trainer.validate_training_limits(dict(args, **{key: value}))

    def test64_game_partial_recovery_reconstructs_all_populations_and_rng(self):
        initial = np.full(1088, -0.001)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            full, recovered = root / "full.npz", root / "recovered.npz"
            one, two = (
                settings_for(root, full, initial),
                settings_for(root, recovered, initial),
            )
            for settings in (one, two):
                settings["arguments"]["fitness_games"] = 64
            run_subspace_search(one, SyntheticFitness(), initial)
            with self.assertRaises(KeyboardInterrupt):
                run_subspace_search(
                    two, SyntheticFitness(interrupt_fitness=True), initial
                )
            run_subspace_search(two, SyntheticFitness(), initial, resume=True)
            for path in (full, recovered):
                audit = audit_subspace_capsule(path, initial)
                self.assertTrue(
                    audit[
                        "population_weights_optimizer_history_and_next_rng_reconstructed"
                    ]
                )
                self.assertTrue(
                    audit[
                        "projected1088_weights_and_three_dimensional_optimizer_reconstructed"
                    ]
                )
                self.assertGreaterEqual(audit["raw_training_games"], 512)
            for suffix in ("", ".last"):
                a = full.with_name(full.stem + suffix + ".npz")
                b = recovered.with_name(recovered.stem + suffix + ".npz")
                with (
                    np.load(a, allow_pickle=False) as x,
                    np.load(b, allow_pickle=False) as y,
                ):
                    for key in x.files:
                        np.testing.assert_array_equal(x[key], y[key])

    def test_new_supervisor_keeps_full_validation_before_reserved_dispatch(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = SubspaceFixture(Path(directory))
            result = comparison(
                fixture.protocol_path,
                fixture.root,
                fixture.prefix,
                fixture.training_audit,
                evaluate=fixture.evaluate,
                audit_choice=fixture.audit_choice,
                audit_result=fixture.audit_result,
                source_hashes={
                    "scripts/tools/run_doom_reference_subspace_comparison_v2.py": digest(
                        "scripts/tools/run_doom_reference_subspace_comparison_v2.py"
                    )
                },
            )
            self.assertTrue(result["new_policy_won"])
            self.assertEqual(fixture.events[:4], ["validation"] * 4)
            self.assertLess(
                fixture.events.index("audit_choice"), fixture.events.index("test")
            )

    def test_ready_gate_requires_actual_exit_and_projected_parent_evidence(self):
        protocol = dict(
            search_space={"dimensions": 3},
            initializer_parameters_sha256="weights",
            initializer_parent_protocol="parent.json",
            maximum_training_fitness_games=2048,
            maximum_training_holdout_games=128,
        )
        state = {}
        audit = dict(
            protocol_sha256="protocol",
            population_weights_optimizer_history_and_next_rng_reconstructed=True,
            all_raw_game_pairs_verified=True,
            projected1088_weights_and_three_dimensional_optimizer_reconstructed=True,
            projected_parent_initializer_reconstructed=True,
            initializer_parent_protocol_sha256="parent",
            training_method=METHOD,
            search_space=protocol["search_space"],
            initializer_parameters_sha256="weights",
            raw_training_games=2176,
            files={},
        )

        def read(path):
            return state if str(path).endswith("task_state.json") else audit

        with (
            patch.object(supervisor, "read", side_effect=read),
            patch.object(
                supervisor,
                "digest",
                side_effect=lambda path: (
                    "parent" if str(path) == "parent.json" else "protocol"
                ),
            ),
            patch.object(Path, "exists", return_value=True),
        ):
            path, missing = supervisor.current_training_ready(protocol)
            self.assertIsNone(path)
            self.assertTrue(missing)
            state.update(
                reference_round16_training_supervisor_exit_verified=True,
                reference_round16_training_session_exit_code=0,
            )
            audit["projected_parent_initializer_reconstructed"] = False
            with self.assertRaisesRegex(ValueError, "capsule differs"):
                supervisor.current_training_ready(protocol)
            audit["projected_parent_initializer_reconstructed"] = True
            audit["initializer_parent_protocol_sha256"] = "foreign"
            with self.assertRaisesRegex(ValueError, "capsule differs"):
                supervisor.current_training_ready(protocol)
            audit["initializer_parent_protocol_sha256"] = "parent"
            path, missing = supervisor.current_training_ready(protocol)
            self.assertIsNotNone(path)
            self.assertFalse(missing)


if __name__ == "__main__":
    unittest.main()
