"""CPU checks for selected warm starts and fresh-policy comparison gates."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.tools.audit_doom_reference_comparison_v2 import (
    frozen_choice_audit,
    real_result_audit,
    write_audit,
)
from scripts.tools.run_doom_reference_comparison_v2 import comparison
from scripts.tools.audit_doom_reference_real_training import audit_search_capsule
from scripts.tools.train_doom_reference_real import run_search
from src.doom_reference_comparison_v2 import digest, identity, public_parameters
from src.doom_reference_initializer import registered_initializer
from tests.test_direct_real_comparison import Fixture
from tests.test_doom_real_training import SyntheticFitness, fixture_settings


class WarmFixture(Fixture):
    def __init__(self, root):
        super().__init__(root)
        self.protocol.update(
            public_parameters_sha256=identity(self.public),
            initializer_parameters_sha256=identity(np.full(1088, -0.001)),
            environment_template_report=str(self.parent_report),
        )
        self.protocol_path.write_text(json.dumps(self.protocol))
        for path in (self.best,):
            metadata_path = Path(str(path) + ".json")
            metadata = json.loads(metadata_path.read_text())
            metadata["input_sha256"] = self.protocol["frozen_inputs"]
            metadata_path.write_text(json.dumps(metadata))
        audit = json.loads(self.training_audit.read_text())
        audit["protocol_sha256"] = digest(self.protocol_path)
        self.training_audit.write_text(json.dumps(audit))

    def audit_choice(self, protocol, freeze, output):
        self.events.append("audit_choice")
        write_audit(output, frozen_choice_audit(protocol, freeze, resamples=300))

    def audit_result(self, protocol, result, output):
        self.events.append("audit_result")
        write_audit(output, real_result_audit(protocol, result, resamples=300))

    def run(self, **callbacks):
        return comparison(
            self.protocol_path,
            self.root,
            self.prefix,
            self.training_audit,
            evaluate=callbacks.get("evaluate", self.evaluate),
            audit_choice=callbacks.get("audit_choice", self.audit_choice),
            audit_result=self.audit_result,
            source_hashes={
                "src/doom_reference_comparison_v2.py": digest(
                    "src/doom_reference_comparison_v2.py"
                )
            },
        )


class TestWarmStart(unittest.TestCase):
    def test_nonzero_initializer_and_larger_cohorts_replay_exactly(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "warm.npz"
            settings = fixture_settings(root, output)
            settings["arguments"].update(
                generations=1, fitness_games=8, holdout_games=64
            )
            initial = np.full(1088, -0.001)
            run_search(settings, SyntheticFitness(), initial)
            result = audit_search_capsule(output, initial)
            self.assertEqual(result["raw_training_games"], 160)
            self.assertTrue(
                result[
                    "population_weights_optimizer_history_and_next_rng_reconstructed"
                ]
            )
            with self.assertRaises(ValueError):
                audit_search_capsule(output, np.zeros(1088))

    def test_public_control_is_distinct_from_search_initializer(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = WarmFixture(Path(directory))
            np.testing.assert_array_equal(
                public_parameters(fixture.protocol), fixture.public
            )
            result = fixture.run()
            self.assertTrue(result["new_policy_won"])
            self.assertEqual(len(result["test_tasks"]), 2)
            self.assertEqual(fixture.events[:4], ["validation"] * 4)
            self.assertLess(
                fixture.events.index("audit_choice"), fixture.events.index("test")
            )

    def test_failed_independent_audit_prevents_reserved_dispatch(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = WarmFixture(Path(directory))

            def reject(*args):
                raise ValueError("Synthetic independent audit rejected choice")

            with self.assertRaisesRegex(ValueError, "rejected choice"):
                fixture.run(audit_choice=reject)
            self.assertNotIn("test", fixture.events)

    def test_initializer_reconstructs_complete_prior_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            fixture.scores["val_prior_initializer"] = 1000
            result = fixture.run()
            self.assertFalse(result["new_policy_won"])
            selected_path = str(
                fixture.root / f"{fixture.prefix}_frozen_selection.json"
            )
            audit_path = str(
                fixture.root / f"{fixture.prefix}_frozen_selection_cpu_audit.json"
            )
            protocol = dict(
                fixture.protocol,
                initializer_selection_frozen=selected_path,
                initializer_selection_cpu_audit=audit_path,
                initializer_parent_protocol=str(fixture.protocol_path),
                initializer_controller=str(fixture.prior),
                initializer_parameters_sha256=identity(np.full(1088, -0.001)),
            )
            protocol["frozen_inputs"] = dict(protocol["frozen_inputs"])
            for path in (selected_path, audit_path):
                protocol["frozen_inputs"][path] = digest(path)
            np.testing.assert_array_equal(
                registered_initializer(protocol), np.full(1088, -0.001)
            )
            protocol["initializer_controller"] = str(fixture.best)
            with self.assertRaisesRegex(ValueError, "validation-selected"):
                registered_initializer(protocol)
            protocol["initializer_controller"] = str(fixture.prior)
            report = json.loads(Path(result["selected"]["report"]).read_text())
            report["episodes_detail"].pop()
            Path(result["selected"]["report"]).write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "cohort/hash"):
                registered_initializer(protocol)


if __name__ == "__main__":
    unittest.main()
