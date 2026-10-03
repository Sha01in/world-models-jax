"""Controlled80/100-game fixtures reject changed cohorts and control identities."""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.tools.summarize_doom_reference_results import audit_result, digest
from tests.test_reference_results import report


class TestReferenceControlledAudit(unittest.TestCase):
    def test_nondefault_cohort_and_trained_control_are_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prior, new = root / "prior.npz", root / "new.npz"
            np.savez(prior, params=np.ones(1088))
            np.savez(new, params=np.full(1088, 2.0))
            public = root / "controller.json"
            public.write_text(json.dumps([np.zeros(1088).tolist()]))
            inputs = {str(public): digest(public)}
            for name in ("vae.json", "rnn.json", "take_cover.wad", "freedoom2.wad"):
                path = root / name
                path.write_bytes(b"CPU fixture world, not actual pretrained weights")
                inputs[str(path)] = digest(path)
            source = root / "source.txt"
            source.write_bytes(b"CPU fixture source")
            source_sha = {str(source): digest(source)}
            protocol_path = root / "protocol.json"
            protocol = dict(
                preregistered_at="2026-10-03T20:00:00+00:00",
                arguments=dict(
                    output=str(new), reference_dir=str(root), temperature=1.25
                ),
                previous_selected_policy=str(prior),
                previous_selected_policy_sha256=digest(prior),
                validation_seed_range=[130020, 130099],
                validation_games=80,
                test_seed_range=[150000, 150099],
                test_games=100,
                frozen_inputs=inputs,
                frozen_training_source=source_sha,
            )
            protocol_path.write_text(json.dumps(protocol))
            tasks, validation = [], []
            for name, actor, score, generation in (
                ("prior", str(prior), 1100, 270),
                ("public", None, 1000, 0),
                ("new", str(new), 1200, 210),
            ):
                params = np.full(1088, {"prior": 1.0, "public": 0.0, "new": 2.0}[name])
                task = dict(
                    name=name,
                    controller=actor,
                    inference="posterior",
                    seed=130020,
                    episodes=80,
                    kind="candidate" if name == "new" else "control",
                    parameters_sha256=hashlib.sha256(params.tobytes()).hexdigest(),
                    report=str(root / (name + ".json")),
                )
                value = report(np.full(80, score), own=generation > 0)
                value["protocol"]["input_sha256"] = dict(inputs)
                if actor:
                    value["protocol"]["input_sha256"][actor] = digest(actor)
                value["protocol"]["source_sha256"] = source_sha
                value["protocol"]["evaluation_role"] = "validation"
                value["protocol"]["controller_provenance"].update(
                    controller=actor,
                    generation=generation,
                    dream_temperature=1.25 if name == "new" else 1.15,
                )
                value["measured_at"] = "2026-10-03T21:00:00+00:00"
                for index, row in enumerate(value["episodes_detail"]):
                    row["seed"] = 130020 + index
                Path(task["report"]).write_text(json.dumps(value))
                tasks.append(task)
                validation.append(value)
            freeze_path = root / "freeze.json"
            frozen = dict(
                frozen_at="2026-10-03T21:01:00+00:00",
                protocol_sha256=digest(protocol_path),
                validation_tasks=tasks,
                selected=tasks[2],
                new_policy_won=True,
                validation_scores={
                    t["name"]: r["mean"] for t, r in zip(tasks, validation)
                },
                validation_report_sha256={
                    t["report"]: digest(t["report"]) for t in tasks
                },
                selected_controller_provenance=validation[2]["protocol"][
                    "controller_provenance"
                ],
                frozen_inputs=validation[2]["protocol"]["input_sha256"],
                frozen_source=source_sha,
                test_seed_range=[150000, 150099],
            )
            freeze_path.write_text(json.dumps(frozen))
            test_paths = []
            for name, score, original in (
                ("selected", 1250, validation[2]),
                ("control", 1050, validation[0]),
            ):
                value = report(np.full(100, score))
                value["protocol"] = copy.deepcopy(original["protocol"])
                value["protocol"]["evaluation_role"] = "test"
                value["measured_at"] = "2026-10-03T21:02:00+00:00"
                for index, row in enumerate(value["episodes_detail"]):
                    row["seed"] = 150000 + index
                path = root / (name + "_test.json")
                path.write_text(json.dumps(value))
                test_paths.append(path)
            result_path = root / "result.json"
            result = dict(
                selected=tasks[2],
                protocol_sha256=digest(protocol_path),
                frozen_selection=str(freeze_path),
                frozen_selection_sha256=digest(freeze_path),
                selected_test=str(test_paths[0]),
                selected_test_sha256=digest(test_paths[0]),
                control_test=str(test_paths[1]),
                control_test_sha256=digest(test_paths[1]),
                test_mean=1250.0,
                test_std=0.0,
                own_policy_update=True,
            )
            result_path.write_text(json.dumps(result))
            analysis = audit_result(
                result_path, protocol_path=protocol_path, resamples=128
            )
            self.assertEqual(analysis["paired_gain95"], [200.0, 200.0])
            with self.assertRaisesRegex(ValueError, "Unexpected reserved"):
                audit_result(result_path, resamples=128)
            wrong_control = json.loads(test_paths[1].read_text())
            wrong_control["protocol"]["controller_provenance"]["controller"] = None
            test_paths[1].write_text(json.dumps(wrong_control))
            result["control_test_sha256"] = digest(test_paths[1])
            result_path.write_text(json.dumps(result))
            with self.assertRaisesRegex(ValueError, "controller differs"):
                audit_result(result_path, protocol_path=protocol_path, resamples=128)


if __name__ == "__main__":
    unittest.main()
