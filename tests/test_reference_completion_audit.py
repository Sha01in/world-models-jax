import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.tools.summarize_doom_reference_results import audit_result, digest
from tests.test_reference_results import report


class TestReferenceCompletionAudit(unittest.TestCase):
    def test_full_audit_rejects_selection_changed_after_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            policy = root / "policy.npz"
            policy.write_bytes(b"CPU fixture policy, not actual trained weights")
            world = {}
            for name in ("vae.json", "rnn.json", "take_cover.wad", "freedoom2.wad"):
                path = root / name
                path.write_bytes(b"CPU fixture world input")
                world[str(path)] = digest(path)
            source = root / "source.txt"
            source.write_bytes(b"CPU audit fixture source")
            tasks, validation = [], []
            for name, score, controller in (
                ("control", 1000, None),
                ("own", 1150, str(policy)),
            ):
                path = root / (name + ".json")
                task = dict(
                    name=name,
                    controller=controller,
                    inference="posterior",
                    role="validation",
                    seed=130000,
                    episodes=20,
                    report=str(path),
                )
                value = report(np.full(20, score), own=bool(controller))
                value["protocol"]["input_sha256"] = dict(world)
                value["protocol"]["source_sha256"] = {str(source): digest(source)}
                value["protocol"]["evaluation_role"] = "validation"
                value["measured_at"] = "2026-10-03T21:00:00+00:00"
                if controller:
                    value["protocol"]["controller_provenance"]["controller"] = (
                        controller
                    )
                    value["protocol"]["input_sha256"][controller] = digest(controller)
                for index, row in enumerate(value["episodes_detail"]):
                    row["seed"] = 130000 + index
                path.write_text(json.dumps(value))
                tasks.append(task)
                validation.append(value)
            freeze_path = root / "freeze.json"
            freeze = dict(
                frozen_at="2026-10-03T21:01:00+00:00",
                selected=tasks[1],
                validation_tasks=tasks,
                validation_scores={"control": 1000.0, "own": 1150.0},
                validation_report_sha256={
                    t["report"]: digest(t["report"]) for t in tasks
                },
                selected_controller_provenance=validation[1]["protocol"][
                    "controller_provenance"
                ],
                frozen_inputs=validation[1]["protocol"]["input_sha256"],
                frozen_source=validation[1]["protocol"]["source_sha256"],
                test_seed_range=[140000, 140099],
            )
            freeze_path.write_text(json.dumps(freeze))
            test_paths = []
            for index, score in enumerate((1200, 1050)):
                value = report(np.full(100, score), own=index == 0)
                value["protocol"] = copy.deepcopy(validation[1 - index]["protocol"])
                value["protocol"]["evaluation_role"] = "test"
                value["measured_at"] = "2026-10-03T21:02:00+00:00"
                path = root / f"test{index}.json"
                path.write_text(json.dumps(value))
                test_paths.append(path)
            result = dict(
                selected=tasks[1],
                frozen_selection=str(freeze_path),
                frozen_selection_sha256=digest(freeze_path),
                selected_test=str(test_paths[0]),
                selected_test_sha256=digest(test_paths[0]),
                control_test=str(test_paths[1]),
                control_test_sha256=digest(test_paths[1]),
                test_mean=1200.0,
                test_std=0.0,
                own_policy_update=True,
            )
            result_path = root / "result.json"
            result_path.write_text(json.dumps(result))
            analysis = audit_result(result_path, resamples=128)
            self.assertEqual(analysis["paired_gain95"], [150.0, 150.0])
            self.assertTrue(analysis["selection_recomputed_from_validation_only"])
            self.assertTrue(analysis["completion_still_requires_review"])
            freeze["selected"] = tasks[0]
            freeze_path.write_text(json.dumps(freeze))
            result["selected"] = tasks[0]
            result["frozen_selection_sha256"] = digest(freeze_path)
            result_path.write_text(json.dumps(result))
            with self.assertRaisesRegex(ValueError, "validation winner"):
                audit_result(result_path, resamples=128)


if __name__ == "__main__":
    unittest.main()
