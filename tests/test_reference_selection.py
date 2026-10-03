import copy
import unittest

import numpy as np

from src.doom_reference_selection import reference_validation_decision
from tests.test_reference_results import report


def case(name, score, kind, *, generation=1):
    task = dict(
        name=name,
        controller=name + ".npz",
        inference="posterior",
        seed=130020,
        episodes=80,
        kind=kind,
        parameters_sha256=name,
    )
    value = report(np.full(80, score))
    value["protocol"]["evaluation_role"] = "validation"
    value["protocol"]["controller_provenance"].update(
        controller=task["controller"],
        generation=generation,
        own_policy_update=generation > 0,
    )
    for index, row in enumerate(value["episodes_detail"]):
        row["seed"] = 130020 + index
    return task, value


class TestReferenceSelection(unittest.TestCase):
    def test_only_updated_candidate_winning_validation_triggers_reserved_tests(self):
        tasks, reports = map(
            list,
            zip(
                case("prior", 1000, "control"),
                case("supplied", 950, "control", generation=0),
                case("new", 1200, "candidate"),
            ),
        )
        self.assertTrue(reference_validation_decision(tasks, reports)["new_policy_won"])
        tasks[2], reports[2] = case("new", 1000, "candidate")
        decision = reference_validation_decision(tasks, reports)
        self.assertEqual(decision["selected"]["name"], "prior")
        self.assertFalse(decision["new_policy_won"])
        tasks[2], reports[2] = case("new", 1200, "candidate", generation=0)
        self.assertFalse(
            reference_validation_decision(tasks, reports)["new_policy_won"]
        )

    def test_duplicate_policy_or_missing_mismatched_validation_is_rejected(self):
        tasks, reports = map(
            list, zip(case("prior", 1000, "control"), case("new", 1200, "candidate"))
        )
        duplicate = copy.deepcopy(tasks)
        duplicate[1]["parameters_sha256"] = duplicate[0]["parameters_sha256"]
        with self.assertRaisesRegex(ValueError, "deduplicated"):
            reference_validation_decision(duplicate, reports)
        missing = copy.deepcopy(reports)
        missing[1]["episodes_detail"].pop()
        with self.assertRaises(ValueError):
            reference_validation_decision(tasks, missing)
        different = copy.deepcopy(tasks)
        different[1]["seed"] += 1
        with self.assertRaisesRegex(ValueError, "share seeds"):
            reference_validation_decision(different, reports)


if __name__ == "__main__":
    unittest.main()
