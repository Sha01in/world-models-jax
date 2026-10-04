"""Synthetic CPU fixtures for fresh-validation and reserved-test gates."""

from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.tools.audit_doom_reference_comparison import (
    frozen_choice_audit,
    real_result_audit,
    write_audit,
)
from scripts.tools.run_doom_reference_comparison import comparison, write_new
from src.doom_reference_comparison import (
    archive_info,
    checked_partial,
    digest,
    identity,
    recorded_seed_audit,
    validation_cases,
)


class Fixture:
    """No game engine or GPU is invoked; all fixture reports are temporary."""

    def __init__(self, root, *, duplicate_best=False, exclude_last=False):
        self.root = root
        self.prefix = "synthetic_direct"
        self.ref = root / "reference"
        self.ref.mkdir()
        self.public = np.zeros(1088, dtype=np.float64)
        for name in ("vae.json", "rnn.json", "take_cover.wad", "freedoom2.wad"):
            (self.ref / name).write_text(name)
        (self.ref / "controller.json").write_text(json.dumps([self.public.tolist()]))
        self.world_files = {str(path): digest(path) for path in self.ref.iterdir()}
        self.sources = {
            path: digest(path)
            for path in (
                "src/doom_reference.py",
                "src/doom_reference_env.py",
                "src/doom_evaluation.py",
                str(Path("scripts/tools/evaluate_doom_reference.py").resolve()),
            )
        }
        packages = {
            name: version(name)
            for name in (
                "jax",
                "jaxlib",
                "equinox",
                "vizdoom",
                "numpy",
                "pillow",
                "cma",
            )
        }
        self.world = dict(
            reference_commit="synthetic-reference",
            input_sha256=self.world_files,
            source_sha256=self.sources,
            preprocessing="synthetic-template",
            difficulty=4,
            episode_start_time=14,
            episode_timeout=2100,
            threshold=0.3333,
            world_precision="FP32",
            posterior_and_controller_precision="FP64",
            rng="Per explicit game seed",
            package_versions={k: v for k, v in packages.items() if k != "cma"},
            inference_posterior_sampling=True,
        )
        self.prior = root / "prior.npz"
        self.best = root / "direct.npz"
        self.last = root / "direct.last.npz"
        self.arguments = dict(
            reference_dir=str(self.ref),
            output=str(self.best),
            generations=16,
            temperature=None,
        )
        self.parent_report = root / "parent_public.json"
        write_new(self.parent_report, dict(protocol=self.world))
        self.parent_audit = root / "parent_audit.json"
        write_new(
            self.parent_audit, dict(selected=dict(report=str(self.parent_report)))
        )
        inputs = dict(self.world_files)
        inputs[str(self.parent_report)] = digest(self.parent_report)
        inputs[str(self.parent_audit)] = digest(self.parent_audit)
        self.protocol = dict(
            arguments=self.arguments,
            reference_commit="synthetic-reference",
            frozen_inputs=inputs,
            frozen_source={"src/doom_reference.py": digest("src/doom_reference.py")},
            package_versions=packages,
            initializer_parameters_sha256=identity(self.public),
            initializer_selection_cpu_audit=str(self.parent_audit),
            maximum_training_fitness_games=1024,
            maximum_training_holdout_games=80,
            validation_games=80,
            validation_seed_range=[130580, 130659],
            test_games=100,
            test_seed_range=[220000, 220099],
            maximum_validation_policies=4,
            previous_failed_candidates={},
        )
        self._archive(self.prior, np.full(1088, -0.001), 500, direct=False)
        best_weights = self.public if duplicate_best else np.full(1088, 0.001)
        self._archive(self.best, best_weights, 8, direct=True)
        self._archive(self.last, np.full(1088, 0.002), 16, direct=True, base=self.best)
        self.protocol["validation_controls"] = [
            dict(
                name="val_supplied",
                controller=None,
                kind="control",
                parameters_sha256=identity(self.public),
            ),
            dict(
                name="val_prior_initializer",
                controller=str(self.prior),
                kind="control",
                controller_sha256=digest(self.prior),
            ),
        ]
        if exclude_last:
            previous = root / "failed_selection.json"
            write_new(previous, dict(synthetic_previous_failure=True))
            self.protocol["previous_failed_candidates"] = {
                str(self.last): dict(
                    checkpoint_sha256=digest(self.last),
                    parameters_sha256=identity(np.full(1088, 0.002)),
                    source_frozen_selection=str(previous),
                    source_frozen_selection_sha256=digest(previous),
                )
            }
        self.protocol_path = root / "protocol.json"
        write_new(self.protocol_path, self.protocol)
        self.training_audit = root / "training_audit.json"
        write_new(
            self.training_audit,
            dict(
                protocol_sha256=digest(self.protocol_path),
                population_weights_optimizer_history_and_next_rng_reconstructed=True,
                all_raw_game_pairs_verified=True,
                raw_training_games=1104,
                files={
                    str(self.best): digest(self.best),
                    str(self.last): digest(self.last),
                },
            ),
        )
        self.events = []
        self.scores = dict(
            val_supplied=700,
            val_prior_initializer=800,
            val_direct_best=950,
            val_direct_last=900,
            test_selected=1200,
            test_control=950,
        )

    def _archive(self, path, params, generation, *, direct, base=None):
        with path.open("wb") as stream:
            np.savez(
                stream,
                params=params,
                generation=generation,
                type="reference_linear",
                state_mode="ch",
                posterior_sampling=True,
                canonical_actions=False,
                training_method="direct_real_survival_cma" if direct else "dream",
            )
        base = path if base is None else base
        metadata_path = Path(str(base) + ".json")
        resume_path = Path(str(base) + ".resume.json")
        if not metadata_path.exists():
            arguments = (
                self.arguments
                if direct
                else dict(reference_dir=str(self.ref), temperature=1.15)
            )
            write_new(
                metadata_path,
                dict(
                    arguments=arguments,
                    reference_commit="synthetic-reference",
                    input_sha256=self.protocol["frozen_inputs"]
                    if direct
                    else self.world_files,
                    source_sha256=self.protocol["frozen_source"],
                    training_method="direct_real_survival_cma" if direct else "dream",
                ),
            )
        resume = (
            dict(
                complete=True,
                generation=16 if direct else 500,
                best_generation=generation,
                best_sha256=digest(path),
                last_sha256=digest(path),
            )
            if base == path
            else json.loads(resume_path.read_text())
        )
        if base != path:
            resume.update(last_sha256=digest(path), generation=generation)
        if direct:
            state = self.root / "synthetic_state.pkl"
            state.write_bytes(b"Synthetic fixture only; no actual optimizer")
            resume.update(state_path=str(state), state_sha256=digest(state))
        resume_path.write_text(json.dumps(resume))

    def report(self, case, *, partial=False):
        actor = case["controller"]
        info = (
            archive_info(actor, self.protocol, direct=case["kind"] == "candidate")
            if actor
            else None
        )
        provenance = dict(imported_public_world=True, own_policy_update=bool(actor))
        inputs = dict(self.world_files)
        if actor:
            provenance.update(
                controller=actor,
                generation=info["generation"],
                dream_temperature=info["temperature"],
            )
            inputs.update(info["files"])
        header = dict(
            self.world,
            input_sha256=inputs,
            controller_provenance=provenance,
            evaluation_role=case["role"],
        )
        length = self.scores[case["name"]]
        rows = [
            dict(
                seed=seed,
                survival_steps=length,
                score=float(length),
                actions_left_right_wait=[0, 0, length],
                terminated=length < 2100,
                truncated=length == 2100,
            )
            for seed in range(case["seed"], case["seed"] + case["episodes"])
        ]
        if partial:
            return dict(
                protocol=header,
                expected_seeds=[row["seed"] for row in rows],
                episodes_detail=rows[:5],
            )
        return dict(
            measured_at=datetime.now(timezone.utc).isoformat(),
            protocol=header,
            episodes_detail=rows,
            episodes=len(rows),
            mean=float(length),
            std=0.0,
            deaths=len(rows) if length < 2100 else 0,
            timeouts=len(rows) if length == 2100 else 0,
            workers=8,
        )

    def evaluate(self, protocol, case):
        self.events.append(case["role"])
        if case["role"] == "test":
            assert (
                self.root / f"{self.prefix}_frozen_selection_cpu_audit.json"
            ).exists()
            assert "audit_choice" in self.events
        if not Path(case["report"]).exists():
            write_new(case["report"], self.report(case))

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
                str(Path("src/doom_reference_comparison.py")): digest(
                    "src/doom_reference_comparison.py"
                )
            },
        )


class TestDirectRealComparison(unittest.TestCase):
    def test_nested_records_block_reusing_test_seeds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts, checkpoints = root / "artifacts", root / "checkpoints"
            artifacts.mkdir()
            nested = checkpoints / "search" / "games"
            nested.mkdir(parents=True)
            write_new(nested / "cohort.json", dict(episodes_detail=[dict(seed=220007)]))
            result = recorded_seed_audit(
                dict(test=list(range(220000, 220100))),
                artifact_directory=artifacts,
                checkpoint_directory=checkpoints,
            )
            self.assertEqual(result["overlap"]["test"], [220007])

    def test_incomplete_control_test_cannot_support_target_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            fixture.run()
            path = fixture.root / f"{fixture.prefix}_test_control.json"
            report = json.loads(path.read_text())
            report["episodes_detail"].pop()
            path.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "incomplete"):
                real_result_audit(
                    fixture.protocol_path,
                    fixture.root / f"{fixture.prefix}_real_evaluation_result.json",
                    resamples=300,
                )

    def test_new_winner_is_audited_before_complete_paired_test(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            result = fixture.run()
            self.assertTrue(result["new_policy_won"])
            self.assertEqual(result["paired_control"]["controller"], str(fixture.prior))
            self.assertEqual(
                fixture.events,
                ["validation"] * 4 + ["audit_choice", "test", "test", "audit_result"],
            )
            audit = json.loads(
                (
                    fixture.root / f"{fixture.prefix}_real_result_cpu_audit.json"
                ).read_text()
            )
            self.assertTrue(audit["all200_reserved_raw_records_verified"])
            self.assertEqual(audit["paired_test"]["selected_mean"], 1200)
            self.assertEqual(audit["paired_test"]["paired_gain"], 250)
            self.assertTrue(audit["observed1092"])
            self.assertTrue(
                audit["goal_completion_unproven"]
            )  # Synthetic; not actual goal evidence.
            fixture.run()  # Recovery preserves completed immutable reports/audits.

    def test_controls_win_ties_and_consume_no_reserved_games(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            fixture.scores.update(
                val_supplied=1000, val_prior_initializer=1000, val_direct_best=1000
            )
            result = fixture.run()
            self.assertFalse(result["new_policy_won"])
            self.assertIsNone(result["selected"]["controller"])
            self.assertNotIn("test", fixture.events)
            audit = json.loads(
                (
                    fixture.root / f"{fixture.prefix}_real_result_cpu_audit.json"
                ).read_text()
            )
            self.assertTrue(audit["no_reserved_consumption_verified"])

    def test_independent_audit_rejection_blocks_every_reserved_game(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))

            def reject(*args):
                raise ValueError("independent audit rejected")

            with self.assertRaisesRegex(ValueError, "rejected"):
                fixture.run(audit_choice=reject)
            self.assertNotIn("test", fixture.events)
            self.assertFalse(
                (
                    fixture.root / f"{fixture.prefix}_real_evaluation_result.json"
                ).exists()
            )

    def test_missing_last_cohort_blocks_selection_and_testing(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))

            def missing(protocol, case):
                if case["name"] != "val_direct_last":
                    fixture.evaluate(protocol, case)

            with self.assertRaises(FileNotFoundError):
                fixture.run(evaluate=missing)
            self.assertNotIn("audit_choice", fixture.events)
            self.assertNotIn("test", fixture.events)

    def test_duplicate_and_previously_failed_weights_are_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory), duplicate_best=True, exclude_last=True)
            cases = validation_cases(fixture.protocol, fixture.root, fixture.prefix)
            self.assertEqual([case["kind"] for case in cases], ["control", "control"])
            result = fixture.run()
            self.assertFalse(result["new_policy_won"])
            self.assertNotIn("test", fixture.events)

    def test_partial_recovery_requires_matching_actor_and_unique_valid_games(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            case = validation_cases(fixture.protocol, fixture.root, fixture.prefix)[2]
            path = Path(case["report"]).with_suffix(".partial.json")
            saved = fixture.report(case, partial=True)
            write_new(path, saved)
            self.assertEqual(
                len(checked_partial(fixture.protocol, case, path)["episodes_detail"]), 5
            )
            saved["episodes_detail"].append(saved["episodes_detail"][0])
            path.write_text(json.dumps(saved))
            with self.assertRaisesRegex(ValueError, "duplicate"):
                checked_partial(fixture.protocol, case, path)
            saved = fixture.report(case, partial=True)
            saved["protocol"]["controller_provenance"]["controller"] = str(fixture.last)
            path.write_text(json.dumps(saved))
            with self.assertRaisesRegex(ValueError, "actor"):
                checked_partial(fixture.protocol, case, path)

    def test_world_change_rejects_complete_validation_before_tests(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))

            def changed(protocol, case):
                fixture.evaluate(protocol, case)
                if case["name"] == "val_direct_last":
                    report = json.loads(Path(case["report"]).read_text())
                    report["protocol"]["difficulty"] = 5
                    Path(case["report"]).write_text(json.dumps(report))

            with self.assertRaisesRegex(ValueError, "differs"):
                fixture.run(evaluate=changed)
            self.assertNotIn("test", fixture.events)


if __name__ == "__main__":
    unittest.main()
