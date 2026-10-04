"""CPU recovery, optimizer replay, projected inference and reserved-test gates."""

from datetime import datetime, timezone
import json
from pathlib import Path
import pickle
import tempfile
import unittest

import cma
import jax
import jax.numpy as jnp
import numpy as np

from scripts.tools.audit_doom_reference_subspace_comparison import (
    frozen_choice_audit,
    real_result_audit,
    write_audit,
)
from scripts.tools.run_doom_reference_subspace_comparison import comparison
from src.doom_controller_gains import controller_from_log_gains
from src.doom_real_training import make_population_policy, parameter_fingerprint
from src.doom_reference import make_reference_policy
from src.doom_subspace_audit import audit_subspace_capsule
from src.doom_subspace_comparison import archive_info, digest, identity
from src.doom_subspace_initializer import registered_initializer
from src.doom_subspace_training import (
    METHOD,
    SEARCH_SPACE,
    mean_log_gains,
    run_subspace_search,
)
from tests.test_doom_real_training import (
    SyntheticFitness,
    ToyVAE,
    ToyRNN,
    fixture_settings,
)
from tests.test_doom_reference_warm_start import WarmFixture


def settings_for(root, output, initial):
    settings = fixture_settings(root, output)
    settings["arguments"].update(sigma=0.25, seed=96)
    settings.update(
        initializer_parameters_sha256=parameter_fingerprint(initial),
        training_method=METHOD,
        search_space=SEARCH_SPACE,
    )
    return settings


class SubspaceFixture(WarmFixture):
    """Temporary complete raw reports; no simulator or GPU is invoked."""

    def __init__(self, root):
        super().__init__(root)
        self.arguments["generations"] = 4
        initial = np.full(1088, -0.001)
        self.protocol.update(
            initializer_controller=str(self.prior),
            initializer_parameters_sha256=identity(initial),
            search_space=SEARCH_SPACE,
            maximum_training_fitness_games=512,
            maximum_training_holdout_games=128,
        )
        self.protocol["frozen_inputs"][str(self.prior)] = digest(self.prior)
        self.protocol["frozen_source"]["src/doom_controller_gains.py"] = digest(
            "src/doom_controller_gains.py"
        )
        for path, gain, generation in ((self.best, 2.0, 2), (self.last, 3.0, 4)):
            with np.load(path, allow_pickle=False) as archive:
                values = {key: archive[key].copy() for key in archive.files}
            values.update(
                params=controller_from_log_gains(initial, np.full(3, np.log(gain))),
                log_gains=np.full(3, np.log(gain)),
                feature_block_sizes=np.array([64, 512, 512]),
                initializer_parameters_sha256=identity(initial),
                generation=generation,
                training_method=METHOD,
            )
            with path.open("wb") as stream:
                np.savez(stream, **values)
        metadata_path = Path(str(self.best) + ".json")
        metadata = json.loads(metadata_path.read_text())
        metadata.update(
            arguments=self.arguments,
            input_sha256=self.protocol["frozen_inputs"],
            source_sha256=self.protocol["frozen_source"],
            training_method=METHOD,
            search_space=SEARCH_SPACE,
            initializer_parameters_sha256=identity(initial),
        )
        metadata_path.write_text(json.dumps(metadata))
        pointer = Path(str(self.best) + ".resume.json")
        saved = json.loads(pointer.read_text())
        saved.update(
            generation=4,
            best_generation=2,
            best_sha256=digest(self.best),
            last_sha256=digest(self.last),
        )
        pointer.write_text(json.dumps(saved))
        self.protocol_path.write_text(json.dumps(self.protocol))
        training = json.loads(self.training_audit.read_text())
        training.update(
            protocol_sha256=digest(self.protocol_path),
            raw_training_games=640,
            projected1088_weights_and_three_dimensional_optimizer_reconstructed=True,
            training_method=METHOD,
            search_space=SEARCH_SPACE,
            initializer_parameters_sha256=identity(initial),
            files={str(path): digest(path) for path in (self.best, self.last)},
        )
        self.training_audit.write_text(json.dumps(training))

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
                "src/doom_subspace_comparison.py": digest(
                    "src/doom_subspace_comparison.py"
                )
            },
        )


class TestSubspaceTraining(unittest.TestCase):
    def test_interruption_recovers_projected_weights_optimizer_and_next_rng(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            initial = np.full(1088, 0.003)
            outputs = [root / "full.npz", root / "recovered.npz"]
            run_subspace_search(
                settings_for(root, outputs[0], initial), SyntheticFitness(), initial
            )
            settings = settings_for(root, outputs[1], initial)
            with self.assertRaises(KeyboardInterrupt):
                run_subspace_search(
                    settings, SyntheticFitness(interrupt_fitness=True), initial
                )
            partial = Path(str(outputs[1]) + ".search/games/g001_fitness.partial.json")
            self.assertEqual(len(json.loads(partial.read_text())["episodes_detail"]), 3)
            run_subspace_search(settings, SyntheticFitness(), initial, resume=True)
            next_populations = []
            for output in outputs:
                audit = audit_subspace_capsule(output, initial)
                self.assertEqual(audit["raw_training_games"], 25)
                self.assertTrue(
                    audit[
                        "projected1088_weights_and_three_dimensional_optimizer_reconstructed"
                    ]
                )
                pointer = json.loads(Path(str(output) + ".resume.json").read_text())
                state = pickle.loads(Path(pointer["state_path"]).read_bytes())
                self.assertEqual(state["optimizer"].mean.shape, (3,))
                np.random.set_state(state["numpy_random_state"])
                next_populations.append(state["optimizer"].ask())
            np.testing.assert_array_equal(*next_populations)
            for suffix in (".npz", ".last.npz"):
                with (
                    np.load(root / ("full" + suffix)) as first,
                    np.load(root / ("recovered" + suffix)) as second,
                ):
                    for key in ("params", "log_gains", "generation", "score"):
                        np.testing.assert_array_equal(first[key], second[key])
            with self.assertRaisesRegex(ValueError, "Completed"):
                run_subspace_search(settings, SyntheticFitness(), initial, resume=True)
            with self.assertRaises(FileExistsError):
                run_subspace_search(settings, SyntheticFitness(), initial)

    def test_wrong_initializer_and_forged_projected_checkpoint_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "policy.npz"
            initial = np.full(1088, 0.003)
            run_subspace_search(
                settings_for(root, output, initial), SyntheticFitness(), initial
            )
            with self.assertRaisesRegex(ValueError, "initializer"):
                audit_subspace_capsule(output, initial * 1.01)
            with np.load(output) as archive:
                values = {key: archive[key].copy() for key in archive.files}
            values["params"][0] += 0.01
            with output.open("wb") as stream:
                np.savez(stream, **values)
            pointer = Path(str(output) + ".resume.json")
            saved = json.loads(pointer.read_text())
            saved["best_sha256"] = digest(output)
            pointer.write_text(json.dumps(saved))
            with self.assertRaises(AssertionError):
                audit_subspace_capsule(output, initial)

    def test_corrupted_optimizer_covariance_and_partial_input_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "policy.npz"
            initial = np.full(1088, 0.003)
            settings = settings_for(root, output, initial)
            with self.assertRaises(KeyboardInterrupt):
                run_subspace_search(
                    settings, SyntheticFitness(interrupt_fitness=True), initial
                )
            partial = Path(str(output) + ".search/games/g001_fitness.partial.json")
            saved = json.loads(partial.read_text())
            saved["parameters_sha256"][0] = "changed"
            partial.write_text(json.dumps(saved))
            with self.assertRaisesRegex(ValueError, "different inputs"):
                run_subspace_search(settings, SyntheticFitness(), initial, resume=True)
            other = root / "complete.npz"
            run_subspace_search(
                settings_for(root, other, initial), SyntheticFitness(), initial
            )
            pointer = Path(str(other) + ".resume.json")
            saved = json.loads(pointer.read_text())
            state_path = Path(saved["state_path"])
            state = pickle.loads(state_path.read_bytes())
            state["optimizer"].sm.C[0, 0] *= 1.01
            state_path.write_bytes(pickle.dumps(state))
            saved["state_sha256"] = digest(state_path)
            pointer.write_text(json.dumps(saved))
            with self.assertRaises(AssertionError):
                audit_subspace_capsule(other, initial)

    def test_bounded_mean_and_projected_actor_keep_original_calculation(self):
        opt = cma.CMAEvolutionStrategy(
            np.zeros(3),
            0.25,
            {
                "popsize": 8,
                "seed": 96,
                "verbose": -9,
                "bounds": [float(np.log(0.25)), float(np.log(4.0))],
            },
        )
        opt.mean = np.array([3.0, -4.0, 0.0])
        points = mean_log_gains(opt)
        initial = np.linspace(-0.01, 0.01, 1088)
        controller_from_log_gains(initial, points)
        self.assertFalse(np.array_equal(points, opt.mean))
        vae, rnn = ToyVAE(), ToyRNN()
        dynamic = make_population_policy(vae, rnn)
        rng = np.random.default_rng(96)
        images = rng.integers(0, 256, (4, 64, 64, 3), dtype=np.uint8)
        hidden = tuple(
            jnp.asarray(rng.normal(size=(4, 512)), dtype=jnp.float32) for _ in range(2)
        )
        keys = jnp.stack([jax.random.PRNGKey(n) for n in range(4)])
        steps = jnp.arange(4, dtype=jnp.int32)
        controllers = jnp.asarray(
            np.stack(
                [
                    controller_from_log_gains(initial, np.full(3, np.log(g)))
                    for g in (0.25, 0.5, 1, 4)
                ]
            ),
            dtype=jnp.float64,
        )
        actual = dynamic(images, hidden, keys, steps, controllers)
        for slot in range(4):
            expected = make_reference_policy(
                vae, rnn, controllers[slot], posterior=True
            )(
                images[slot : slot + 1],
                tuple(h[slot : slot + 1] for h in hidden),
                keys[slot : slot + 1],
                steps[slot : slot + 1],
            )
            for one, two in zip(
                jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
            ):
                np.testing.assert_array_equal(np.asarray(one)[slot], np.asarray(two)[0])


class TestSubspaceSelection(unittest.TestCase):
    def test_full_validation_and_choice_audit_precede_both_reserved_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = SubspaceFixture(Path(directory))
            result = fixture.run()
            self.assertTrue(result["new_policy_won"])
            self.assertEqual(len(result["test_tasks"]), 2)
            self.assertEqual(fixture.events[:4], ["validation"] * 4)
            self.assertLess(
                fixture.events.index("audit_choice"), fixture.events.index("test")
            )
            report = Path(result["test_tasks"][1]["report"])
            value = json.loads(report.read_text())
            value["episodes_detail"].pop()
            report.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "incomplete"):
                real_result_audit(
                    fixture.protocol_path,
                    fixture.root / f"{fixture.prefix}_real_evaluation_result.json",
                    resamples=300,
                )

    def test_failed_choice_audit_and_inconsistent_gains_prevent_test_dispatch(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = SubspaceFixture(Path(directory))

            def reject(*args):
                raise ValueError("Synthetic independent rejection")

            with self.assertRaisesRegex(ValueError, "rejection"):
                fixture.run(audit_choice=reject)
            self.assertNotIn("test", fixture.events)
            with np.load(fixture.best) as archive:
                values = {k: archive[k].copy() for k in archive.files}
            values["log_gains"][0] += 0.01
            with fixture.best.open("wb") as stream:
                np.savez(stream, **values)
            pointer = Path(str(fixture.best) + ".resume.json")
            saved = json.loads(pointer.read_text())
            saved["best_sha256"] = digest(fixture.best)
            pointer.write_text(json.dumps(saved))
            with self.assertRaisesRegex(ValueError, "Projected"):
                archive_info(fixture.best, fixture.protocol, direct=True)

    def test_initializer_uses_complete_v2_parent_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = WarmFixture(Path(directory))
            fixture.scores["val_prior_initializer"] = 1000
            result = fixture.run()
            self.assertFalse(result["new_policy_won"])
            selected = str(fixture.root / f"{fixture.prefix}_frozen_selection.json")
            audit = str(
                fixture.root / f"{fixture.prefix}_frozen_selection_cpu_audit.json"
            )
            protocol = dict(
                fixture.protocol,
                initializer_selection_frozen=selected,
                initializer_selection_cpu_audit=audit,
                initializer_parent_protocol=str(fixture.protocol_path),
                initializer_controller=str(fixture.prior),
                initializer_parameters_sha256=identity(np.full(1088, -0.001)),
            )
            protocol["frozen_inputs"] = dict(protocol["frozen_inputs"])
            for path in (selected, audit):
                protocol["frozen_inputs"][path] = digest(path)
            np.testing.assert_array_equal(
                registered_initializer(protocol), np.full(1088, -0.001)
            )
            report = Path(result["selected"]["report"])
            value = json.loads(report.read_text())
            value["episodes_detail"].pop()
            report.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "cohort/hash"):
                registered_initializer(protocol)


if __name__ == "__main__":
    unittest.main()
