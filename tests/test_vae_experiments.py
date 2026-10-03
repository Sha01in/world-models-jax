import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from scripts.tools.encode_vae_split import encode_split
from scripts.tools.probe_doom_vision import auc

from src.vae import VAE, load_vae
from src.vae_data import file_sha256, frame_indices, load_split, observation_header
from src.vae_training import (
    BestCheckpoint,
    batches,
    load_frame_cache,
    loss_components,
    prepare_frame_cache,
    save_bundle,
)


class TestVAEExperiment(unittest.TestCase):
    def test_reference_geometry_and_parameter_count(self):
        model = VAE(64, jax.random.PRNGKey(0), "paper")
        image = jnp.zeros((3, 64, 64))
        encoder_sizes = []
        for layer in model.encoder.layers:
            image = layer(image)
            encoder_sizes.append(image.shape[-1])
        self.assertEqual(encoder_sizes, [31, 14, 6, 2])
        image = model.decoder.linear(jnp.zeros(64)).reshape(model.decoder.initial_shape)
        decoder_sizes = []
        for layer in model.decoder.layers:
            image = layer(image)
            decoder_sizes.append(image.shape[-1])
        self.assertEqual(decoder_sizes, [5, 13, 30, 64])
        self.assertEqual(image.shape, (3, 64, 64))
        self.assertEqual(
            sum(a.size for a in jax.tree_util.tree_leaves(model) if eqx.is_array(a)),
            4446915,
        )

    def test_legacy_checkpoint_roundtrip_preserves_bytes(self):
        path = Path(
            "checkpoints/VizdoomTakeCover-v0/reproduction_refined_selected/vae.eqx"
        )
        if not path.exists():
            self.skipTest("Frozen incumbent exists only in the experiment workspace")
        model = load_vae(path, 64)
        self.assertEqual(model.architecture, "current")
        with tempfile.TemporaryDirectory() as temporary:
            saved = Path(temporary) / "legacy.eqx"
            eqx.tree_serialise_leaves(saved, model)
            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(),
                hashlib.sha256(saved.read_bytes()).hexdigest(),
            )
        self.assertEqual(model.decoder(jnp.zeros(64)).shape, (3, 64, 64))

    def test_reference_bundle_metadata_and_corruption_detection(self):
        model = VAE(64, jax.random.PRNGKey(1), "paper")
        optimizer = optax.adam(0.0001)
        opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
        with tempfile.TemporaryDirectory() as temporary:
            save_bundle(
                temporary, "vae", model, opt_state, jax.random.PRNGKey(4), {"epoch": 0}
            )
            path = Path(temporary) / "vae.eqx"
            loaded = load_vae(path, 64)
            self.assertEqual(loaded.architecture, "paper")
            np.testing.assert_array_equal(loaded.mu_head.weight, model.mu_head.weight)
            with self.assertRaisesRegex(ValueError, "latent width"):
                load_vae(path, 32)
            with path.open("ab") as stream:
                stream.write(b"changed")
            with self.assertRaisesRegex(ValueError, "sidecar"):
                load_vae(path, 64)

    def test_loss_matches_reference_sum_and_kl_floor_and_masks_padding(self):
        class ZeroModel(eqx.Module):
            def __call__(self, image, key):
                return jnp.zeros_like(image), jnp.zeros(64), jnp.zeros(64)

        batch = jnp.stack([jnp.ones((3, 64, 64)), jnp.ones((3, 64, 64)) * 10])
        loss, (reconstruction, raw_kl) = loss_components(
            ZeroModel(), batch, jax.random.PRNGKey(0), jnp.array([1.0, 0.0])
        )
        self.assertEqual(float(reconstruction), 12288)
        self.assertEqual(float(raw_kl), 0)
        self.assertEqual(float(loss), 12320)
        pieces = list(batches(np.zeros((3, 64, 64, 3), np.uint8), 2))
        self.assertEqual([p[2] for p in pieces], [2, 1])
        np.testing.assert_array_equal(pieces[-1][1], [1, 0])

    def test_holdout_aliases_rejected_and_frame_samples_are_order_independent(self):
        episode = {
            "path": "a.npz",
            "sha256": "content",
            "signature": "01" * 32,
            "frames": 200,
        }
        other = dict(episode, path="alias.npz", signature="02" * 32)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "split.json"
            path.write_text(
                json.dumps(
                    {"training_episodes": [episode], "validation_episodes": [other]}
                )
            )
            with self.assertRaisesRegex(ValueError, "overlap"):
                load_split(path)
            raw = Path(temporary) / "raw.npz"
            np.savez_compressed(raw, obs=np.zeros((3, 64, 64, 3), np.uint8))
            self.assertEqual(observation_header(raw), (3, 64, 64, 3))
        first = frame_indices(episode, 64, 73)
        np.testing.assert_array_equal(first, frame_indices(dict(episode), 64, 73))
        self.assertEqual(len(np.unique(first)), 64)

    def test_epoch_zero_eligible_and_consecutive_patience(self):
        selection = BestCheckpoint(3)
        self.assertTrue(selection.consider(10, 0))
        self.assertFalse(selection.consider(11, 1))
        self.assertTrue(selection.consider(9, 2))
        for epoch in range(3, 6):
            self.assertFalse(selection.consider(9, epoch))
        self.assertTrue(selection.should_stop)
        self.assertEqual(selection.best_epoch, 2)
        self.assertEqual(selection.best_loss, 9)
        with self.assertRaises(FloatingPointError):
            selection.consider(float("nan"), 6)

    def test_shared_cache_rejects_changed_sampling_and_changed_content(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            groups = []
            for index in range(2):
                path = root / f"raw{index}.npz"
                frames = np.full((4, 64, 64, 3), index * 127, np.uint8)
                np.savez_compressed(path, obs=frames)
                groups.append(
                    [
                        dict(
                            path=str(path),
                            sha256=file_sha256(path),
                            frames=4,
                            signature=f"{index + 1:064x}",
                        )
                    ]
                )
            manifest = dict(training_episodes=groups[0], validation_episodes=groups[1])
            split = root / "split.json"
            split.write_text(json.dumps(manifest))
            directory = root / "frames"
            prepare_frame_cache(manifest, split, directory, 3, 2, 73)
            train, val = load_frame_cache(directory, split, 3, 2, 73)
            self.assertEqual(train.shape, (3, 64, 64, 3))
            self.assertEqual(val.shape, (2, 64, 64, 3))
            self.assertTrue(np.all(train == 0))
            self.assertTrue(np.all(val == 127))
            with self.assertRaisesRegex(ValueError, "settings"):
                load_frame_cache(directory, split, 3, 2, 74)
            with (directory / "training.npy").open("ab") as stream:
                stream.write(b"changed")
            with self.assertRaisesRegex(ValueError, "content changed"):
                load_frame_cache(directory, split, 3, 2, 73)

    def test_probe_auc_handles_ties_and_single_class(self):
        truth = [0, 1, 0, 1]
        self.assertEqual(auc(truth, [0, 1, 0, 1]), 1)
        self.assertEqual(auc(truth, [1, 0, 1, 0]), 0)
        self.assertEqual(auc(truth, [0, 0, 0, 0]), 0.5)
        self.assertIsNone(auc([1, 1], [0, 1]))

    def test_encoding_retains_fatal_action_and_separates_holdouts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = VAE(64, jax.random.PRNGKey(0), "paper")
            optimizer = optax.adam(0.0001)
            save_bundle(
                root,
                "vae",
                model,
                optimizer.init(eqx.filter(model, eqx.is_array)),
                jax.random.PRNGKey(0),
                {},
            )
            actions = np.array([[-1], [0], [1]], np.float32)
            dones = np.array([0, 0, 1], np.float32)
            manifest = dict(preprocessing="existing_full_frame_rgb64_uint8")
            for index, name in enumerate(("training", "validation"), 1):
                path = root / f"raw{index}.npz"
                np.savez_compressed(
                    path,
                    obs=np.full((3, 64, 64, 3), index * 63, np.uint8),
                    actions=actions,
                    rewards=np.ones(3),
                    dones=dones,
                )
                manifest[name + "_episodes"] = [
                    dict(
                        path=str(path),
                        sha256=file_sha256(path),
                        frames=3,
                        signature=f"{index:064x}",
                    )
                ]
            split = root / "split.json"
            split.write_text(json.dumps(manifest))
            destination = root / "series"
            metadata = encode_split(split, root / "vae.eqx", destination, batch_size=2)
            self.assertTrue(metadata["complete"])
            for partition in ("training", "validation"):
                files = list((destination / partition).glob("*.npz"))
                self.assertEqual(len(files), 1)
                with np.load(files[0]) as data:
                    self.assertEqual(data["mu"].shape, (3, 64))
                    np.testing.assert_array_equal(data["actions"], actions)
                    np.testing.assert_array_equal(data["dones"], dones)
                    self.assertEqual(
                        str(data["vae_sha256"]), file_sha256(root / "vae.eqx")
                    )
            first_hash = file_sha256(next((destination / "training").glob("*.npz")))
            encode_split(split, root / "vae.eqx", destination, batch_size=2)
            self.assertEqual(
                first_hash, file_sha256(next((destination / "training").glob("*.npz")))
            )
            # Matching provenance must not allow stray old files into RNN input.
            (destination / "training" / "stray.npz").write_bytes(b"stale")
            with self.assertRaisesRegex(ValueError, "contaminate"):
                encode_split(split, root / "vae.eqx", destination, batch_size=2)


if __name__ == "__main__":
    unittest.main()
