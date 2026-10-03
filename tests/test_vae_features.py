import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import equinox as eqx
import jax
import numpy as np
import optax

from scripts.tools.encode_vae_split import encode_split
from scripts.tools.probe_doom_vision import main as probe
from src.vae import VAE
from src.vae_data import file_sha256
from src.vae_features import encode_frame_features, load_feature_cache
from src.vae_training import prepare_frame_cache, save_bundle


class TestVAEFeatures(unittest.TestCase):
    def test_sampled_features_match_episode_probe_and_preserve_provenance(self):
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
            vae = root / "vae.eqx"
            manifest = dict(preprocessing="existing_full_frame_rgb64_uint8")
            for index, partition in enumerate(("training", "validation"), 1):
                images = np.full((6, 64, 64, 3), 35 + index * 10, np.uint8)
                for frame in (0, 2, 4):
                    images[frame, 28:30, 10 + frame * 5 : 12 + frame * 5] = [
                        255,
                        220,
                        0,
                    ]
                for frame in (1, 2, 5):
                    images[frame, 24, 19 + frame] = [255, 220, 0]
                raw = root / f"raw{index}.npz"
                np.savez_compressed(
                    raw,
                    obs=images,
                    actions=np.full((6, 1), index - 1, np.float32),
                    rewards=np.ones(6),
                    dones=[0, 0, 0, 0, 0, 1],
                )
                manifest[partition + "_episodes"] = [
                    dict(
                        path=str(raw),
                        sha256=file_sha256(raw),
                        signature=f"{index:064x}",
                        frames=6,
                    )
                ]
            split = root / "split.json"
            split.write_text(json.dumps(manifest))
            frames, features, episodes = (
                root / "frames",
                root / "features",
                root / "episodes",
            )
            prepare_frame_cache(manifest, split, frames, 6, 6, 73)
            encode_split(split, vae, episodes, batch_size=2)
            encode_frame_features(frames, split, vae, features, batch_size=2)
            arrays = load_feature_cache(features, frames, split, vae)
            for array, partition in zip(arrays, ("training", "validation")):
                with np.load(next((episodes / partition).glob("*.npz"))) as data:
                    np.testing.assert_array_equal(array, data["mu"])
            reports = []
            for argument, source in (
                ("--encoded-dir", episodes),
                ("--encoded-frame-cache", features),
            ):
                output = root / f"probe{len(reports)}.json"
                arguments = [
                    "probe",
                    "--split",
                    str(split),
                    "--frame-cache-dir",
                    str(frames),
                    "--vae",
                    str(vae),
                    argument,
                    str(source),
                    "--source-contains",
                    "raw",
                    "--frames-per-episode",
                    "6",
                    "--validation-frames-per-episode",
                    "6",
                    "--bootstrap-resamples",
                    "20",
                    "--output",
                    str(output),
                ]
                with (
                    patch.object(sys, "argv", arguments),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    probe()
                reports.append(json.loads(output.read_text()))
            self.assertEqual(reports[0]["classification"], reports[1]["classification"])
            self.assertEqual(
                reports[0]["heldout_episode_bootstrap"],
                reports[1]["heldout_episode_bootstrap"],
            )
            self.assertEqual(
                reports[0]["largest_component_position_mae_pixels"],
                reports[1]["largest_component_position_mae_pixels"],
            )
            original_hash = file_sha256(features / "training_mu.npy")
            encode_frame_features(frames, split, vae, features, batch_size=2)
            self.assertEqual(original_hash, file_sha256(features / "training_mu.npy"))
            with self.assertRaisesRegex(ValueError, "calculation shape"):
                encode_frame_features(frames, split, vae, features, batch_size=3)
            with (features / "training_mu.npy").open("ab") as stream:
                stream.write(b"changed")
            with self.assertRaisesRegex(ValueError, "content changed"):
                load_feature_cache(features, frames, split, vae)


if __name__ == "__main__":
    unittest.main()
