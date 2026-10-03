import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import glob
import os
from tqdm import tqdm
from src.vae import VAE

import argparse
import hashlib
import zipfile
from pathlib import Path
from src.config import get_config

# Settings
BATCH_SIZE = 128


def process_data():
    parser = argparse.ArgumentParser(
        description="Process collected data for VAE/RNN training"
    )
    parser.add_argument(
        "--env", type=str, default="CarRacing-v3", help="Environment name"
    )
    parser.add_argument("--data_dir", default=None, help="Raw rollout directory")
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Use a fresh directory for a new VAE/data version",
    )
    parser.add_argument("--vae", default=None, help="VAE checkpoint path")
    args = parser.parse_args()

    env_name = args.env
    config = get_config(env_name)

    # Paths
    data_dir_base = "data/rollouts"
    output_dir = args.output_dir or os.path.join("data/series", env_name)
    checkpoint_dir = os.path.join("checkpoints", env_name)
    vae_path = args.vae or os.path.join(checkpoint_dir, "vae.eqx")

    # Data patterns - look in env specific dir
    # We assume data is collected into data/rollouts/{env_name}
    # Recursive search to find data in subdirectories (e.g. main, aggressive, etc.)
    raw_root = args.data_dir or os.path.join(data_dir_base, env_name)
    data_pattern = os.path.join(raw_root, "**", "*.npz")
    vae_digest = hashlib.sha256(Path(vae_path).read_bytes()).hexdigest()
    if glob.glob(os.path.join(output_dir, "series_[0-9]*_orig.npz")):
        raise ValueError(
            "Legacy indexed series may contain duplicates or stale encodings. Use --output_dir with a fresh directory."
        )
    manifest_path = Path(output_dir) / "metadata.json"
    if manifest_path.exists():
        import json

        if json.loads(manifest_path.read_text())["vae_sha256"] != vae_digest:
            raise ValueError(
                "This output directory belongs to another VAE. Use a fresh --output_dir."
            )

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    print(f"Loading VAE from {vae_path}...")
    # Initialize VAE
    model = VAE(latent_dim=config.latent_dim, key=jax.random.PRNGKey(0))
    try:
        model = eqx.tree_deserialise_leaves(vae_path, model)
    except Exception as e:
        print(f"Error loading VAE: {e}")
        print(
            "Ensure you have trained the VAE first: python run_vae_training.py --env "
            + env_name
        )
        return

    # Optimized Encoder function
    @eqx.filter_jit
    def encode_batch(m, x):
        def encode_single(img):
            features = m.encoder(img)
            features = jnp.reshape(features, (-1,))
            mu = m.mu_head(features)
            logvar = m.logvar_head(features)
            return mu, logvar

        return jax.vmap(encode_single)(x)

    # Get files
    files = sorted(glob.glob(data_pattern, recursive=True))

    # Shuffle to mix them up during processing (optional but good practice)
    import json

    manifest_path.write_text(
        json.dumps(
            {"vae_sha256": vae_digest, "raw_root": raw_root, "episodes": len(files)},
            indent=2,
        )
        + "\n"
    )

    print(f"Processing {len(files)} episodes for {env_name}...")

    for i, f in enumerate(tqdm(files)):
        # Stable source IDs make reruns idempotent instead of leaving shuffled duplicates.
        source = os.path.relpath(f, raw_root)
        source_id = hashlib.sha256(source.encode()).hexdigest()[:16]
        source_digest = hashlib.sha256(Path(f).read_bytes()).hexdigest()
        suffixes = ("_orig",) if config.is_doom else ("_orig", "_flip")
        current = True
        for suffix in suffixes:
            target = Path(output_dir) / f"episode_{source_id}{suffix}.npz"
            try:
                with np.load(target) as encoded:
                    if (
                        str(encoded["source_sha256"]) != source_digest
                        or str(encoded["vae_sha256"]) != vae_digest
                    ):
                        current = False
                        break
            except (OSError, KeyError, ValueError, zipfile.BadZipFile):
                current = False
                break
        if current:
            continue
        try:
            with np.load(f) as data:
                obs = data["obs"]
                actions = data["actions"]
                rewards = data["rewards"]
                dones = data["dones"]
            # A collector may have finished writing while we opened the archive.
            source_digest = hashlib.sha256(Path(f).read_bytes()).hexdigest()
        except Exception as e:
            print(f"Skipping corrupt file {f}: {e}")
            continue

        # --- Helper to encode and save a sequence ---
        def save_sequence(obs_in, actions_in, suffix):
            n_steps = obs_in.shape[0]
            mu_seq = []
            logvar_seq = []

            # Batch processing to save GPU memory
            for j in range(0, n_steps, BATCH_SIZE):
                batch_obs = obs_in[j : j + BATCH_SIZE]
                current_batch_size = batch_obs.shape[0]

                # Prepare for JAX (Normalize + Transpose)
                batch_jax = jnp.array(batch_obs, dtype=jnp.float32) / 255.0
                batch_jax = jnp.transpose(batch_jax, (0, 3, 1, 2))

                # Pad if necessary to match BATCH_SIZE (avoids JIT recompilation)
                if current_batch_size < BATCH_SIZE:
                    pad_amount = BATCH_SIZE - current_batch_size
                    # Pad with zeros: ((0, pad), (0,0), (0,0), (0,0))
                    batch_jax = jnp.pad(
                        batch_jax, ((0, pad_amount), (0, 0), (0, 0), (0, 0))
                    )

                mu, logvar = encode_batch(model, batch_jax)

                # Slice back to original size if padded
                if current_batch_size < BATCH_SIZE:
                    mu = mu[:current_batch_size]
                    logvar = logvar[:current_batch_size]

                mu_seq.append(np.array(mu))
                logvar_seq.append(np.array(logvar))

            mu_data = np.concatenate(mu_seq, axis=0)
            logvar_data = np.concatenate(logvar_seq, axis=0)

            np.savez_compressed(
                os.path.join(output_dir, f"episode_{source_id}{suffix}.npz"),
                mu=mu_data,
                logvar=logvar_data,
                actions=actions_in,
                rewards=rewards,
                dones=dones,
                source=source,
                vae_sha256=vae_digest,
                source_sha256=source_digest,
            )

        # --- 1. Save Original ---
        save_sequence(obs, actions, "_orig")

        # --- 2. Save Mirrored (The "Anti-Spin" Fix) ---
        # Only for CarRacing where steering is symmetric and index 0
        if not config.is_doom:
            # Flip image horizontally (Axis 2 is width for format N,H,W,C)
            obs_flipped = np.flip(obs, axis=2)

            # Negate steering (Action index 0)
            actions_flipped = actions.copy()
            actions_flipped[:, 0] *= -1.0

            save_sequence(obs_flipped, actions_flipped, "_flip")

    print(f"Data processing complete for {env_name}.")


if __name__ == "__main__":
    process_data()
