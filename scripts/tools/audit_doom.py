"""Measure death detection and latent uncertainty on saved episodes."""

import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("JAX_PLATFORMS", "cuda")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import jax
import jax.numpy as jnp
import numpy as np

from src.config import get_config
from src.rnn import load_rnn
from src.vae_data import file_sha256, transition_signature


def audit_latents(data, rng, sample_posterior, legacy_float16=False):
    """Match train_rnn.load_batch, retaining the old rounding only on request."""
    z = data["mu"].copy() if legacy_float16 else data["mu"].astype(np.float32)
    if sample_posterior:
        noise = rng.standard_normal(z.shape)
        if not legacy_float16:
            noise = noise.astype(np.float32)
        z += np.exp(0.5 * data["logvar"]) * noise
    if not np.isfinite(z).all():
        raise FloatingPointError("Nonfinite audit latents")
    return z


def death_counts(scores, dones):
    scores, dones = np.asarray(scores), np.asarray(dones, bool)
    if scores.shape != dones.shape or not np.isfinite(scores).all():
        raise ValueError("Death scores/labels are invalid or unaligned")
    predicted = scores >= 0.5
    return dict(
        fatal_transitions=int(dones.sum()),
        detected_deaths=int(np.count_nonzero(predicted & dones)),
        live_transitions=int((~dones).sum()),
        false_deaths=int(np.count_nonzero(predicted & ~dones)),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/series/VizdoomTakeCover-v0")
    parser.add_argument("--rnn", default="checkpoints/VizdoomTakeCover-v0/rnn.eqx")
    parser.add_argument("--episodes", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="artifacts/doom_audit.json")
    parser.add_argument(
        "--legacy-float16-posterior",
        action="store_true",
        help="Reproduce historical audit rounding; default matches float32 training inputs",
    )
    parser.add_argument(
        "--mean_latents",
        action="store_true",
        help="Audit the real-inference mean-latent ablation",
    )
    parser.add_argument(
        "--heldout",
        action="store_true",
        help="Use the checkpoint's recorded validation split",
    )
    parser.add_argument(
        "--validation_manifest",
        default=None,
        help="Compare models on the same recorded validation-file list",
    )
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("Episode count must be positive")
    if Path(args.output).exists():
        raise FileExistsError("Preserve existing audits; use a fresh output path")
    rng = np.random.default_rng(args.seed)
    files = sorted(Path(args.data_dir).glob("*.npz"))
    metadata_path = Path(args.rnn + ".json")
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    if args.heldout or args.validation_manifest:
        split_metadata = (
            json.loads(Path(args.validation_manifest).read_text())
            if args.validation_manifest
            else metadata
        )
        files = [Path(f) for f in split_metadata["validation_files"]]
    if not files:
        raise ValueError("No audit episodes found")
    chosen = rng.choice(len(files), min(args.episodes, len(files)), replace=False)
    cfg = get_config("VizdoomTakeCover-v0")
    model = load_rnn(args.rnn, cfg)

    @jax.jit
    def predictions(inputs):
        def step(hidden, x):
            (pi, mu, logsigma, reward, done), hidden = jax.vmap(model)(x, hidden)
            expected = jnp.sum(jnp.exp(pi) * mu, axis=1)
            std = jnp.sqrt(
                jnp.sum(
                    jnp.exp(pi)
                    * (jnp.exp(2 * logsigma) + (mu - expected[:, None, :]) ** 2),
                    axis=1,
                )
            )
            return hidden, (jax.nn.sigmoid(done[:, 0]), reward[:, 0], expected, std)

        zeros = jnp.zeros((inputs.shape[1], cfg.hidden_size))
        return jax.lax.scan(step, (zeros, zeros), inputs)[1]

    lengths, terminal_probs, early_probs, live_probs = [], [], [], []
    posterior_std, predictive_std, residuals, rewards, mean_latents = [], [], [], [], []
    internal_dones = missing_dones = invalid = censored = 0
    episode_records = []
    for offset in range(0, len(chosen), 16):
        batch = []
        for index in chosen[offset : offset + 16]:
            with np.load(files[index]) as data:
                batch.append({k: data[k] for k in data.files})
        max_len = max(len(d["mu"]) for d in batch)
        inputs = np.zeros((max_len, 16, cfg.latent_dim + cfg.action_dim), np.float32)
        for i, d in enumerate(batch):
            n = len(d["mu"])
            if any(len(d[k]) != n for k in ("actions", "rewards", "dones", "logvar")):
                raise ValueError("Audit episode has inconsistent transition lengths")
            z = audit_latents(
                d,
                rng,
                metadata.get("posterior_sampling", False) and not args.mean_latents,
                args.legacy_float16_posterior,
            )
            a = d["actions"]
            if metadata.get("canonical_actions", False):
                a = np.where(a < -0.3, -1.0, np.where(a > 0.3, 1.0, 0.0))
            inputs[:n, i] = np.concatenate([z, a], axis=-1)
        probs, r_pred, expected, std = map(np.asarray, predictions(jnp.asarray(inputs)))
        for i, d in enumerate(batch):
            n = len(d["mu"])
            lengths.append(n)
            is_done = np.asarray(d["dones"], bool)
            episode_records.append(
                dict(
                    path=str(files[chosen[offset + i]]),
                    transition_signature=transition_signature(d),
                    frames=n,
                    counts=death_counts(probs[:n, i], is_done),
                )
            )
            internal_dones += int(is_done[:-1].sum())
            censored += int(not is_done[-1] and n == 2100)
            missing_dones += int(not is_done[-1] and n != 2100)
            terminal_probs.extend(probs[:n, i][is_done].tolist())
            # Old training shifts the terminal target one frame early.
            if n > 1 and is_done[-1]:
                early_probs.append(float(probs[n - 2, i]))
            live_probs.extend(probs[:n, i][~is_done].tolist())
            rewards.extend(r_pred[:n, i].tolist())
            posterior_std.append(np.exp(0.5 * d["logvar"]).mean(axis=0))
            predictive_std.append(std[:n, i].mean(axis=0))
            mean_latents.append(d["mu"].std(axis=0))
            if n > 1:
                residuals.append(
                    float(np.mean((expected[: n - 1, i] - d["mu"][1:]) ** 2))
                )

    def stats(values):
        arr = np.asarray(values)
        return (
            {
                "mean": float(arr.mean()),
                "p10": float(np.quantile(arr, 0.1)),
                "p50": float(np.median(arr)),
                "p90": float(np.quantile(arr, 0.9)),
            }
            if arr.size
            else {}
        )

    report = {
        "rnn": args.rnn,
        "rnn_sha256": file_sha256(args.rnn),
        "device": str(jax.devices()[0]),
        "inference_batch_size": 16,
        "posterior_input_protocol": "historical_storage_dtype_rounding"
        if args.legacy_float16_posterior
        else "train_rnn_float32_inputs_and_noise",
        "data_dir": args.data_dir,
        "available_episodes": len(files),
        "sampled_episodes": len(lengths),
        "seed": args.seed,
        "heldout": args.heldout,
        "mean_latents_override": args.mean_latents,
        "inference_posterior_sampling": bool(
            metadata.get("posterior_sampling", False) and not args.mean_latents
        ),
        "validation_manifest": args.validation_manifest,
        "done_positive_weight": metadata.get("done_positive_weight", 1.0),
        "death_score_semantics": "Sigmoid of weighted-BCE logit; this is a detection score, not a calibrated probability",
        "lengths": stats(lengths),
        "internal_terminal_labels": internal_dones,
        "missing_final_terminal_labels": missing_dones,
        "time_limited_episodes": censored,
        "invalid_length_episodes": invalid,
        "death_probability_at_actual_terminal_transition": stats(terminal_probs),
        "death_probability_one_step_before_terminal": stats(early_probs),
        "death_recall_at_0.5": float(np.mean(np.asarray(terminal_probs) >= 0.5))
        if terminal_probs
        else None,
        "live_probability": stats(live_probs),
        "false_death_rate_at_0.5": float(np.mean(np.asarray(live_probs) >= 0.5))
        if live_probs
        else None,
        "predicted_reward": stats(rewards),
        "posterior_std_per_dimension": np.mean(posterior_std, axis=0).tolist(),
        "rnn_std_per_dimension": np.mean(predictive_std, axis=0).tolist(),
        "latent_mean_std_per_dimension": np.mean(mean_latents, axis=0).tolist(),
        "teacher_forced_latent_mse": stats(residuals),
        "episode_records": episode_records,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if not k.endswith("per_dimension") and k != "episode_records"
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
