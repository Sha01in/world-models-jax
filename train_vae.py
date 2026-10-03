"""Train an isolated VAE experiment on an explicit, immutable episode split."""

import argparse
import json
import os
from pathlib import Path
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import equinox as eqx
import jax
import numpy as np
import optax

from src.config import get_config
from src.vae import ARCHITECTURES, VAE
from src.vae_data import file_sha256, load_split
from src.vae_training import (
    BestCheckpoint,
    batches,
    evaluate,
    load_frame_cache,
    load_training_bundle,
    prepare_frame_cache,
    restored_selection,
    save_bundle,
    train_step,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default="VizdoomTakeCover-v0")
    parser.add_argument("--split", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--architecture", choices=ARCHITECTURES, default="current")
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Maximum total epochs, including a resumed source",
    )
    parser.add_argument(
        "--resume-dir",
        default=None,
        help="Continue a completed budget-limited run into a fresh output directory",
    )
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--kl-tolerance", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--frames-per-episode", type=int, default=64)
    parser.add_argument("--validation-frames-per-episode", type=int, default=16)
    parser.add_argument("--frame-cache-dir", default=None)
    parser.add_argument("--allow-cpu-smoke", action="store_true")
    args = parser.parse_args()
    if (
        min(args.epochs, args.batch_size) < 1
        or min(
            args.patience, args.frames_per_episode, args.validation_frames_per_episode
        )
        < 0
        or args.learning_rate <= 0
        or args.kl_tolerance < 0
    ):
        parser.error("Invalid training budget or loss settings")
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            "Use a fresh experiment directory; existing models are frozen"
        )
    manifest = load_split(args.split)
    devices = jax.devices()
    if (
        not all(device.platform == "gpu" for device in devices)
        and not args.allow_cpu_smoke
    ):
        raise RuntimeError(
            "CUDA is required; CPU is allowed only for explicit smoke tests"
        )
    output.mkdir(parents=True, exist_ok=True)
    settings = dict(
        vars(args),
        split_sha256=file_sha256(args.split),
        jax_version=jax.__version__,
        device=str(devices[0]),
        preprocessing=manifest["preprocessing"],
        frame_sampling="fixed uniform samples per episode, no replacement",
    )
    (output / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    if args.frame_cache_dir:
        train, val = load_frame_cache(
            args.frame_cache_dir,
            args.split,
            args.frames_per_episode,
            args.validation_frames_per_episode,
            args.seed,
        )
    else:
        train, val = prepare_frame_cache(
            manifest,
            args.split,
            output / "frames",
            args.frames_per_episode,
            args.validation_frames_per_episode,
            args.seed,
        )
    config = get_config(args.env)
    key = jax.random.PRNGKey(args.seed)
    key, init_key = jax.random.split(key)
    optimizer = optax.adam(args.learning_rate)
    selection = BestCheckpoint(args.patience)
    history, steps, start_time = [], 0, time.monotonic()
    first_epoch, elapsed_offset = 0, 0
    if args.resume_dir:
        source = Path(args.resume_dir)
        prior_result = json.loads((source / "result.json").read_text())
        if not prior_result["training_finished"] or prior_result["stopped_early"]:
            raise ValueError("Only a completed budget-limited VAE run may be continued")
        expected = {
            name: settings[name]
            for name in (
                "env",
                "architecture",
                "split_sha256",
                "seed",
                "batch_size",
                "learning_rate",
                "kl_tolerance",
                "patience",
                "frames_per_episode",
                "validation_frames_per_episode",
                "preprocessing",
                "allow_cpu_smoke",
            )
        }
        model, opt_state, key, last_metadata = load_training_bundle(
            source,
            "vae_last",
            optimizer,
            expected,
        )
        best_model, best_state, best_key, best_metadata = load_training_bundle(
            source,
            "vae",
            optimizer,
            expected,
        )
        history = json.loads((source / "history.json").read_text())
        selection = restored_selection(history, args.patience, best_metadata)
        last_epoch = last_metadata["epoch"]
        steps = last_metadata["optimizer_steps"]
        if (
            history[-1]["epoch"] != last_epoch
            or history[-1]["optimizer_steps"] != steps
            or history[-1]["validation"] != last_metadata["validation"]
            or prior_result["final_epoch"] != last_epoch
            or prior_result["optimizer_steps"] != steps
            or prior_result["training_frames"] != len(train)
            or prior_result["validation_frames"] != len(val)
            or selection.should_stop
            or args.epochs <= last_epoch
        ):
            raise ValueError(
                "Resume history/result disagree or total budget does not advance"
            )
        validation = evaluate(
            model,
            val,
            args.batch_size,
            args.seed + 100000,
            args.kl_tolerance,
        )
        if not np.isclose(
            validation["loss"], last_metadata["validation"]["loss"], rtol=1e-6
        ):
            raise ValueError("Resumed validation does not match the saved source")
        settings["resume_source"] = dict(
            directory=str(source.resolve()),
            epoch=last_epoch,
            last_sha256=last_metadata["sha256"],
            best_sha256=best_metadata["sha256"],
            history_sha256=file_sha256(source / "history.json"),
        )
        (output / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
        record_keys = (
            "epoch",
            "optimizer_steps",
            "training",
            "validation",
            "elapsed_seconds",
        )
        for name, bundle_model, bundle_state, bundle_key, metadata in (
            ("vae", best_model, best_state, best_key, best_metadata),
            ("vae_last", model, opt_state, key, last_metadata),
        ):
            record = {name: metadata[name] for name in record_keys}
            save_bundle(
                output,
                name,
                bundle_model,
                bundle_state,
                bundle_key,
                dict(settings, **record),
            )
        first_epoch = last_epoch + 1
        elapsed_offset = last_metadata["elapsed_seconds"]
        print(
            json.dumps(
                dict(
                    resumed_epoch=last_epoch,
                    optimizer_steps=steps,
                    verified_validation=validation,
                )
            ),
            flush=True,
        )
    else:
        model = VAE(config.latent_dim, init_key, args.architecture)
        opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    for epoch in range(first_epoch, args.epochs + 1):
        training_metrics = None
        if epoch:
            rng = np.random.default_rng(np.random.SeedSequence([args.seed, epoch]))
            totals, examples = np.zeros(3), 0
            for batch, weights, n in batches(
                train, args.batch_size, rng.permutation(len(train))
            ):
                key, step_key = jax.random.split(key)
                model, opt_state, loss, (reconstruction, kl) = train_step(
                    model,
                    opt_state,
                    batch,
                    weights,
                    step_key,
                    optimizer,
                    args.kl_tolerance,
                )
                values = np.asarray([loss, reconstruction, kl], np.float64)
                if not np.isfinite(values).all():
                    raise FloatingPointError("Nonfinite VAE training loss")
                totals += values * n
                examples += n
                steps += 1
            training_metrics = dict(
                zip(("loss", "reconstruction", "raw_kl"), (totals / examples).tolist())
            )
        validation = evaluate(
            model, val, args.batch_size, args.seed + 100000, args.kl_tolerance
        )
        record = dict(
            epoch=epoch,
            optimizer_steps=steps,
            training=training_metrics,
            validation=validation,
            elapsed_seconds=elapsed_offset + time.monotonic() - start_time,
        )
        metadata = dict(settings, **record)
        if selection.consider(validation["loss"], epoch):
            save_bundle(output, "vae", model, opt_state, key, metadata)
        save_bundle(output, "vae_last", model, opt_state, key, metadata)
        history.append(record)
        (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        print(json.dumps(dict(record, best_epoch=selection.best_epoch)), flush=True)
        if selection.should_stop:
            break
    result = dict(
        best_epoch=selection.best_epoch,
        best_validation_loss=selection.best_loss,
        final_epoch=epoch,
        optimizer_steps=steps,
        stopped_early=selection.should_stop,
        training_frames=len(train),
        validation_frames=len(val),
        training_finished=True,
        cpu_smoke=args.allow_cpu_smoke,
    )
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
