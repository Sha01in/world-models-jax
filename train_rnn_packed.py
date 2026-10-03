"""GPU Doom RNN training with packed 500-step streams and carried LSTM memory.

The reference trains for 400 epochs, batch 100, with per-value gradient clipping.
Stored physical death labels stay with their actions; timeouts reset memory but
are not positive death targets. Validation uses complete, separate episodes.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cuda")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse
import hashlib
import json
from pathlib import Path
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from src.config import get_config
from src.rnn import MDNRNN, load_rnn
from train_rnn import load_batch, loss_fn, transition_targets


def load_episode(path):
    with np.load(path) as data:
        mu = data["mu"].astype(np.float16)
        logvar = data["logvar"].astype(np.float16)
        actions = data["actions"].astype(np.float32).reshape(-1, 1)
        dones = data["dones"].astype(np.uint8)
        if not (len(mu) == len(logvar) == len(actions) == len(dones)) or not len(mu):
            raise ValueError(f"Inconsistent episode lengths: {path}")
        if (
            np.any(dones[:-1])
            or not np.all(np.isfinite(mu))
            or not np.all(np.isfinite(logvar))
        ):
            raise ValueError(f"Invalid episode: {path}")
        actions = np.where(actions < -0.3, -1, np.where(actions > 0.3, 1, 0)).astype(
            np.int8
        )
        return mu, logvar, actions, dones


def pack_episodes(episodes, order):
    """Pack true episodes without padding; mark even censored episode starts."""
    frames = sum(len(episode[0]) for episode in episodes)
    latent_dim = episodes[0][0].shape[1]
    mu = np.empty((frames, latent_dim), np.float16)
    logvar = np.empty_like(mu)
    actions = np.empty((frames, 1), np.int8)
    dones = np.empty(frames, np.uint8)
    starts = np.zeros(frames, bool)
    offset = 0
    for index in order:
        m, v, a, d = episodes[index]
        end = offset + len(m)
        mu[offset:end], logvar[offset:end] = m, v
        actions[offset:end], dones[offset:end] = a, d
        starts[offset] = True
        offset = end
    return mu, logvar, actions, dones, starts


def packed_batches(packed, batch_size, seq_len, rng, sample_posterior=True):
    """Yield contiguous chunks, reusing each boundary's sampled next observation.

    A stream may begin mid-episode after partitioning. Exclude that fragment from
    both losses until a true episode start. No latent target crosses a reset.
    The shuffled tail shorter than batch_size * seq_len is omitted each epoch.
    """
    mu, logvar, actions, dones, starts = packed
    n_steps = len(mu) // (batch_size * seq_len)
    if n_steps < 1:
        raise ValueError("Not enough frames for one packed batch")
    row_len = n_steps * seq_len
    row_offsets = np.arange(batch_size) * row_len
    has_started = np.zeros(batch_size, bool)
    boundary_z = None

    def sample(indices):
        z = mu[indices].astype(np.float32)
        if sample_posterior:
            z += np.exp(logvar[indices].astype(np.float32) * 0.5) * rng.standard_normal(
                z.shape
            ).astype(np.float32)
        return z

    for step in range(n_steps):
        indices = row_offsets[:, None] + step * seq_len + np.arange(seq_len)
        z = sample(indices)
        if boundary_z is not None:
            z[:, 0] = boundary_z
        next_indices = indices[:, -1] + 1
        next_exists = next_indices < len(mu)
        safe_next = np.minimum(next_indices, len(mu) - 1)
        boundary_z = sample(safe_next)
        targets = np.concatenate([z[:, 1:], boundary_z[:, None]], axis=1)
        restart = starts[indices]
        valid = np.maximum.accumulate(restart, axis=1) | has_started[:, None]
        has_started = valid[:, -1]
        next_start = np.concatenate([restart[:, 1:], starts[safe_next, None]], axis=1)
        latent_valid = valid & ~next_start & ~dones[indices].astype(bool)
        latent_valid[:, -1] &= next_exists
        yield (
            np.concatenate([z, actions[indices].astype(np.float32)], axis=-1),
            targets,
            np.ones(indices.shape, np.float32),
            dones[indices].astype(np.float32),
            valid.astype(np.float32),
            latent_valid.astype(np.float32),
            restart,
        )


@eqx.filter_jit
def packed_step(model, state, hidden, batch, optimizer, learning_rate):
    inputs, tz, tr, td, mask, latent_mask, restart = batch
    (loss, aux), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(
        model,
        inputs,
        tz,
        tr,
        td,
        mask,
        jax.random.PRNGKey(0),
        latent_mask,
        10.0,
        True,
        hidden,
        restart,
        True,
    )
    updates, state = optimizer.update(grads, state, eqx.filter(model, eqx.is_array))
    updates = jax.tree.map(lambda u: u * learning_rate, updates)
    model = eqx.apply_updates(model, updates)
    hidden = jax.tree.map(jax.lax.stop_gradient, aux[3])
    return model, state, hidden, loss, aux[:3]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", required=True, help="JSON training/validation file lists"
    )
    parser.add_argument("--output", required=True, help="Fresh RNN checkpoint path")
    parser.add_argument(
        "--resume", default=None, help="Initial model and optional Adam moments"
    )
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--seq-len", type=int, default=500)
    parser.add_argument("--seed", type=int, default=49)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--min-learning-rate", type=float, default=0.00001)
    parser.add_argument("--decay-rate", type=float, default=0.99999)
    parser.add_argument("--validation-every", type=int, default=5)
    parser.add_argument("--snapshot-every", type=int, default=100)
    parser.add_argument("--mean-latents", action="store_true")
    parser.add_argument("--max-files", type=int, default=None, help="Pilot only")
    args = parser.parse_args()
    if (
        min(
            args.epochs,
            args.batch_size,
            args.seq_len,
            args.validation_every,
            args.snapshot_every,
        )
        < 1
    ):
        parser.error(
            "Epochs, batch size, sequence length and intervals must be positive"
        )
    if (
        not 0 < args.min_learning_rate <= args.learning_rate
        or not 0 < args.decay_rate <= 1
    ):
        parser.error("Invalid learning-rate schedule")
    if args.max_files is not None and args.max_files < 1:
        parser.error("Pilot file count must be positive")
    if jax.devices()[0].platform != "gpu":
        raise RuntimeError("Packed training requires CUDA; CPU fallback is disabled")
    output = Path(args.output)
    if output.exists() or Path(str(output) + ".json").exists():
        raise FileExistsError(
            "Use a fresh output path to preserve existing experiments"
        )
    raw_manifest = Path(args.manifest).read_bytes()
    manifest = json.loads(raw_manifest)
    train_files = sorted(
        manifest.get("training_files", manifest.get("training_sources", []))
    )
    val_files = sorted(
        manifest.get("validation_files", manifest.get("validation_sources", []))
    )
    if not train_files or not val_files:
        raise ValueError("Separate training and validation episodes are required")
    if args.max_files:
        train_files = train_files[: args.max_files]
        val_files = val_files[: min(32, args.max_files)]
    train_sources = {str(Path(p).resolve()) for p in train_files}
    val_sources = {str(Path(p).resolve()) for p in val_files}
    if train_sources.intersection(val_sources) or len(train_sources) != len(
        train_files
    ):
        raise ValueError("Training sources repeat or overlap validation")
    config = get_config("VizdoomTakeCover-v0")
    model = (
        load_rnn(args.resume, config)
        if args.resume
        else MDNRNN(
            config.latent_dim,
            config.action_dim,
            config.hidden_size,
            key=jax.random.PRNGKey(args.seed),
            factorized=True,
        )
    )
    if not model.factorized:
        raise ValueError("Packed Doom training requires the factorized MDN")
    optimizer = optax.chain(optax.clip(1.0), optax.adam(1.0))
    state = optimizer.init(eqx.filter(model, eqx.is_array))
    if args.resume and Path(args.resume + ".opt.eqx").exists():
        state = eqx.tree_deserialise_leaves(args.resume + ".opt.eqx", state)
    global_step = int(state[1][0].count)
    settings = dict(
        vars(args),
        factorized=True,
        posterior_sampling=not args.mean_latents,
        canonical_actions=True,
        done_positive_weight=10.0,
        gradient_clip="per_value_1",
        training_files=train_files,
        validation_files=val_files,
        manifest_sha256=hashlib.sha256(raw_manifest).hexdigest(),
        initial_global_step=global_step,
        jax_version=jax.__version__,
        device=str(jax.devices()[0]),
        initial_model_sha256=hashlib.sha256(Path(args.resume).read_bytes()).hexdigest()
        if args.resume
        else None,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    Path(str(output) + ".json").write_text(json.dumps(settings, indent=2) + "\n")
    print(f"Caching {len(train_files)} training episodes as float16", flush=True)
    episodes = [load_episode(p) for p in train_files]
    frames = sum(len(e[0]) for e in episodes)
    print(
        f"CUDA device: {jax.devices()[0]}; frames={frames}; packed steps/epoch={frames // (args.batch_size * args.seq_len)}",
        flush=True,
    )
    rng = np.random.default_rng(args.seed)
    history = []
    best_loss = float("inf")
    best_path = output.with_name(output.stem + "_best.eqx")

    @eqx.filter_jit
    def validation_loss(m, batch):
        return loss_fn(m, *batch[:5], jax.random.PRNGKey(0), batch[5], 10.0, True)[0]

    def validate():
        val_rng = np.random.default_rng(args.seed + 1)
        losses, weights = [], []
        for i in range(0, len(val_files), 32):
            z, a, r, d, mask = load_batch(
                val_files[i : i + 32], 2100, not args.mean_latents, val_rng, True
            )
            batch = tuple(jnp.asarray(x) for x in transition_targets(z, a, r, d, mask))
            losses.append(float(validation_loss(model, batch)))
            weights.append(float(mask.sum()))
        return float(np.average(losses, weights=weights))

    def save_model(path, epoch):
        eqx.tree_serialise_leaves(path, model)
        Path(str(path) + ".json").write_text(
            json.dumps(
                dict(settings, trained_epochs=epoch, global_step=global_step), indent=2
            )
            + "\n"
        )

    for epoch in range(1, args.epochs + 1):
        began = time.monotonic()
        packed = pack_episodes(episodes, rng.permutation(len(episodes)))
        hidden = jax.vmap(lambda _: model.init_state())(jnp.arange(args.batch_size))
        totals = np.zeros(4)
        steps = 0
        for batch in packed_batches(
            packed, args.batch_size, args.seq_len, rng, not args.mean_latents
        ):
            rate = (
                args.min_learning_rate
                + (args.learning_rate - args.min_learning_rate)
                * args.decay_rate**global_step
            )
            model, state, hidden, loss, aux = packed_step(
                model,
                state,
                hidden,
                tuple(jnp.asarray(x) for x in batch),
                optimizer,
                jnp.asarray(rate, jnp.float32),
            )
            values = np.array([float(loss), *(float(x) for x in aux)])
            if not np.all(np.isfinite(values)):
                raise FloatingPointError(
                    "Non-finite loss; last completed epoch remains saved"
                )
            totals += values
            steps += 1
            global_step += 1
        del packed
        training_seconds = time.monotonic() - began
        val_loss = (
            validate()
            if epoch == 1 or epoch % args.validation_every == 0 or epoch == args.epochs
            else None
        )
        record = dict(
            epoch=epoch,
            global_step=global_step,
            learning_rate=rate,
            training_loss=totals[0] / steps,
            latent_loss=totals[1] / steps,
            death_loss=totals[3] / steps,
            validation_loss=val_loss,
            training_seconds=training_seconds,
            seconds=time.monotonic() - began,
        )
        history.append(record)
        save_model(output, epoch)
        eqx.tree_serialise_leaves(str(output) + ".opt.eqx", state)
        Path(str(output) + ".history.json").write_text(
            json.dumps(history, indent=2) + "\n"
        )
        Path(str(output) + ".rng.json").write_text(
            json.dumps(rng.bit_generator.state, indent=2) + "\n"
        )
        if val_loss is not None and val_loss < best_loss:
            best_loss = val_loss
            save_model(best_path, epoch)
            record["best_validation_loss"] = best_loss
        if epoch % args.snapshot_every == 0:
            save_model(output.with_name(f"{output.stem}_epoch{epoch:03d}.eqx"), epoch)
        print(json.dumps(record), flush=True)
    print(
        f"Packed RNN training complete. Best held-out loss {best_loss:.6f}: {best_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
