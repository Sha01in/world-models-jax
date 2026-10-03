import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import numpy as np
import glob
import os
import sys
import argparse
import random
import json
from dataclasses import dataclass
from pathlib import Path
from src.rnn import MDNRNN, load_rnn
from src.config import get_config
from tqdm import tqdm

# Settings (Defaults)
BATCH_SIZE = 100
LEARNING_RATE = 1e-3
EPOCHS = 20
MAX_SEQ_LEN = 2100


@dataclass
class ValidationStopper:
    patience: int = 0
    best_loss: float = float("inf")
    best_epoch: int | None = None
    bad_epochs: int = 0

    def observe(self, epoch, value):
        if not np.isfinite(value):
            raise FloatingPointError("Non-finite held-out loss")
        improved = value < self.best_loss
        if improved:
            self.best_loss, self.best_epoch, self.bad_epochs = value, epoch, 0
        elif epoch > 0:
            self.bad_epochs += 1
        return improved, bool(self.patience and self.bad_epochs >= self.patience)


def pad_sequence(sequences, max_len, pad_value=0.0):
    batch_size = len(sequences)
    # Check dim
    if sequences[0].ndim > 1:
        feat_dim = sequences[0].shape[1]
        padded = np.full((batch_size, max_len, feat_dim), pad_value, dtype=np.float32)
    else:
        padded = np.full((batch_size, max_len), pad_value, dtype=np.float32)

    mask = np.zeros((batch_size, max_len), dtype=np.float32)

    for i, seq in enumerate(sequences):
        length = min(len(seq), max_len)
        if sequences[0].ndim > 1:
            padded[i, :length, :] = seq[:length]
        else:
            padded[i, :length] = seq[:length]
        mask[i, :length] = 1.0

    return padded, mask


def get_file_paths(data_dir):
    files = glob.glob(os.path.join(data_dir, "*.npz"))
    if not files:
        print(f"\n[ERROR] No processed data found in '{data_dir}'")
        sys.exit(1)
    print(f"Found {len(files)} processed episodes in {data_dir}.")
    return files


def load_batch(
    files,
    max_seq_len=MAX_SEQ_LEN,
    sample_posterior=False,
    rng=None,
    canonical_actions=False,
):
    all_z = []
    all_actions = []
    all_rewards = []
    all_dones = []

    for f in files:
        try:
            with np.load(f) as data:
                z = data["mu"].astype(np.float32)
                if sample_posterior:
                    if rng is None:
                        raise ValueError("Posterior sampling needs an explicit RNG")
                    z += np.exp(0.5 * data["logvar"]) * rng.standard_normal(
                        z.shape
                    ).astype(np.float32)
                actions = data["actions"].astype(np.float32)
                if canonical_actions:
                    actions = np.where(
                        actions < -0.3, -1.0, np.where(actions > 0.3, 1.0, 0.0)
                    ).astype(np.float32)
                if not (
                    len(z) == len(actions) == len(data["rewards"]) == len(data["dones"])
                ):
                    raise ValueError("Observation/action/reward/done lengths disagree")
                if np.any(data["dones"][:-1]):
                    raise ValueError(
                        "Episode contains a transition after a terminal label"
                    )
                all_z.append(z)
                all_actions.append(actions)
                all_rewards.append(data["rewards"])
                all_dones.append(data["dones"])
        except Exception as e:
            raise ValueError(f"Error loading {f}: {e}") from e

    if not all_z:
        return None, None, None, None, None

    # Pad sequences
    # A few fixed length buckets avoid computing 2100 padded steps for short episodes.
    longest = min(max(len(z) for z in all_z), max_seq_len)
    padded_len = min(
        next((n for n in (64, 128, 256, 512, 1024, 2100) if n >= longest), max_seq_len),
        max_seq_len,
    )
    X_z, mask = pad_sequence(all_z, padded_len)
    X_action, _ = pad_sequence(all_actions, padded_len)
    X_reward, _ = pad_sequence(all_rewards, padded_len)
    X_done, _ = pad_sequence(all_dones, padded_len)

    return X_z, X_action, X_reward, X_done, mask


def transition_targets(z, actions, rewards, dones, mask):
    """Stored rewards[t]/dones[t] are outcomes of (obs[t], action[t]).

    Keep the final action's death label, although its next observation is absent.
    """
    next_z = np.concatenate([z[:, 1:], np.zeros_like(z[:, :1])], axis=1)
    next_valid = np.concatenate([mask[:, 1:], np.zeros_like(mask[:, :1])], axis=1)
    latent_mask = mask * next_valid * (1.0 - dones)
    return (
        np.concatenate([z, actions], axis=-1),
        next_z,
        rewards,
        dones,
        mask,
        latent_mask,
    )


def sequence_predictions(model, inputs, initial_state=None, restart_mask=None):
    """Run a batch of streams, clearing memory at true episode boundaries."""

    def step_fn(hidden, item):
        x, restart = item
        hidden = tuple(jnp.where(restart[:, None], 0.0, s) for s in hidden)
        (log_pi, mu, log_sigma, r_pred, d_pred), new_hidden = jax.vmap(model)(x, hidden)
        return new_hidden, (log_pi, mu, log_sigma, r_pred, d_pred)

    if initial_state is None:
        initial_state = jax.vmap(lambda _: model.init_state())(
            jnp.arange(inputs.shape[0])
        )
    if restart_mask is None:
        restart_mask = jnp.zeros(inputs.shape[:2], dtype=bool)
    return jax.lax.scan(
        step_fn,
        initial_state,
        (jnp.transpose(inputs, (1, 0, 2)), restart_mask.T),
    )


def loss_fn(
    model,
    inputs,
    targets_z,
    targets_r,
    targets_d,
    mask,
    key,
    latent_mask=None,
    done_positive_weight=1.0,
    is_doom=False,
    initial_state=None,
    restart_mask=None,
    return_state=False,
):
    final_state, (log_pi, mu, log_sigma, r_seq, d_seq) = sequence_predictions(
        model, inputs, initial_state, restart_mask
    )
    # Transpose back
    log_pi = jnp.transpose(log_pi, (1, 0, 2, 3))
    mu = jnp.transpose(mu, (1, 0, 2, 3))
    log_sigma = jnp.transpose(log_sigma, (1, 0, 2, 3))
    r_seq = jnp.transpose(r_seq, (1, 0, 2))
    d_seq = jnp.transpose(d_seq, (1, 0, 2))

    mask_seq = mask  # (B, T)
    mask_expanded = jnp.expand_dims(mask_seq, -1)  # (B, T, 1)

    # 1. MDN Loss
    y_z = jnp.expand_dims(targets_z, axis=2)
    sigma = jnp.exp(log_sigma)
    log_prob = -0.5 * (jnp.log(2 * jnp.pi) + 2 * log_sigma + ((y_z - mu) / sigma) ** 2)
    if not model.factorized:
        log_prob = jnp.sum(log_prob, axis=-1, keepdims=True)
    total_log_prob = jax.nn.logsumexp(log_pi + log_prob, axis=2)

    if latent_mask is None:
        latent_mask = mask
    masked_log_prob = total_log_prob * latent_mask[..., None]
    total_valid_steps = jnp.sum(mask)
    latent_scale = model.latent_dim if (model.factorized or is_doom) else 1
    loss_mdn = -jnp.sum(masked_log_prob) / (jnp.sum(latent_mask) * latent_scale + 1e-8)

    # 2. Reward Loss
    targets_r_exp = jnp.expand_dims(targets_r, -1)
    diff = r_seq - targets_r_exp
    asymmetric_weight = jnp.where(diff > 0, 5.0, 1.0)
    loss_reward = jnp.sum(asymmetric_weight * (diff**2) * mask_expanded) / (
        total_valid_steps + 1e-8
    )

    # 3. Done Loss
    targets_d_exp = jnp.expand_dims(targets_d, -1)
    bce = optax.sigmoid_binary_cross_entropy(d_seq, targets_d_exp)
    bce *= 1.0 + targets_d_exp * (done_positive_weight - 1.0)
    loss_done = jnp.sum(bce * mask_expanded) / (total_valid_steps + 1e-8)

    # Doom rewards are exactly one per step. Only latent and death predictions matter.
    weighted_loss = (
        loss_mdn + loss_done
        if is_doom
        else loss_mdn + 10.0 * loss_reward + 10.0 * loss_done
    )

    aux = (loss_mdn, loss_reward, loss_done)
    if return_state:
        aux = (*aux, final_state)
    return weighted_loss, aux


@eqx.filter_jit
def make_step(
    model,
    opt_state,
    inputs,
    tz,
    tr,
    td,
    mask,
    key,
    optimizer,
    latent_mask=None,
    done_positive_weight=1.0,
    is_doom=False,
):
    (loss, aux), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(
        model, inputs, tz, tr, td, mask, key, latent_mask, done_positive_weight, is_doom
    )
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss, aux


def train():
    parser = argparse.ArgumentParser(description="Train MDN-RNN World Model")
    parser.add_argument(
        "--epochs", type=int, default=EPOCHS, help="Number of epochs to train"
    )
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument(
        "--env", type=str, default="CarRacing-v3", help="Environment name"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Path to data directory (overrides default)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="RNN checkpoint path; use a new path for experiments",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Load model checkpoint and optimizer state if available",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max_files", type=int, default=None, help="Limit data for a pilot experiment"
    )
    parser.add_argument("--max_seq_len", type=int, default=MAX_SEQ_LEN)
    parser.add_argument("--validation_fraction", type=float, default=0.05)
    parser.add_argument(
        "--validation_dir",
        default=None,
        help="Use separate held-out episodes instead of splitting data_dir",
    )
    parser.add_argument(
        "--done_positive_weight",
        type=float,
        default=None,
        help="Death-class weight (Doom: 10; CarRacing: 1)",
    )
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE)
    parser.add_argument(
        "--save_best",
        action="store_true",
        help="Preserve the best held-out model and optimizer alongside the last model",
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=0,
        help="Stop after this many epochs without improvement; 0 disables stopping",
    )
    parser.add_argument(
        "--mean_latents",
        action="store_true",
        help="Disable VAE posterior sampling (ablation)",
    )
    parser.add_argument(
        "--joint_mixture",
        action="store_true",
        help="Use legacy joint mixture (ablation)",
    )
    parser.add_argument(
        "--raw_actions",
        action="store_true",
        help="Use legacy analog RNN inputs (ablation)",
    )
    args = parser.parse_args()

    epochs = args.epochs
    batch_size = args.batch_size
    env_name = args.env

    config = get_config(env_name)
    if args.done_positive_weight is None:
        args.done_positive_weight = 10.0 if config.is_doom else 1.0
    if args.done_positive_weight <= 0 or args.learning_rate <= 0 or epochs < 1:
        parser.error("Death weight, learning rate and epochs must be positive")
    if args.early_stopping_patience < 0:
        parser.error("Early-stopping patience cannot be negative")
    args.save_best = args.save_best or args.early_stopping_patience > 0

    # Paths
    if args.data_dir:
        data_dir = args.data_dir
        print(f"Using custom data directory: {data_dir}")
    else:
        data_dir = os.path.join("data/series", env_name)
    checkpoint_dir = os.path.join("checkpoints", env_name)
    model_path = args.output or os.path.join(checkpoint_dir, "rnn.eqx")
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)

    # Get all files
    all_files = sorted(get_file_paths(data_dir))
    random.Random(args.seed).shuffle(all_files)
    if args.max_files:
        all_files = all_files[: args.max_files]
    if (
        not 0 <= args.validation_fraction < 1
        or args.max_seq_len < 1
        or args.batch_size < 1
    ):
        parser.error("Invalid validation fraction, sequence length or batch size")
    if args.validation_dir:
        val_files = sorted(get_file_paths(args.validation_dir))
        if {Path(p).resolve() for p in all_files}.intersection(
            Path(p).resolve() for p in val_files
        ):
            parser.error("Training and validation directories share episodes")
    else:
        n_val = (
            max(1, int(len(all_files) * args.validation_fraction))
            if args.validation_fraction
            else 0
        )
        val_files = all_files[:n_val]
        all_files = all_files[n_val:]
    if not all_files:
        parser.error("No training episodes remain after validation split")
    if args.save_best and not val_files:
        parser.error("Best-checkpoint selection requires held-out episodes")
    best_path = Path(model_path).with_name(Path(model_path).stem + "_best.eqx")
    if args.save_best and (
        Path(model_path).exists()
        or best_path.exists()
        or Path(str(best_path) + ".json").exists()
    ):
        parser.error("Use a fresh output path to preserve selected experiments")
    num_samples = len(all_files)

    key = jax.random.PRNGKey(args.seed)
    rng = np.random.default_rng(args.seed)

    # Initialize Model
    model = MDNRNN(
        latent_dim=config.latent_dim,
        action_dim=config.action_dim,
        hidden_size=config.hidden_size,
        factorized=config.is_doom and not args.joint_mixture,
        key=key,
    )

    if args.resume:
        model = load_rnn(args.resume, config)
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0), optax.adam(args.learning_rate)
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    if args.resume and Path(args.resume + ".opt.eqx").exists():
        opt_state = eqx.tree_deserialise_leaves(args.resume + ".opt.eqx", opt_state)

    settings = dict(
        vars(args),
        factorized=model.factorized,
        posterior_sampling=config.is_doom and not args.mean_latents,
        canonical_actions=config.is_doom and not args.raw_actions,
        training_files=all_files,
        validation_files=val_files,
        jax_version=jax.__version__,
        device=str(jax.devices()[0]),
    )
    Path(model_path + ".json").write_text(json.dumps(settings, indent=2) + "\n")
    history = []

    def prepare(files, data_rng):
        z, a, r, d, m = load_batch(
            files,
            args.max_seq_len,
            settings["posterior_sampling"],
            data_rng,
            settings["canonical_actions"],
        )
        return tuple(jnp.asarray(x) for x in transition_targets(z, a, r, d, m))

    @eqx.filter_jit
    def validation_loss(m, batch):
        inputs, tz, tr, td, mask, latent_mask = batch
        return loss_fn(
            m,
            inputs,
            tz,
            tr,
            td,
            mask,
            key,
            latent_mask,
            args.done_positive_weight,
            config.is_doom,
        )[0]

    def validate():
        val_rng = np.random.default_rng(args.seed + 1)
        losses = [
            float(
                validation_loss(model, prepare(val_files[i : i + batch_size], val_rng))
            )
            for i in range(0, len(val_files), batch_size)
        ]
        return float(np.mean(losses)) if losses else None

    def save_checkpoint(path, epoch, heldout_loss):
        eqx.tree_serialise_leaves(path, model)
        eqx.tree_serialise_leaves(str(path) + ".opt.eqx", opt_state)
        Path(str(path) + ".json").write_text(
            json.dumps(
                dict(
                    settings,
                    trained_epochs=epoch,
                    global_step=int(opt_state[1][0].count),
                    validation_loss=heldout_loss,
                ),
                indent=2,
            )
            + "\n"
        )
        Path(str(path) + ".rng.json").write_text(
            json.dumps(
                dict(
                    data_rng=rng.bit_generator.state, jax_key=np.asarray(key).tolist()
                ),
                indent=2,
            )
            + "\n"
        )

    stopper = ValidationStopper(args.early_stopping_patience)
    if args.save_best:
        baseline_loss = validate()
        stopper.observe(0, baseline_loss)
        save_checkpoint(best_path, 0, baseline_loss)
        history.append(dict(epoch=0, validation_loss=baseline_loss, baseline=True))
        print(json.dumps(history[-1]), flush=True)

    print(
        f"Starting RNN (Dream) training for {env_name} on {num_samples} sequences (Streaming)..."
    )
    print(
        f"Config: Latent={config.latent_dim}, Hidden={config.hidden_size}, Action={config.action_dim}"
    )
    print(f"Max Seq Len: {args.max_seq_len}")

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)

    steps_per_epoch = (num_samples + batch_size - 1) // batch_size

    for epoch in range(epochs):
        # Shuffle files
        rng.shuffle(all_files)

        epoch_loss = 0
        with tqdm(
            range(steps_per_epoch), desc=f"Epoch {epoch + 1}/{epochs}", unit="batch"
        ) as pbar:
            for i in pbar:
                # Load Batch from Disk
                batch_files = all_files[i * batch_size : (i + 1) * batch_size]
                inputs, targets_z, targets_r, targets_d, mask_train, latent_mask = (
                    prepare(batch_files, rng)
                )

                # Train Step
                key, subkey = jax.random.split(key)
                model, opt_state, loss, (l_mdn, l_rew, l_done) = make_step(
                    model,
                    opt_state,
                    inputs,
                    targets_z,
                    targets_r,
                    targets_d,
                    mask_train,
                    subkey,
                    optimizer,
                    latent_mask,
                    args.done_positive_weight,
                    config.is_doom,
                )

                if not np.isfinite(loss.item()):
                    raise FloatingPointError(
                        "Non-finite training loss; checkpoint was not overwritten"
                    )
                epoch_loss += loss.item()
                pbar.set_postfix(
                    loss=f"{loss.item():.2f}",
                    mdn=f"{l_mdn.item():.2f}",
                    rew=f"{l_rew.item():.2f}",
                    done=f"{l_done.item():.2f}",
                )

        heldout_loss = validate()
        improved, stop = (
            stopper.observe(epoch + 1, heldout_loss)
            if args.save_best
            else (False, False)
        )
        history.append(
            {
                "epoch": epoch + 1,
                "training_loss": epoch_loss / steps_per_epoch,
                "validation_loss": heldout_loss,
                "best_validation_epoch": stopper.best_epoch if args.save_best else None,
                "epochs_without_improvement": stopper.bad_epochs,
            }
        )
        print(json.dumps(history[-1]), flush=True)
        save_checkpoint(model_path, epoch + 1, heldout_loss)
        if improved:
            save_checkpoint(best_path, epoch + 1, heldout_loss)
        Path(model_path + ".history.json").write_text(
            json.dumps(history, indent=2) + "\n"
        )
        if stop:
            print(
                f"Early stop at epoch {epoch + 1}; best held-out epoch "
                f"{stopper.best_epoch}, loss {stopper.best_loss:.6f}: {best_path}",
                flush=True,
            )
            break

    eqx.tree_serialise_leaves(model_path, model)
    print("RNN Training Complete.")


if __name__ == "__main__":
    train()
