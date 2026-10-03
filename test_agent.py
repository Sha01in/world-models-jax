import numpy as np
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import equinox as eqx
import cv2
import os
import argparse
from src.vae import VAE
from src.controller import get_action_linear, get_action_mlp
from src.config import get_config
from src.env_utils import make_env
from tqdm import tqdm
import multiprocessing
import functools
import time
from src.rnn import load_rnn
from src.controller import canonical_doom_action, controller_memory

# Settings (Defaults)
NUM_EPISODES = 5
VIDEO_DIR = "videos"
DIAGNOSTICS_DIR = "diagnostics"
VIDEO_SCALE = 6


def load_models(env_name, config, checkpoint_dir=None, controller_path=None):
    checkpoint_dir = checkpoint_dir or os.path.join("checkpoints", env_name)
    vae_path = os.path.join(checkpoint_dir, "vae.eqx")
    rnn_path = os.path.join(checkpoint_dir, "rnn.eqx")
    controller_path = controller_path or os.path.join(
        checkpoint_dir, "controller_dream.npz"
    )

    key = jax.random.PRNGKey(0)
    vae = VAE(latent_dim=config.latent_dim, key=key)
    vae = eqx.tree_deserialise_leaves(vae_path, vae)

    rnn = load_rnn(rnn_path, config)

    data = np.load(controller_path)
    params = jnp.array(data["params"])

    # Check for controller type (default to linear for backward compatibility)
    if "type" in data:
        controller_type = str(data["type"])
    else:
        controller_type = "linear"

    # Check for hidden size
    if "hidden_size" in data:
        hidden_size = int(data["hidden_size"])
    else:
        hidden_size = 64  # Default

    return vae, rnn, params, controller_type, hidden_size


def collect_data_parallel(
    env_name,
    num_episodes,
    num_workers,
    save_data,
    config,
    vae,
    rnn,
    controller_params,
    controller_type,
    hidden_size,
    state_mode="h",
    posterior=False,
    canonical=False,
    seed=None,
    output_data_dir=None,
):
    from gymnasium.vector import AsyncVectorEnv, AutoresetMode

    # Define env factory
    # Must be picklable for 'spawn'
    make_env_fn = functools.partial(make_env, env_name, render_mode="rgb_array")

    # Create Vector Env
    print(f"Initializing {num_workers} environments...")
    envs = AsyncVectorEnv(
        [make_env_fn for _ in range(num_workers)],
        context="spawn",
        autoreset_mode=AutoresetMode.SAME_STEP,
    )

    # JIT compiled batched inference functions
    @jax.jit
    def encode_batch(imgs, keys):
        # imgs: (B, H, W, C) -> (B, C, H, W) / 255.0
        x = jnp.array(imgs, dtype=jnp.float32) / 255.0
        x = jnp.transpose(x, (0, 3, 1, 2))

        # VAE expects (B, C, H, W)
        # We need to vmap the VAE call or VAE supports batch?
        # Our VAE __call__ takes (C, H, W). We need vmap.
        # Actually, let's check VAE definition. Usually we vmap it.
        # The single-item function:
        def encode_single(img, key):
            features = vae.encoder(img).reshape(-1)
            mu = vae.mu_head(features)
            if posterior:
                mu += jnp.exp(0.5 * vae.logvar_head(features)) * jax.random.normal(
                    key, mu.shape
                )
            return mu

        return jax.vmap(encode_single)(x, keys)

    all_rewards = []

    @jax.jit
    def get_action_batch(zs, hs):
        if controller_type == "linear":
            return jax.vmap(get_action_linear, in_axes=(None, 0, 0, None))(
                controller_params, zs, hs, config.action_dim
            )
        elif controller_type == "mlp":
            return jax.vmap(
                lambda p, z, h, ad: get_action_mlp(p, z, h, ad, hidden_dim=hidden_size),
                in_axes=(None, 0, 0, None),
            )(controller_params, zs, hs, config.action_dim)
        else:
            return jax.vmap(get_action_linear, in_axes=(None, 0, 0, None))(
                controller_params, zs, hs, config.action_dim
            )

    @jax.jit
    def rnn_step_batch(zs, actions, hs, cs):
        # zs: (B, Latent), actions: (B, Action), hs: (B, Hidden)
        if canonical:
            actions = canonical_doom_action(actions)
        rnn_in = jnp.concatenate([zs, actions], axis=1)
        (log_pi, mu, log_sigma, r_pred, d_pred), (h_new, c_new) = jax.vmap(rnn)(
            rnn_in, (hs, cs)
        )
        return h_new, c_new

    @jax.jit
    def act_and_update(imgs, hs, cs, reset_mask, current_key):
        # Fuse the policy and memory update to avoid host dispatch between GPU stages.
        hs = jnp.where(reset_mask[:, None], 0.0, hs)
        cs = jnp.where(reset_mask[:, None], 0.0, cs)
        next_key, step_key = jax.random.split(current_key)
        zs = encode_batch(imgs, jax.random.split(step_key, num_workers))
        actions = get_action_batch(zs, controller_memory((hs, cs), state_mode))
        hs, cs = rnn_step_batch(zs, actions, hs, cs)
        return actions, hs, cs, next_key

    # Initialize State
    obs_batch, _ = envs.reset(seed=seed)
    key = jax.random.PRNGKey(seed if seed is not None else 42)
    h_batch = jnp.zeros((num_workers, config.hidden_size))
    c_batch = jnp.zeros((num_workers, config.hidden_size))
    reset_mask = np.zeros(num_workers, dtype=bool)

    # Buffers for each worker
    worker_buffers = [
        {"obs": [], "actions": [], "rewards": [], "dones": []}
        for _ in range(num_workers)
    ]

    episodes_collected = 0
    worker_episode_counts = np.zeros(num_workers, dtype=int)
    pbar = tqdm(total=num_episodes, desc="Collecting Data")

    data_dir = output_data_dir or os.path.join("data/rollouts", env_name, "on_policy")
    if save_data and not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)

    while episodes_collected < num_episodes:
        action_batch_jax, h_batch, c_batch, key = act_and_update(
            obs_batch, h_batch, c_batch, reset_mask, key
        )
        action_batch_np = np.array(action_batch_jax)

        # 3. Step Envs
        next_obs_batch, reward_batch, term_batch, trunc_batch, _ = envs.step(
            action_batch_np
        )

        # 5. Store Data & Handle Dones
        for i in range(num_workers):
            # Store current step
            worker_buffers[i]["obs"].append(obs_batch[i])
            worker_buffers[i]["actions"].append(action_batch_np[i])
            worker_buffers[i]["rewards"].append(reward_batch[i])
            done = term_batch[i] or trunc_batch[i]
            worker_buffers[i]["dones"].append(
                bool(term_batch[i]) if config.is_doom else done
            )

            if done:
                # Save Episode
                if save_data and episodes_collected < num_episodes:
                    episode_id = int(time.time() * 1000) + i + episodes_collected
                    save_path = os.path.join(data_dir, f"on_policy_ep_{episode_id}.npz")
                    np.savez_compressed(
                        save_path,
                        obs=np.array(worker_buffers[i]["obs"]),
                        actions=np.array(worker_buffers[i]["actions"]),
                        rewards=np.array(worker_buffers[i]["rewards"]),
                        dones=np.array(worker_buffers[i]["dones"]),
                        collector_initial_seed=seed if seed is not None else -1,
                        worker_index=i,
                        worker_episode_index=worker_episode_counts[i],
                    )

                # Track reward
                total_reward = sum(worker_buffers[i]["rewards"])
                if episodes_collected < num_episodes:
                    all_rewards.append(total_reward)

                if episodes_collected < num_episodes:
                    episodes_collected += 1
                    pbar.update(1)

                # Reset Buffer
                worker_buffers[i] = {
                    "obs": [],
                    "actions": [],
                    "rewards": [],
                    "dones": [],
                }

                worker_episode_counts[i] += 1

        # The next inference resets each finished worker before reading its new frame.
        reset_mask = term_batch | trunc_batch
        obs_batch = next_obs_batch

    envs.close()
    pbar.close()

    # Calculate metrics
    # Calculate metrics
    # We need to extract total reward from each episode
    # worker_buffers contains lists of rewards per step.
    # But wait, worker_buffers are reset. We need to track completed episode rewards.
    # The current implementation saves to disk immediately if save_data is True.
    # If save_data is False, we might be losing the data?
    # Let's check where 'save_episode' is called.
    # It seems we need to track episode rewards explicitly.

    print(f"Parallel collection complete. Saved {episodes_collected} episodes.")

    if len(all_rewards) > 0:
        mean_score = np.mean(all_rewards)
        std_score = np.std(all_rewards)
        print(f"Final Results ({len(all_rewards)} episodes):")
        print(f"Mean Score: {mean_score:.2f} +/- {std_score:.2f}")
        print(f"Min: {np.min(all_rewards):.2f}, Max: {np.max(all_rewards):.2f}")


def main():
    parser = argparse.ArgumentParser(description="Test Trained Agent")
    parser.add_argument(
        "--episodes", type=int, default=NUM_EPISODES, help="Number of episodes to test"
    )
    parser.add_argument(
        "--no_video", action="store_true", help="Disable video saving (faster)"
    )
    parser.add_argument(
        "--save_data",
        action="store_true",
        help="Save rollout data for training (Curriculum Learning)",
    )
    parser.add_argument(
        "--env", type=str, default="CarRacing-v3", help="Environment name"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers for data collection",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Fixed seed for reproducibility"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable verbose debug logging"
    )
    parser.add_argument(
        "--controller_type",
        type=str,
        default=None,
        choices=["linear", "mlp"],
        help="Override controller type",
    )
    parser.add_argument(
        "--hidden_size", type=int, default=None, help="Override hidden size"
    )
    parser.add_argument(
        "--checkpoint_dir", default=None, help="Experiment model directory"
    )
    parser.add_argument("--controller", default=None, help="Controller checkpoint path")
    parser.add_argument(
        "--data_dir", default=None, help="Raw rollout output directory with --save_data"
    )
    args = parser.parse_args()

    num_episodes = args.episodes
    save_video = not args.no_video
    save_data = args.save_data
    env_name = args.env
    num_workers = args.workers

    config = get_config(env_name)

    vae, rnn, controller_params, controller_type, hidden_size = load_models(
        env_name, config, args.checkpoint_dir, args.controller
    )
    controller_path = args.controller or os.path.join(
        args.checkpoint_dir or os.path.join("checkpoints", env_name),
        "controller_dream.npz",
    )
    with np.load(controller_path) as controller_data:
        state_mode = (
            str(controller_data["state_mode"])
            if "state_mode" in controller_data
            else "h"
        )
        posterior = (
            bool(controller_data["posterior_sampling"])
            if "posterior_sampling" in controller_data
            else False
        )
        canonical = (
            bool(controller_data["canonical_actions"])
            if "canonical_actions" in controller_data
            else False
        )

    # Override if provided
    if args.controller_type is not None:
        controller_type = args.controller_type
    if args.hidden_size is not None:
        hidden_size = args.hidden_size

    print(f"Loaded Controller Type: {controller_type} (Hidden={hidden_size} if MLP)")

    if num_workers > 1:
        if save_video:
            print("Warning: Parallel mode disables video saving.")

        collect_data_parallel(
            env_name,
            num_episodes,
            num_workers,
            save_data,
            config,
            vae,
            rnn,
            controller_params,
            controller_type,
            hidden_size,
            state_mode,
            posterior,
            canonical,
            args.seed,
            args.data_dir,
        )
        return

    @jax.jit
    def encode_and_recon(img, key):
        x = jnp.array(img, dtype=jnp.float32) / 255.0
        x = jnp.transpose(x, (2, 0, 1))
        features = vae.encoder(x).reshape(-1)
        z = vae.mu_head(features)
        if posterior:
            z += jnp.exp(0.5 * vae.logvar_head(features)) * jax.random.normal(
                key, z.shape
            )
        recon = vae.decoder(z) if save_video else jnp.zeros_like(x)
        return z, recon

    @jax.jit
    def get_step_action(z, h):
        if controller_type == "linear":
            return get_action_linear(controller_params, z, h, config.action_dim)
        elif controller_type == "mlp":
            return get_action_mlp(
                controller_params, z, h, config.action_dim, hidden_dim=hidden_size
            )
        else:
            # Fallback
            return get_action_linear(controller_params, z, h, config.action_dim)

    @jax.jit
    def decode_from_z(z):
        # Decode a latent vector back to an image
        recon = vae.decoder(z)
        return recon

    @jax.jit
    def rnn_next(z, a, h, c):
        if canonical:
            a = canonical_doom_action(a)
        rnn_in = jnp.concatenate([z, a], axis=0)
        (log_pi, mu, log_sigma, r_pred, d_pred), (h_new, c_new) = rnn(rnn_in, (h, c))

        # Calculate expected z (weighted average of Gaussians)
        pi = jnp.exp(log_pi)
        # mu shape: (5, 32), pi shape: (5, 1)
        expected_z = jnp.sum(pi * mu, axis=0)

        return h_new, c_new, expected_z, r_pred, jax.nn.sigmoid(d_pred[0])

    # Use render_mode="rgb_array" to get pixels
    env = make_env(env_name, render_mode="rgb_array")

    # Video dir per env
    video_dir = os.path.join(VIDEO_DIR, env_name)
    diagnostics_dir = os.path.join(DIAGNOSTICS_DIR, env_name)

    # Data saving dir
    if save_data:
        data_dir = args.data_dir or os.path.join("data/rollouts", env_name, "on_policy")
        if not os.path.exists(data_dir):
            os.makedirs(data_dir, exist_ok=True)

    if save_video:
        if not os.path.exists(video_dir):
            os.makedirs(video_dir, exist_ok=True)
        if not os.path.exists(diagnostics_dir):
            os.makedirs(diagnostics_dir, exist_ok=True)

    print(f"Testing Agent on {env_name}: {num_episodes} episodes...")
    all_scores = []

    for episode in range(num_episodes):
        # Generate a random seed for this episode
        if args.seed is not None:
            seed = args.seed + episode
        else:
            seed = np.random.randint(0, 1000000)
        obs, _ = env.reset(seed=seed)
        episode_key = jax.random.PRNGKey(seed)
        h = jnp.zeros(config.hidden_size)
        c = jnp.zeros(config.hidden_size)
        total_reward = 0
        frames_combined = []
        filmstrip_frames = []

        # Current held action
        current_action = np.zeros(config.action_dim, dtype=np.float32)

        # Previous prediction for surprise calculation
        prev_expected_z = None
        total_surprise = 0.0

        # Data collection for analysis
        telemetry_data = {
            "actions": [],
            "rewards": [],
            "z": [],
            "h_norm": [],
            "surprise": [],
            "r_pred": [],
            "death_score": [],
            "terminated": [],
        }

        # Raw data for training
        obs_seq = []
        action_seq = []
        reward_seq = []
        done_seq = []

        for t in range(2100):
            # Obs is already resized by wrapper
            obs_small = obs

            if save_data:
                obs_seq.append(obs_small)

            episode_key, step_key = jax.random.split(episode_key)
            z, recon_jax = encode_and_recon(obs_small, step_key)

            # Calculate Surprise (MSE between predicted z and actual z)
            surprise = 0.0
            if prev_expected_z is not None:
                surprise = jnp.mean((z - prev_expected_z) ** 2)
                total_surprise += surprise

            # Record Video
            if save_video:
                # 1. Real Observation (already in obs_small)
                # 2. VAE Reconstruction
                recon_img = jnp.transpose(recon_jax, (1, 2, 0))
                recon_img = jnp.array(recon_img * 255.0, dtype=jnp.uint8)

                # 3. RNN Dream (Prediction for NEXT frame, shifted by 1 for vis?)
                if prev_expected_z is not None:
                    dream_jax = decode_from_z(prev_expected_z)
                    dream_img = jnp.transpose(dream_jax, (1, 2, 0))
                    dream_img = jnp.array(dream_img * 255.0, dtype=jnp.uint8)
                else:
                    dream_img = jnp.zeros_like(recon_img)

                # Combine: [Real | Recon | Dream]
                combined = np.hstack(
                    (obs_small, np.array(recon_img), np.array(dream_img))
                )
                frames_combined.append(combined)

                # Add to filmstrip every 20 frames
                if t % 20 == 0:
                    filmstrip_frames.append(combined)

            # --- LOGIC START ---
            # CarRacing retains its initial straight-driving warmup.
            if t < 50 and not config.is_doom:
                current_action = np.array([0.0, 0.5, 0.0], dtype=np.float32)
                rnn_action = jnp.array(current_action)

            # 2. DRIVING (Frames 50+): Use Brain
            else:
                # NO ACTION REPEAT: Change decisions every frame
                action_jax = get_step_action(z, controller_memory((h, c), state_mode))
                current_action = np.array(action_jax)
                rnn_action = action_jax

            # 3. Telemetry
            h_norm = jnp.linalg.norm(h)
            if t % 100 == 0:
                # print(f"T={t} | Action: {current_action} | H-Norm: {h_norm:.2f}")
                pass

            obs, reward, term, trunc, _ = env.step(current_action)
            total_reward += reward

            h, c, expected_z, r_pred_val, death_score = rnn_next(z, rnn_action, h, c)
            prev_expected_z = expected_z

            # Store Telemetry
            telemetry_data["actions"].append(current_action)
            telemetry_data["z"].append(z)
            telemetry_data["h_norm"].append(h_norm)
            telemetry_data["surprise"].append(surprise)
            telemetry_data["rewards"].append(reward)
            telemetry_data["death_score"].append(float(death_score))
            telemetry_data["terminated"].append(term)
            try:
                telemetry_data["r_pred"].append(float(r_pred_val.item()))
            except Exception:
                telemetry_data["r_pred"].append(float(r_pred_val[0]))

            # Store Training Data
            if save_data:
                action_seq.append(current_action)
                reward_seq.append(reward)
                done_seq.append(term if config.is_doom else term or trunc)

            if term or trunc:
                break

        avg_surprise = total_surprise / t if t > 0 else 0.0
        all_scores.append(total_reward)

        # Enhanced Logging
        # 1. Action Distribution
        # Use telemetry_data['actions'] which is always populated
        actions_np = np.array(telemetry_data["actions"]).flatten()

        if np.isnan(actions_np).any():
            action_dist = "NAN_DETECTED"
        elif len(actions_np) == 0:
            action_dist = "EMPTY"
        elif config.is_doom:
            # Doom actions are continuous but mapped to discrete.
            # < -0.3 Left, > 0.3 Right, else Wait
            lefts = np.sum(actions_np < -0.3)
            rights = np.sum(actions_np > 0.3)
            waits = len(actions_np) - lefts - rights
            total = len(actions_np)
            action_dist = (
                f"L:{lefts / total:.2f}|R:{rights / total:.2f}|W:{waits / total:.2f}"
            )
        else:
            # CarRacing: Mean Steer/Gas/Brake
            # Reshape back to (T, 3) for mean calculation if needed, but for now just print mean
            means = np.mean(actions_np)
            action_dist = f"Mean:{means:.2f}"

        # 2. RNN Confidence (Predicted Reward/Survival)
        # A constant unit reward does not estimate survival probability.
        prediction_label = "Avg death score" if config.is_doom else "Avg R_Pred"
        prediction_key = "death_score" if config.is_doom else "r_pred"
        r_preds = np.array(telemetry_data[prediction_key])
        avg_r_pred = np.mean(r_preds) if len(r_preds) > 0 else 0.0

        if args.debug:
            print(
                f"Episode {episode + 1}: Score = {total_reward:.1f} | Avg Surprise = {avg_surprise:.4f} | Actions: {action_dist} | {prediction_label}: {avg_r_pred:.4f}"
            )

            # 3. Last 20 Actions (The "Death Sequence")
            if config.is_doom:
                # Decode continuous back to discrete for readability
                # < -0.3 Left (L), > 0.3 Right (R), else Wait (.)
                last_actions = actions_np[-20:]
                seq_str = ""
                for a in last_actions:
                    if a < -0.3:
                        seq_str += "L"
                    elif a > 0.3:
                        seq_str += "R"
                    else:
                        seq_str += "."
                print(f"    Death Sequence (Last 20): [{seq_str}]")
        else:
            # Minimal logging
            print(f"Episode {episode + 1}: Score = {total_reward:.1f}")

        # Save Training Data
        if save_data:
            save_path = os.path.join(data_dir, f"on_policy_ep_{seed}.npz")
            np.savez_compressed(
                save_path,
                obs=np.array(obs_seq),
                actions=np.array(action_seq),
                rewards=np.array(reward_seq),
                dones=np.array(done_seq),
            )

        # Save Telemetry
        telemetry_dir = os.path.join("telemetry", env_name)
        if not os.path.exists(telemetry_dir):
            os.makedirs(telemetry_dir, exist_ok=True)

        np.savez(
            os.path.join(telemetry_dir, f"ep_{episode + 1}.npz"),
            seed=seed,
            actions=np.array(telemetry_data["actions"]),
            rewards=np.array(telemetry_data["rewards"]),
            z=np.array(telemetry_data["z"]),
            h_norm=np.array(telemetry_data["h_norm"]),
            surprise=np.array(telemetry_data["surprise"]),
            r_pred=np.array(telemetry_data["r_pred"]),
            death_score=np.array(telemetry_data["death_score"]),
            terminated=np.array(telemetry_data["terminated"]),
        )

        # Save Video per episode
        if save_video and frames_combined:
            h_orig, w_orig, _ = frames_combined[0].shape
            h_scaled, w_scaled = h_orig * VIDEO_SCALE, w_orig * VIDEO_SCALE

            video_path = os.path.join(video_dir, f"final_agent_ep{episode + 1}.mp4")

            # Try H.264 (avc1) for better quality, fallback to mp4v
            fourcc_names = ["avc1", "mp4v"]
            video = None

            for name in fourcc_names:
                fourcc = cv2.VideoWriter_fourcc(*name)
                # Determine FPS based on environment
                fps = 35 if config.is_doom else 50
                temp_video = cv2.VideoWriter(
                    video_path, fourcc, fps, (w_scaled, h_scaled)
                )
                if temp_video.isOpened():
                    video = temp_video
                    print(f"Using codec: {name}")
                    break

            if video is None:
                print("Error: Could not create video writer.")
            else:
                for f in frames_combined:
                    # Nearest neighbor scaling to preserve pixel sharpness
                    f_scaled = cv2.resize(
                        f, (w_scaled, h_scaled), interpolation=cv2.INTER_NEAREST
                    )
                    video.write(cv2.cvtColor(f_scaled, cv2.COLOR_RGB2BGR))
                video.release()
                print(f"Video saved to {video_path} ({w_scaled}x{h_scaled})")

            # Save Filmstrip per episode
            if len(filmstrip_frames) > 0:
                filmstrip_img = np.hstack(filmstrip_frames)
                fs_path = os.path.join(
                    diagnostics_dir, f"debug_filmstrip_ep{episode + 1}.png"
                )
                cv2.imwrite(fs_path, cv2.cvtColor(filmstrip_img, cv2.COLOR_RGB2BGR))
                print(f"Saved {fs_path}")

    env.close()
    print(
        f"Mean score over {len(all_scores)} episodes: {np.mean(all_scores):.2f} +/- {np.std(all_scores):.2f}"
    )


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    main()
