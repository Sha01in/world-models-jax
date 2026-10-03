"""Test linear accessibility of bright-projectile proxies in frozen VAE means.

Targets come from color components, not human-verified object annotations. The
probe is fitted on training episodes and measured on separate held-out episodes.
It never trains or initializes a GPU model.
"""

import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2
import numpy as np
from scipy.stats import rankdata

from src.vae_data import file_sha256, frame_indices, load_split
from src.vae_training import load_frame_cache
from src.vae_features import load_feature_cache
from src.episode_statistics import episode_bootstrap


def color_targets(image):
    r, g, b = np.moveaxis(image.astype(np.float32) / 255, -1, 0)
    mask = ((r > 180 / 255) & (g > 130 / 255) & (r > 1.1 * b) & (g > 1.1 * b)).astype(
        np.uint8
    )
    _, _, stats, centers = cv2.connectedComponentsWithStats(mask, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    large, small = areas >= 3, (areas > 0) & (areas < 3)
    center = centers[1:][np.argmax(areas)] if large.any() else np.zeros(2)
    return np.array([large.any(), small.any(), *center], np.float64)


def auc(truth, scores):
    truth = np.asarray(truth, bool)
    positive, negative = int(truth.sum()), int((~truth).sum())
    if not positive or not negative:
        return None
    return float(
        (rankdata(scores)[truth].sum() - positive * (positive + 1) / 2)
        / (positive * negative)
    )


def ridge(features, targets, regularization=1):
    gram = features.T @ features
    penalty = np.eye(gram.shape[0]) * regularization
    penalty[-1, -1] = 0
    return np.linalg.solve(gram + penalty, features.T @ targets)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True)
    parser.add_argument("--frame-cache-dir", required=True)
    feature_source = parser.add_mutually_exclusive_group(required=True)
    feature_source.add_argument("--encoded-dir")
    feature_source.add_argument("--encoded-frame-cache")
    parser.add_argument("--vae", required=True)
    parser.add_argument("--source-contains", default="reproduction_refined_round2")
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--frames-per-episode", type=int, default=64)
    parser.add_argument("--validation-frames-per-episode", type=int, default=16)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=73501)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("Preserve previous probes; use a fresh output")
    predictions_output = output.with_name(output.stem + "_predictions.npz")
    if predictions_output.exists():
        raise FileExistsError("Preserve previous probe predictions; use a fresh output")
    if args.bootstrap_resamples < 1:
        parser.error("Bootstrap resample count must be positive")
    manifest = load_split(args.split)
    pools = load_frame_cache(
        args.frame_cache_dir,
        args.split,
        args.frames_per_episode,
        args.validation_frames_per_episode,
        args.seed,
    )
    vae_hash = file_sha256(args.vae)
    encoded = {}
    cached_means = None
    if args.encoded_frame_cache:
        cached_means = load_feature_cache(
            args.encoded_frame_cache, args.frame_cache_dir, args.split, args.vae
        )
    else:
        for path in sorted(Path(args.encoded_dir).rglob("*.npz")):
            with np.load(path, allow_pickle=False) as data:
                if str(data["vae_sha256"]) != vae_hash:
                    raise ValueError("Encoded frames belong to a different VAE")
                raw_hash = str(data["source_sha256"])
            if raw_hash in encoded:
                raise ValueError("Duplicate raw encoding in probe source")
            encoded[raw_hash] = path
    datasets, counts = [], []
    for partition_index, (partition, pool, limit) in enumerate(
        zip(
            ("training", "validation"),
            pools,
            (args.frames_per_episode, args.validation_frames_per_episode),
        )
    ):
        features, targets, identities, frames, offset, episodes = [], [], [], [], 0, 0
        for episode in manifest[f"{partition}_episodes"]:
            indices = frame_indices(episode, limit, args.seed)
            if args.source_contains in episode["path"]:
                if cached_means is not None:
                    features.extend(
                        np.asarray(
                            cached_means[partition_index][
                                offset : offset + len(indices)
                            ],
                            np.float64,
                        )
                    )
                else:
                    with np.load(
                        encoded[episode["sha256"]], allow_pickle=False
                    ) as data:
                        features.extend(np.asarray(data["mu"][indices], np.float64))
                targets.extend(
                    color_targets(image)
                    for image in pool[offset : offset + len(indices)]
                )
                identities.extend([episode["signature"]] * len(indices))
                frames.extend(indices)
                episodes += 1
            offset += len(indices)
        if not episodes:
            raise ValueError("No episodes in the specified probe cohort")
        datasets.append(
            (
                np.asarray(features),
                np.asarray(targets),
                np.asarray(identities),
                np.asarray(frames),
            )
        )
        counts.append(dict(episodes=episodes, frames=len(features)))
    (train_x, train_y, _, _), (val_x, val_y, val_ids, val_frames) = datasets
    center, scale = train_x.mean(axis=0), np.maximum(train_x.std(axis=0), 1e-6)
    train_x = np.column_stack(((train_x - center) / scale, np.ones(len(train_x))))
    val_x = np.column_stack(((val_x - center) / scale, np.ones(len(val_x))))
    classification = ridge(train_x, train_y[:, :2])
    scores = val_x @ classification
    classes = {}
    for index, name in enumerate(("larger_bright_component", "small_bright_component")):
        classes[name] = dict(
            training_prevalence=float(train_y[:, index].mean()),
            validation_prevalence=float(val_y[:, index].mean()),
            validation_auc=auc(val_y[:, index], scores[:, index]),
            training_auc=auc(train_y[:, index], (train_x @ classification)[:, index]),
        )
    train_positive, val_positive = train_y[:, 0].astype(bool), val_y[:, 0].astype(bool)
    coordinates = ridge(train_x[train_positive], train_y[train_positive, 2:])
    all_positions = val_x @ coordinates
    prediction = all_positions[val_positive]
    baseline = train_y[train_positive, 2:].mean(axis=0)

    def statistics(rows):
        labels = val_y[rows]
        large = labels[:, 0].astype(bool)
        if large.any():
            position_error = np.abs(
                all_positions[rows][large] - labels[large, 2:]
            ).mean(axis=0)
            baseline_error = np.abs(baseline - labels[large, 2:]).mean(axis=0)
        else:
            position_error = baseline_error = np.full(2, np.nan)
        aucs = [auc(labels[:, index], scores[rows, index]) for index in range(2)]
        return [np.nan if value is None else value for value in aucs] + [
            position_error[0],
            position_error[1],
            baseline_error[0] - position_error[0],
            baseline_error[1] - position_error[1],
        ]

    uncertainty = episode_bootstrap(
        val_ids,
        statistics,
        resamples=args.bootstrap_resamples,
        seed=args.bootstrap_seed,
    )
    uncertainty["metric_order"] = [
        "larger_component_auc",
        "small_component_auc",
        "horizontal_mae_pixels",
        "vertical_mae_pixels",
        "horizontal_error_reduction_vs_constant_pixels",
        "vertical_error_reduction_vs_constant_pixels",
    ]
    report = dict(
        target_description="Bright yellow connected components; color proxies are not verified fireball labels",
        target_thresholds=dict(
            red_min=180 / 255,
            green_min=130 / 255,
            red_blue_ratio=1.1,
            green_blue_ratio=1.1,
            large_min_pixels=3,
        ),
        fit="Linear ridge probe; standardization fitted only on training; fixed regularization=1; no validation tuning",
        split_sha256=file_sha256(args.split),
        vae_sha256=vae_hash,
        feature_source=args.encoded_frame_cache or args.encoded_dir,
        source_contains=args.source_contains,
        training=counts[0],
        validation=counts[1],
        classification=classes,
        heldout_episode_bootstrap=uncertainty,
        predictions=str(predictions_output.resolve()),
        largest_component_position_mae_pixels=np.abs(
            prediction - val_y[val_positive, 2:]
        )
        .mean(axis=0)
        .tolist(),
        constant_position_baseline_mae_pixels=np.abs(baseline - val_y[val_positive, 2:])
        .mean(axis=0)
        .tolist(),
        limitation="Linear accessibility of these proxies does not prove that all projectiles are represented, that RNN dynamics are accurate, or that a controller will survive longer.",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        predictions_output,
        episode_signatures=val_ids,
        frame_indices=val_frames,
        targets=val_y,
        presence_scores=scores,
        positions=all_positions,
        constant_position=baseline,
        split_sha256=file_sha256(args.split),
        vae_sha256=vae_hash,
    )
    report["predictions_sha256"] = file_sha256(predictions_output)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
