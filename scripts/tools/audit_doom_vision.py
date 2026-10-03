"""Measure VAE reconstruction on unseen fatal windows and save an inspection atlas.

The atlas and whole-image metrics are diagnostics, not projectile ground truth.
Optional manually annotated projectile boxes measure localized retention.
"""

import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image, ImageDraw

from src.vae import load_vae
from src.vae_data import file_sha256, load_frames, load_split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True)
    parser.add_argument("--vae", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--source-contains", default="reproduction_refined_round2")
    parser.add_argument("--annotations", default=None)
    args = parser.parse_args()
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a fresh audit directory")
    if args.episodes < 1:
        parser.error("At least one episode is required")
    manifest = load_split(args.split)
    episodes = [
        e
        for e in manifest["validation_episodes"]
        if args.source_contains in e["path"] and e["physical_death"]
    ]
    if not episodes:
        raise ValueError("No fatal held-out episodes match this cohort")
    rng = np.random.default_rng(args.seed)
    selected = rng.choice(
        len(episodes), min(args.episodes, len(episodes)), replace=False
    )
    frames, records = [], []
    for row, index in enumerate(selected):
        episode = episodes[index]
        # obs[-1] is the last pre-action frame; no synthetic post-death image.
        indices = np.maximum(0, episode["frames"] - np.array([33, 17, 9, 1]))
        images = load_frames(episode, indices)
        frames.extend(images)
        records.extend(
            dict(
                row=row,
                path=episode["path"],
                signature=episode["signature"],
                frame=int(i),
                steps_before_fatal_action=int(episode["frames"] - 1 - i),
            )
            for i in indices
        )
    original = np.asarray(frames)
    model = load_vae(args.vae, 64)

    @eqx.filter_jit
    def reconstruct(images, keys):
        def one(image, key):
            mu, logvar = model.encode(image)
            posterior = mu + jnp.exp(0.5 * logvar) * jax.random.normal(key, mu.shape)
            return model.decoder(mu), model.decoder(posterior), mu, logvar

        return jax.vmap(one)(images, keys)

    mean_images, sampled_images, mus, logvars = [], [], [], []
    key = jax.random.PRNGKey(args.seed)
    for start in range(0, len(frames), 8):
        images = (
            original[start : start + 8].transpose(0, 3, 1, 2).astype(np.float32) / 255
        )
        keys = jax.vmap(lambda i: jax.random.fold_in(key, i))(
            jnp.arange(start, start + len(images), dtype=jnp.uint32)
        )
        mean, sampled, mu, logvar = map(np.asarray, reconstruct(images, keys))
        mean_images.extend(mean.transpose(0, 2, 3, 1))
        sampled_images.extend(sampled.transpose(0, 2, 3, 1))
        mus.extend(mu)
        logvars.extend(logvar)
    mean_images, sampled_images = np.asarray(mean_images), np.asarray(sampled_images)
    original_float = original.astype(np.float32) / 255
    for index, record in enumerate(records):
        record["mean_pixel_mse"] = float(
            np.mean((original_float[index] - mean_images[index]) ** 2)
        )
        record["posterior_pixel_mse"] = float(
            np.mean((original_float[index] - sampled_images[index]) ** 2)
        )
    boxes = []
    if args.annotations:
        annotations = json.loads(Path(args.annotations).read_text())
        lookup = {(r["signature"], r["frame"]): i for i, r in enumerate(records)}
        for annotation in annotations["projectile_boxes"]:
            index = lookup[(annotation["signature"], annotation["frame"])]
            x0, y0, x1, y1 = annotation["box_xyxy"]
            if not (0 <= x0 < x1 <= 64 and 0 <= y0 < y1 <= 64):
                raise ValueError("Projectile boxes must fit the original 64x64 image")
            original_roi = original_float[index, y0:y1, x0:x1]
            boxes.append(
                dict(
                    annotation,
                    mean_roi_mse=float(
                        np.mean((original_roi - mean_images[index, y0:y1, x0:x1]) ** 2)
                    ),
                    posterior_roi_mse=float(
                        np.mean(
                            (original_roi - sampled_images[index, y0:y1, x0:x1]) ** 2
                        )
                    ),
                )
            )
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "frames.npz",
        original=original,
        mean=mean_images,
        posterior=sampled_images,
        mu=np.asarray(mus),
        logvar=np.asarray(logvars),
    )
    # Four times, each showing original | mean | posterior, at 2x pixel scale.
    sheet = Image.new("RGB", (4 * 3 * 128, len(selected) * 154), "white")
    draw = ImageDraw.Draw(sheet)
    for index, record in enumerate(records):
        column, row = index % 4, index // 4
        draw.text(
            (column * 384 + 2, row * 154 + 2),
            f"ep{row} t={record['frame']} (-{record['steps_before_fatal_action']}): raw | mean | sample",
            fill="black",
        )
        panels = (
            original[index],
            np.uint8(np.clip(mean_images[index] * 255, 0, 255)),
            np.uint8(np.clip(sampled_images[index] * 255, 0, 255)),
        )
        for panel, pixels in enumerate(panels):
            image = Image.fromarray(pixels).resize((128, 128), Image.Resampling.NEAREST)
            sheet.paste(image, (column * 384 + panel * 128, row * 154 + 22))
    sheet.save(output / "atlas.png")
    report = dict(
        vae=str(Path(args.vae).resolve()),
        vae_sha256=file_sha256(args.vae),
        architecture=model.architecture,
        device=str(jax.devices()[0]),
        seed=args.seed,
        split_sha256=file_sha256(args.split),
        episodes=len(selected),
        frames=len(frames),
        cohort_filter=args.source_contains,
        mean_pixel_mse=float(np.mean((original_float - mean_images) ** 2)),
        posterior_pixel_mse=float(np.mean((original_float - sampled_images) ** 2)),
        projectile_boxes=boxes,
        records=records,
        limitation="Whole-image error and an atlas do not establish projectile detection or gameplay performance; annotation metrics measure only listed boxes.",
    )
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k not in ("records", "projectile_boxes")
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
