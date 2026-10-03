"""Prepare a shared, fingerprinted uint8 frame pool without GPU work."""

import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.vae_data import load_split
from src.vae_training import prepare_frame_cache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--frames-per-episode", type=int, default=64)
    parser.add_argument("--validation-frames-per-episode", type=int, default=16)
    args = parser.parse_args()
    if min(args.frames_per_episode, args.validation_frames_per_episode) < 0:
        parser.error("Frame limits must be nonnegative; zero means all frames")
    arrays = prepare_frame_cache(
        load_split(args.split),
        args.split,
        args.output_dir,
        args.frames_per_episode,
        args.validation_frames_per_episode,
        args.seed,
    )
    print(
        json.dumps(
            dict(training_frames=len(arrays[0]), validation_frames=len(arrays[1]))
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
