import os
import glob
import re
import argparse

def clean_stale_series(env_name, threshold_index):
    data_dir = os.path.join("data/series", env_name)
    if not os.path.exists(data_dir):
        print(f"Directory not found: {data_dir}")
        return

    files = glob.glob(os.path.join(data_dir, "series_*.npz"))
    
    print(f"Scanning {data_dir}...")
    print(f"Found {len(files)} total files.")
    
    deleted_count = 0
    for f in files:
        basename = os.path.basename(f)
        # Match series_123_orig.npz or series_123.npz
        match = re.search(r"series_(\d+)", basename)
        if match:
            idx = int(match.group(1))
            if idx >= threshold_index:
                try:
                    os.remove(f)
                    deleted_count += 1
                except OSError as e:
                    print(f"Error deleting {f}: {e}")
    
    print(f"Deleted {deleted_count} stale files (index >= {threshold_index}).")
    remaining = len(files) - deleted_count
    print(f"Remaining files: {remaining}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="VizdoomTakeCover-v0")
    parser.add_argument("--threshold", type=int, default=4000)
    args = parser.parse_args()
    
    clean_stale_series(args.env, args.threshold)
