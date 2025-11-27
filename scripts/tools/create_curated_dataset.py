import os
import glob
import numpy as np
import shutil
import argparse
from tqdm import tqdm
import re

def get_file_index(filename):
    match = re.search(r'series_(\d+)_orig.npz', filename)
    return int(match.group(1)) if match else -1

def create_curated_dataset(env_name, delusional_start_idx=12401, top_k_historical=1500):
    data_dir = f"data/series/{env_name}"
    output_dir = f"data/series/{env_name}-curated"
    
    if not os.path.exists(data_dir):
        print(f"Error: Data directory {data_dir} not found.")
        return

    if os.path.exists(output_dir):
        print(f"Removing existing curated directory: {output_dir}")
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    files = glob.glob(os.path.join(data_dir, "*.npz"))
    files.sort(key=get_file_index)
    
    print(f"Total files found: {len(files)}")
    
    # 1. Identify Delusional Data
    delusional_files = [f for f in files if get_file_index(f) >= delusional_start_idx]
    print(f"Found {len(delusional_files)} Delusional episodes (Index >= {delusional_start_idx})")
    
    # 2. Identify Historical Data
    historical_files = [f for f in files if get_file_index(f) < delusional_start_idx]
    print(f"Found {len(historical_files)} Historical episodes")
    
    # 3. Select Top Historical Data
    print(f"Scanning historical data to find Top {top_k_historical} episodes...")
    historical_scores = []
    
    for f in tqdm(historical_files, desc="Scanning Historical"):
        try:
            data = np.load(f)
            reward = np.sum(data['rewards'])
            historical_scores.append((f, reward))
        except Exception as e:
            print(f"Error reading {f}: {e}")
            
    # Sort by reward descending
    historical_scores.sort(key=lambda x: x[1], reverse=True)
    top_historical = historical_scores[:top_k_historical]
    
    print(f"Top Historical Score Range: {top_historical[0][1]:.2f} to {top_historical[-1][1]:.2f}")
    
    selected_historical_files = [x[0] for x in top_historical]
    
    # 4. Combine and Symlink
    final_files = delusional_files + selected_historical_files
    print(f"Creating symlinks for {len(final_files)} total files...")
    
    for src in tqdm(final_files, desc="Symlinking"):
        basename = os.path.basename(src)
        dst = os.path.join(output_dir, basename)
        os.symlink(os.path.abspath(src), dst)
        
    print(f"\nCurated dataset created at: {output_dir}")
    print(f"Composition: {len(delusional_files)} Delusional + {len(selected_historical_files)} Historical")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="VizdoomTakeCover-v0")
    args = parser.parse_args()
    
    create_curated_dataset(args.env)
