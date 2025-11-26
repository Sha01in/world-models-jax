import numpy as np
import os
import glob
import matplotlib.pyplot as plt

import argparse

def analyze_actions():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="VizdoomTakeCover-v0")
    parser.add_argument("--type", type=str, default="random", help="random or on_policy")
    args = parser.parse_args()
    
    env_name = args.env
    data_type = args.type
    
    data_dir = os.path.join("data/rollouts", env_name, data_type)
    files = glob.glob(os.path.join(data_dir, "*.npz"))
    
    print(f"Analyzing {len(files)} files in {data_dir}...")
    
    all_actions = []
    
    for f in files[:500]: # Sample 500 files to be fast
        try:
            with np.load(f) as data:
                actions = data['actions'] # (T, 1)
                all_actions.append(actions.flatten())
        except:
            pass
            
    all_actions = np.concatenate(all_actions)
    
    print(f"Total actions analyzed: {len(all_actions)}")
    print(f"Min: {all_actions.min():.4f}, Max: {all_actions.max():.4f}, Mean: {all_actions.mean():.4f}")
    
    # Count discrete categories based on env_utils logic
    # < -0.3: Left
    # > 0.3: Right
    # Else: No-Op
    
    n_left = np.sum(all_actions < -0.3)
    n_right = np.sum(all_actions > 0.3)
    n_noop = np.sum((all_actions >= -0.3) & (all_actions <= 0.3))
    
    total = len(all_actions)
    print(f"Left (< -0.3): {n_left} ({n_left/total*100:.1f}%)")
    print(f"Right (> 0.3): {n_right} ({n_right/total*100:.1f}%)")
    print(f"No-Op (Abs <= 0.3): {n_noop} ({n_noop/total*100:.1f}%)")

if __name__ == "__main__":
    analyze_actions()
