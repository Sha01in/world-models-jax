import numpy as np
import glob
import os
import re

def get_file_index(filename):
    match = re.search(r'series_(\d+)_orig.npz', filename)
    return int(match.group(1)) if match else -1

data_dir = "data/series/VizdoomTakeCover-v0"
files = glob.glob(os.path.join(data_dir, "*.npz"))
files.sort(key=get_file_index)

print(f"Total Files: {len(files)}")

# Sample every 1000th file
indices = list(range(0, len(files), 1000))
if indices[-1] != len(files) - 1:
    indices.append(len(files) - 1)

print(f"{'Index':<10} | {'Reward':<10} | {'Frames':<10} | {'Action Mean':<20} | {'Action Std':<20}")
print("-" * 80)

for idx in indices:
    if idx >= len(files): break
    f = files[idx]
    try:
        data = np.load(f)
        reward = np.sum(data['rewards'])
        frames = len(data['mu'])
        actions = data['actions']
        act_mean = np.mean(actions, axis=0)
        act_std = np.std(actions, axis=0)
        
        # Format action stats
        if len(act_mean) == 1:
            act_str = f"[{act_mean[0]:.2f}]"
            std_str = f"[{act_std[0]:.2f}]"
        else:
            act_str = str(np.round(act_mean, 2))
            std_str = str(np.round(act_std, 2))
            
        print(f"{idx:<10} | {reward:<10.2f} | {frames:<10} | {act_str:<20} | {std_str:<20}")
    except Exception as e:
        print(f"{idx:<10} | Error: {e}")
