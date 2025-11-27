import os
import argparse
import numpy as np
from tqdm import tqdm
import shutil

def filter_data(env_name, keep_top_n=2000, min_score=None, dry_run=False):
    data_dir = os.path.join("data/rollouts", env_name, "on_policy")
    if not os.path.exists(data_dir):
        print(f"Directory not found: {data_dir}")
        return

    print(f"Scanning {data_dir}...")
    
    files = []
    for f in os.listdir(data_dir):
        if f.endswith(".npz"):
            files.append(os.path.join(data_dir, f))
            
    print(f"Found {len(files)} episodes.")
    
    episode_stats = []
    print("Reading episode metadata...")
    for filepath in tqdm(files):
        try:
            with np.load(filepath) as data:
                if 'rewards' in data:
                    rewards = data['rewards']
                    score = np.sum(rewards)
                    length = len(rewards)
                    episode_stats.append({
                        'path': filepath,
                        'score': score,
                        'length': length
                    })
                else:
                    print(f"Warning: No rewards in {filepath}, marking for deletion.")
                    episode_stats.append({
                        'path': filepath,
                        'score': -1,
                        'length': 0
                    })
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            episode_stats.append({
                'path': filepath,
                'score': -1,
                'length': 0
            })
            
    # Sort by score (descending)
    episode_stats.sort(key=lambda x: x['score'], reverse=True)
    
    to_keep = []
    to_delete = []

    if min_score is not None:
        print(f"Filtering for Score > {min_score}...")
        for item in episode_stats:
            if item['score'] > min_score:
                to_keep.append(item)
            else:
                to_delete.append(item)
    else:
        # Default behavior: Keep top N
        if len(files) <= keep_top_n:
            print(f"Total files ({len(files)}) is less than or equal to target ({keep_top_n}). No filtering needed.")
            return
        to_keep = episode_stats[:keep_top_n]
        to_delete = episode_stats[keep_top_n:]
    
    print(f"\nFiltering Results:")
    print(f"Keeping: {len(to_keep)} episodes")
    print(f"Deleting: {len(to_delete)} episodes")
    
    if len(to_keep) > 0:
        print(f"Score Range (Kept): {to_keep[-1]['score']:.1f} - {to_keep[0]['score']:.1f}")
    
    if dry_run:
        print("\n[DRY RUN] No files deleted.")
        if len(to_delete) > 0:
            print("Sample files to delete:")
            for item in to_delete[:5]:
                print(f"  {item['path']} (Score: {item['score']})")
    else:
        print("\nDeleting files...")
        deleted_count = 0
        for item in tqdm(to_delete):
            try:
                os.remove(item['path'])
                deleted_count += 1
            except OSError as e:
                print(f"Error deleting {item['path']}: {e}")
        print(f"Successfully deleted {deleted_count} files.")

def main():
    parser = argparse.ArgumentParser(description="Filter On-Policy Data")
    parser.add_argument("--env", type=str, default="VizdoomTakeCover-v0", help="Environment name")
    parser.add_argument("--keep", type=int, default=2000, help="Number of top episodes to keep (if min_score not set)")
    parser.add_argument("--min_score", type=float, default=None, help="Minimum score to keep (overrides --keep)")
    parser.add_argument("--dry_run", action="store_true", help="Dry run mode")
    args = parser.parse_args()
    
    filter_data(args.env, args.keep, args.min_score, args.dry_run)

if __name__ == "__main__":
    main()
