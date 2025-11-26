import os
import argparse
import numpy as np
from tqdm import tqdm

def check_and_clean(data_dir, dry_run=False):
    print(f"Scanning {data_dir}...")
    
    files = []
    for root, _, filenames in os.walk(data_dir):
        for filename in filenames:
            if filename.endswith(".npz"):
                files.append(os.path.join(root, filename))
                
    print(f"Found {len(files)} .npz files.")
    
    corrupt_count = 0
    empty_count = 0
    
    for filepath in tqdm(files):
        remove = False
        reason = ""
        
        # 1. Check size
        if os.path.getsize(filepath) == 0:
            remove = True
            reason = "Empty file (0 bytes)"
            empty_count += 1
        else:
            # 2. Check integrity
            try:
                with np.load(filepath) as data:
                    # Accessing keys forces a read
                    _ = data.files
            except Exception as e:
                remove = True
                reason = f"Corrupt ({e})"
                corrupt_count += 1
                
        if remove:
            if dry_run:
                print(f"[DRY RUN] Would delete: {filepath} ({reason})")
            else:
                try:
                    os.remove(filepath)
                    print(f"Deleted: {filepath} ({reason})")
                except OSError as e:
                    print(f"Error deleting {filepath}: {e}")
                    
    print("-" * 30)
    print(f"Scan Complete.")
    print(f"Empty files found: {empty_count}")
    print(f"Corrupt files found: {corrupt_count}")
    if dry_run:
        print("No files were deleted (Dry Run).")
    else:
        print(f"Total deleted: {empty_count + corrupt_count}")

def main():
    parser = argparse.ArgumentParser(description="Clean corrupt .npz files")
    parser.add_argument("--env", type=str, default="VizdoomTakeCover-v0", help="Environment name")
    parser.add_argument("--dry_run", action="store_true", help="Scan without deleting")
    parser.add_argument("--all", action="store_true", help="Scan all environments")
    args = parser.parse_args()
    
    base_dir = "data/rollouts"
    
    if args.all:
        target_dir = base_dir
    else:
        target_dir = os.path.join(base_dir, args.env)
        
    if not os.path.exists(target_dir):
        print(f"Directory not found: {target_dir}")
        return
        
    check_and_clean(target_dir, dry_run=args.dry_run)

if __name__ == "__main__":
    main()
