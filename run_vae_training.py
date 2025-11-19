import numpy as np
import glob
import os
import equinox as eqx
from train_vae import train_vae

DATA_PATTERN = "data/rollouts/*.npz"
CHECKPOINT_DIR = "checkpoints"
MODEL_PATH = os.path.join(CHECKPOINT_DIR, "vae.eqx")

def load_data():
    files = glob.glob(DATA_PATTERN)
    print(f"Found {len(files)} episode files.")
    
    # Pre-allocate the array to avoid memory fragmentation
    # 200 episodes * 1000 steps = 200,000 frames
    # Shape: (Total, 3, 64, 64)
    total_frames = len(files) * 1000
    dataset = np.empty((total_frames, 3, 64, 64), dtype=np.uint8)
    
    print("Loading data into memory (uint8)...")
    idx = 0
    for f in files:
        with np.load(f) as data:
            obs = data['obs'] # (1000, 64, 64, 3)
            
            # Transpose to (1000, 3, 64, 64)
            obs = np.transpose(obs, (0, 3, 1, 2))
            
            # Insert directly into pre-allocated array
            dataset[idx : idx + len(obs)] = obs
            idx += len(obs)
    
    # Trim in case some files had fewer frames
    dataset = dataset[:idx]
            
    print(f"Dataset Shape: {dataset.shape}")
    print(f"RAM Usage: {dataset.nbytes / 1e9:.2f} GB (uint8)")
    return dataset

def main():
    data = load_data()
    
    print("Starting training...")
    # Note: data is now uint8 (0-255), train_vae must handle the conversion
    trained_model = train_vae(data, epochs=10) 
    
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)
        
    print(f"Saving model to {MODEL_PATH}...")
    eqx.tree_serialise_leaves(MODEL_PATH, trained_model)
    print("Model saved successfully.")

if __name__ == "__main__":
    main()