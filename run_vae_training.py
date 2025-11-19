import numpy as np
import glob
import os
import equinox as eqx
from train_vae import train_vae  # Importing the logic we wrote earlier

DATA_PATTERN = "data/rollouts/*.npz"
CHECKPOINT_DIR = "checkpoints"
MODEL_PATH = os.path.join(CHECKPOINT_DIR, "vae.eqx")

def load_data():
    files = glob.glob(DATA_PATTERN)
    print(f"Found {len(files)} episode files.")
    
    all_obs = []
    
    print("Loading data into memory...")
    for f in files:
        with np.load(f) as data:
            # Extract observations
            obs = data['obs'] # Shape: (T, 64, 64, 3)
            all_obs.append(obs)
            
    # Concatenate into one big array
    # Shape: (Total_Frames, 64, 64, 3)
    dataset = np.concatenate(all_obs, axis=0)
    
    # 1. Convert to Float32 (0.0 to 1.0)
    dataset = dataset.astype(np.float32) / 255.0
    
    # 2. Transpose to Channel-First format for JAX/Equinox
    # Current: (N, H, W, C) -> (N, 64, 64, 3)
    # Target:  (N, C, H, W) -> (N, 3, 64, 64)
    dataset = np.transpose(dataset, (0, 3, 1, 2))
    
    print(f"Dataset Shape: {dataset.shape} | Size: {dataset.nbytes / 1e9:.2f} GB")
    return dataset

def main():
    # 1. Prepare Data
    data = load_data()
    
    # 2. Train
    # We use more epochs since we have real data now
    print("Starting training...")
    trained_model = train_vae(data, epochs=10) 
    
    # 3. Save
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)
        
    print(f"Saving model to {MODEL_PATH}...")
    eqx.tree_serialise_leaves(MODEL_PATH, trained_model)
    print("Model saved successfully.")

if __name__ == "__main__":
    main()