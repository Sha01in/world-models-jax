import gymnasium as gym
import numpy as np
import os
import cv2
from tqdm import tqdm

# Settings
NUM_EPISODES = 200     # Start small (200) to test. Paper used 10,000.
MAX_STEPS = 1000       # Steps per episode
DATA_DIR = "data/rollouts"
IMG_SIZE = 64

def collect_data():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    # Note: render_mode="rgb_array" is crucial for WSL2/Headless
    env = gym.make("CarRacing-v3", render_mode="rgb_array")
    
    print(f"Collecting {NUM_EPISODES} episodes...")
    
    for episode in tqdm(range(NUM_EPISODES)):
        obs, _ = env.reset()
        
        # Storage for this episode
        obs_sequence = []
        action_sequence = []
        
        for t in range(MAX_STEPS):
            # Resize observation to 64x64 immediately to save RAM/Disk
            # Original obs is 96x96x3
            obs_small = cv2.resize(obs, (IMG_SIZE, IMG_SIZE))
            obs_sequence.append(obs_small)
            
            # Sample random action
            action = env.action_space.sample()
            
            # Heuristic: Bias slightly towards gas (action[1]) so car moves
            # otherwise it just sits at start line and VAE learns nothing.
            if np.random.rand() < 0.1:
                action[1] = 1.0  # Full gas occasionally
            
            action_sequence.append(action)
            
            obs, reward, done, truncated, _ = env.step(action)
            if done or truncated:
                break
                
        # Save to disk as compressed numpy
        np.savez_compressed(
            os.path.join(DATA_DIR, f"rollout_{episode}.npz"),
            obs=np.array(obs_sequence),
            actions=np.array(action_sequence)
        )
        
    env.close()
    print("Data collection complete.")

if __name__ == "__main__":
    collect_data()