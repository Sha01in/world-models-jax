import gymnasium as gym
import numpy as np
import os
import cv2
from tqdm import tqdm

# Settings
NUM_EPISODES = 200     
MAX_STEPS = 1000       
DATA_DIR = "data/rollouts"
IMG_SIZE = 64

def collect_data():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    env = gym.make("CarRacing-v3", render_mode="rgb_array")
    
    print(f"Collecting {NUM_EPISODES} episodes (with Rewards/Dones)...")
    
    for episode in tqdm(range(NUM_EPISODES)):
        obs, _ = env.reset()
        
        obs_sequence = []
        action_sequence = []
        reward_sequence = []
        done_sequence = []
        
        for t in range(MAX_STEPS):
            obs_small = cv2.resize(obs, (IMG_SIZE, IMG_SIZE))
            obs_sequence.append(obs_small)
            
            action = env.action_space.sample()
            if np.random.rand() < 0.1:
                action[1] = 1.0 
            
            action_sequence.append(action)
            
            obs, reward, terminated, truncated, _ = env.step(action)
            
            reward_sequence.append(reward)
            done = terminated or truncated
            done_sequence.append(done)
            
            if done:
                break
                
        # Save everything
        np.savez_compressed(
            os.path.join(DATA_DIR, f"rollout_{episode}.npz"),
            obs=np.array(obs_sequence),
            actions=np.array(action_sequence),
            rewards=np.array(reward_sequence),   # <--- New
            dones=np.array(done_sequence)        # <--- New
        )
        
    env.close()
    print("Data collection complete.")

if __name__ == "__main__":
    collect_data()