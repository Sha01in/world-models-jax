import gymnasium as gym
import numpy as np
import os
import cv2
from tqdm import tqdm

# Settings
NUM_EPISODES = 1000
MAX_STEPS = 600 
DATA_DIR = "data/rollouts_random"
IMG_SIZE = 64

def collect_data():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    env = gym.make("CarRacing-v3", render_mode="rgb_array")
    
    print(f"Collecting {NUM_EPISODES} smooth random episodes (Brownian Noise)...")
    
    for i in tqdm(range(NUM_EPISODES)):
        obs, _ = env.reset()
        
        obs_seq, action_seq, reward_seq, done_seq = [], [], [], []
        
        # Initialize random action
        action = np.array([0.0, 0.0, 0.0])
        
        for t in range(MAX_STEPS):
            obs_small = cv2.resize(obs, (IMG_SIZE, IMG_SIZE))
            obs_seq.append(obs_small)
            
            # Brownian Noise (Ornstein-Uhlenbeck-ish)
            # Drift towards a new random target
            target = env.action_space.sample()
            
            # Strong smoothing factor (0.9 old + 0.1 new) to avoid jitter
            action = 0.9 * action + 0.1 * target
            
            action_seq.append(action)
            
            obs, reward, term, trunc, _ = env.step(action)
            
            reward_seq.append(reward)
            done = term or trunc
            
            done_seq.append(done) 
            
            if done:
                break
        
        # Save
        np.savez_compressed(
            os.path.join(DATA_DIR, f"random_rollout_{i}.npz"),
            obs=np.array(obs_seq),
            actions=np.array(action_seq),
            rewards=np.array(reward_seq),
            dones=np.array(done_seq)
        )
            
    env.close()
    print("Random collection complete.")

if __name__ == "__main__":
    collect_data()
