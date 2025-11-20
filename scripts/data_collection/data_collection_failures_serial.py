import gymnasium as gym
import numpy as np
import os
import cv2
from tqdm import tqdm

# Settings
NUM_EPISODES = 300      # 300 Bad episodes to balance the 500 Good ones
MAX_STEPS = 300         # Short episodes (crashes happen fast)
DATA_DIR = "data/rollouts_bad"
IMG_SIZE = 64

def collect_data():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    env = gym.make("CarRacing-v3", render_mode="rgb_array")
    
    print(f"Collecting {NUM_EPISODES} failure episodes (Random/Crash Data)...")
    
    for i in tqdm(range(NUM_EPISODES)):
        obs, _ = env.reset()
        
        obs_seq, action_seq, reward_seq, done_seq = [], [], [], []
        
        # Force gas initially so we actually move and crash
        action = np.array([0.0, 1.0, 0.0]) 
        
        for t in range(MAX_STEPS):
            obs_small = cv2.resize(obs, (IMG_SIZE, IMG_SIZE))
            obs_seq.append(obs_small)
            
            # Change action every 5 frames to be erratic
            if t % 5 == 0:
                action = env.action_space.sample()
                # Bias towards gas (0.5 to 1.0) to ensure high-speed crashes
                action[1] = np.random.uniform(0.5, 1.0)
            
            action_seq.append(action)
            
            obs, reward, term, trunc, _ = env.step(action)
            
            reward_seq.append(reward)
            done = term or trunc
            
            # We record the 'done' flag to teach the RNN about death
            done_seq.append(done) 
            
            if done:
                break
        
        # Save
        np.savez_compressed(
            os.path.join(DATA_DIR, f"bad_rollout_{i}.npz"),
            obs=np.array(obs_seq),
            actions=np.array(action_seq),
            rewards=np.array(reward_seq),
            dones=np.array(done_seq)
        )
            
    env.close()
    print("Failure collection complete.")

if __name__ == "__main__":
    collect_data()