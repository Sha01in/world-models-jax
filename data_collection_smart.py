import gymnasium as gym
import numpy as np
import os
import cv2
from tqdm import tqdm

# Settings
NUM_EPISODES = 500      
MAX_STEPS = 1000        
DATA_DIR = "data/rollouts"
IMG_SIZE = 64

# NEW: Lower threshold. 
# Anything > 0 means we stayed on track longer than we were off it.
MIN_SCORE = 5.0         

def collect_data():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    env = gym.make("CarRacing-v3", render_mode="rgb_array")
    
    print(f"Collecting {NUM_EPISODES} episodes (Score > {MIN_SCORE})...")
    
    collected_counts = 0
    attempts = 0
    
    # Buffer for logging recent scores
    recent_scores = []
    
    pbar = tqdm(total=NUM_EPISODES)
    
    while collected_counts < NUM_EPISODES:
        attempts += 1
        obs, _ = env.reset()
        
        obs_seq, action_seq, reward_seq, done_seq = [], [], [], []
        total_reward = 0
        
        # Start with gentle gas
        action = np.array([0.0, 0.5, 0.0]) 
        
        steps_to_stick = 0
        
        for t in range(MAX_STEPS):
            obs_small = cv2.resize(obs, (IMG_SIZE, IMG_SIZE))
            obs_seq.append(obs_small)
            
            if steps_to_stick <= 0:
                # Action Strategy:
                # 1. Hold actions longer (8-15 frames) to simulate smooth driving
                steps_to_stick = np.random.randint(8, 15)
                
                # 2. Gentler Steering (-0.5 to 0.5)
                # Full -1 to 1 often causes instant spin-outs
                steer = np.random.uniform(-0.5, 0.5)
                
                # 3. Constant Gas bias
                gas = np.random.uniform(0.1, 0.8)
                brake = 0.0
                
                # Occasional braking
                if np.random.rand() < 0.05: 
                    brake = np.random.uniform(0, 0.5)
                    gas = 0
                
                action = np.array([steer, gas, brake])
            
            steps_to_stick -= 1
            action_seq.append(action)
            
            obs, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            
            reward_seq.append(reward)
            done_seq.append(done)
            total_reward += reward
            
            if done:
                break
        
        recent_scores.append(total_reward)
        if len(recent_scores) > 50: recent_scores.pop(0)
        
        if total_reward > MIN_SCORE:
            np.savez_compressed(
                os.path.join(DATA_DIR, f"rollout_{collected_counts}.npz"),
                obs=np.array(obs_seq),
                actions=np.array(action_seq),
                rewards=np.array(reward_seq),
                dones=np.array(done_seq)
            )
            collected_counts += 1
            pbar.update(1)
        
        # Update description every 10 attempts so we aren't blind
        if attempts % 10 == 0:
            avg = np.mean(recent_scores)
            max_s = np.max(recent_scores)
            pbar.set_description(f"Saved: {collected_counts} | Avg Score: {avg:.1f} | Max: {max_s:.1f}")
            
    env.close()
    print(f"\nData collection complete. Total Attempts: {attempts}")

if __name__ == "__main__":
    collect_data()