import gymnasium as gym
import numpy as np
import os
import cv2
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

# Settings
NUM_ENVS = 16           # Parallel workers
NUM_EPISODES = 500      
MAX_STEPS = 1000        
DATA_DIR = "data/rollouts"
IMG_SIZE = 64
# Lower threshold for "Smart Random"
# Anything > 5 means we stayed on track somewhat
MIN_SCORE = 5.0         

def collect_single_episode(episode_idx):
    # Uniquely seed based on episode index to avoid identical runs
    np.random.seed(episode_idx * 777)
    
    env = gym.make("CarRacing-v3", render_mode="rgb_array")
    obs, _ = env.reset(seed=episode_idx)
    
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
            
    env.close()
    
    if total_reward > MIN_SCORE:
        save_path = os.path.join(DATA_DIR, f"rollout_{episode_idx}.npz")
        np.savez_compressed(
            save_path,
            obs=np.array(obs_seq),
            actions=np.array(action_seq),
            rewards=np.array(reward_seq),
            dones=np.array(done_seq)
        )
        return True, total_reward
    
    return False, total_reward

def collect_data():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    print(f"Collecting {NUM_EPISODES} smart random episodes (Score > {MIN_SCORE}) in parallel...")
    
    collected_counts = 0
    attempts = 0
    pbar = tqdm(total=NUM_EPISODES)
    
    # Using ProcessPoolExecutor to run episodes in parallel
    with ProcessPoolExecutor(max_workers=NUM_ENVS) as executor:
        # Launch an initial batch
        BATCH_MULTIPLIER = 1.5
        
        while collected_counts < NUM_EPISODES:
            needed = NUM_EPISODES - collected_counts
            to_launch = int(needed * BATCH_MULTIPLIER) + 1
            
            futures = [executor.submit(collect_single_episode, attempts + i) for i in range(to_launch)]
            attempts += to_launch
            
            for f in futures:
                success, score = f.result()
                if success:
                    collected_counts += 1
                    pbar.update(1)
                    pbar.set_description(f"Saved: {collected_counts} (Last Score: {score:.1f})")
                
                if collected_counts >= NUM_EPISODES:
                    break
            
    print(f"\nData collection complete. Total Attempts: {attempts}")

if __name__ == "__main__":
    collect_data()

