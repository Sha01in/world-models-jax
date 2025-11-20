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
MIN_SCORE = 20.0        # We can raise the bar now!

def get_heuristic_action(obs, t):
    """
    A simple 'cheat' driver that looks at the pixels.
    The road is gray (approx 100-120 in RGB).
    """
    # Crop to the area just ahead of the car
    # Car is at bottom center. Look at rows 60-80.
    crop = obs[60:84, :, :]
    
    # Detect Road: Approximate gray color
    # Grayscale thresholds for the road in CarRacing
    lower_gray = np.array([90, 90, 90])
    upper_gray = np.array([120, 120, 120])
    
    mask = cv2.inRange(crop, lower_gray, upper_gray)
    
    # Find center of the road
    moments = cv2.moments(mask)
    if moments['m00'] > 0:
        cx = int(moments['m10'] / moments['m00'])
        # Center of image is 48 (96/2)
        error = cx - 48
        
        # P-Controller
        steer = error * 0.02
        steer = np.clip(steer, -1.0, 1.0)
        
        # Gas logic: Slow down on turns
        if abs(steer) > 0.3:
            gas = 0.1
            brake = 0.0
        else:
            gas = 0.5
            brake = 0.0
            
    else:
        # Lost the road? Random search
        steer = np.random.uniform(-0.5, 0.5)
        gas = 0.1
        brake = 0.0

    # Add Noise! We don't want perfect driving.
    # The VAE needs to see slight errors to learn robustly.
    steer += np.random.normal(0, 0.1)
    
    return np.array([steer, gas, brake])

def collect_data():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    env = gym.make("CarRacing-v3", render_mode="rgb_array")
    
    print(f"Collecting {NUM_EPISODES} heuristic episodes (Score > {MIN_SCORE})...")
    
    collected_counts = 0
    attempts = 0
    recent_scores = []
    pbar = tqdm(total=NUM_EPISODES)
    
    while collected_counts < NUM_EPISODES:
        attempts += 1
        obs, _ = env.reset()
        
        obs_seq, action_seq, reward_seq, done_seq = [], [], [], []
        total_reward = 0
        
        for t in range(MAX_STEPS):
            # Resize for storage
            obs_small = cv2.resize(obs, (IMG_SIZE, IMG_SIZE))
            obs_seq.append(obs_small)
            
            # Use Heuristic
            action = get_heuristic_action(obs, t)
            
            obs, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            
            # Record
            action_seq.append(action)
            reward_seq.append(reward)
            done_seq.append(done)
            total_reward += reward
            
            if done:
                break
        
        recent_scores.append(total_reward)
        if len(recent_scores) > 50:
            recent_scores.pop(0)
        
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
        
        if attempts % 5 == 0:
            avg = np.mean(recent_scores)
            pbar.set_description(f"Saved: {collected_counts} | Avg Score: {avg:.1f}")
            
    env.close()
    print(f"\nData collection complete. Total Attempts: {attempts}")

if __name__ == "__main__":
    collect_data()