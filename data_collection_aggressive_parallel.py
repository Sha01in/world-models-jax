import gymnasium as gym
import numpy as np
import os
import cv2
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

# Settings
NUM_ENVS = 16           # Parallel workers
NUM_EPISODES = 500      # Total episodes
MAX_STEPS = 600 
DATA_DIR = "data/rollouts_aggressive"
IMG_SIZE = 64

def get_aggressive_action(obs, t):
    """
    A heuristic driver that enters corners too fast.
    It delays braking and steering until the last moment.
    """
    # Crop to the area just ahead of the car
    crop = obs[60:84, :, :]
    
    # Detect Road
    lower_gray = np.array([90, 90, 90])
    upper_gray = np.array([120, 120, 120])
    mask = cv2.inRange(crop, lower_gray, upper_gray)
    
    moments = cv2.moments(mask)
    if moments['m00'] > 0:
        cx = int(moments['m10'] / moments['m00'])
        error = cx - 48
        
        # Aggressive Steering: Less sensitive, waits for error to build up
        # Normal was 0.02, this is 0.015 so it reacts later
        steer = error * 0.015
        steer = np.clip(steer, -1.0, 1.0)
        
        # Aggressive Gas: Refuses to brake until steer is extreme
        # Normal threshold 0.3, Aggressive 0.8
        if abs(steer) > 0.8:
            gas = 0.0
            brake = 0.5 # Panic brake
        else:
            gas = 0.8 # Full gas on straights/mild turns
            brake = 0.0
    else:
        # Lost the road? Hard random turn
        steer = np.random.choice([-1.0, 1.0])
        gas = 0.3
        brake = 0.0

    return np.array([steer, gas, brake])

def collect_single_episode(episode_idx):
    # Uniquely seed based on episode index
    np.random.seed(episode_idx + 10000) 
    
    try:
        env = gym.make("CarRacing-v3", render_mode="rgb_array")
        obs, _ = env.reset()
        
        obs_seq, action_seq, reward_seq, done_seq = [], [], [], []
        
        for t in range(MAX_STEPS):
            obs_small = cv2.resize(obs, (IMG_SIZE, IMG_SIZE))
            obs_seq.append(obs_small)
            
            action = get_aggressive_action(obs, t)
            
            action_seq.append(action)
            
            obs, reward, term, trunc, _ = env.step(action)
            reward_seq.append(reward)
            done_seq.append(term or trunc)
            
            if term or trunc:
                break
        
        env.close()
        
        # Save locally
        save_path = os.path.join(DATA_DIR, f"episode_{episode_idx}.npz")
        np.savez_compressed(
            save_path,
            obs=np.array(obs_seq),
            actions=np.array(action_seq),
            rewards=np.array(reward_seq),
            dones=np.array(done_seq)
        )
        return True
    except Exception as e:
        print(f"Error in episode {episode_idx}: {e}")
        return False

def collect_data_parallel():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    print(f"Launching {NUM_ENVS} parallel workers for {NUM_EPISODES} aggressive episodes...")
    
    with ProcessPoolExecutor(max_workers=NUM_ENVS) as executor:
        results = list(tqdm(executor.map(collect_single_episode, range(NUM_EPISODES)), total=NUM_EPISODES))
        
    print(f"Data collection complete. Success rate: {sum(results)}/{NUM_EPISODES}")

if __name__ == "__main__":
    collect_data_parallel()
