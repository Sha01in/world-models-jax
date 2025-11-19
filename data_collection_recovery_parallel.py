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
DATA_DIR = "data/rollouts_recovery"
IMG_SIZE = 64

def get_heuristic_action(obs, t):
    """
    A simple 'cheat' driver that looks at the pixels.
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
        steer = error * 0.02
        steer = np.clip(steer, -1.0, 1.0)
        if abs(steer) > 0.3:
            gas = 0.1
            brake = 0.0
        else:
            gas = 0.5
            brake = 0.0
    else:
        steer = np.random.uniform(-0.5, 0.5)
        gas = 0.1
        brake = 0.0

    steer += np.random.normal(0, 0.05) # Reduced noise for stability
    return np.array([steer, gas, brake])

def collect_single_episode(episode_idx):
    # Uniquely seed based on episode index to avoid identical runs
    np.random.seed(episode_idx) 
    
    try:
        env = gym.make("CarRacing-v3", render_mode="rgb_array")
        obs, _ = env.reset()
        
        obs_seq, action_seq, reward_seq, done_seq = [], [], [], []
        
        perturb_active = False
        perturb_timer = 0
        perturb_action = np.array([0.0, 0.0, 0.0])
        
        for t in range(MAX_STEPS):
            obs_small = cv2.resize(obs, (IMG_SIZE, IMG_SIZE))
            obs_seq.append(obs_small)
            
            # 1. Decide on Action
            if perturb_active:
                action = perturb_action
                perturb_timer -= 1
                if perturb_timer <= 0:
                    perturb_active = False
            else:
                # Normal Heuristic Driving
                action = get_heuristic_action(obs, t)
                
                # 2. Trigger Perturbation?
                # Only perturb if on road (gas > 0.2)
                if t > 50 and action[1] > 0.2 and np.random.rand() < 0.01:
                    perturb_active = True
                    perturb_timer = np.random.randint(5, 10) # Short burst
                    # Random hard steer
                    steer = np.random.choice([-1.0, 1.0])
                    perturb_action = np.array([steer, 0.3, 0.0])
            
            action_seq.append(action)
            
            obs, reward, term, trunc, _ = env.step(action)
            reward_seq.append(reward)
            done_seq.append(term or trunc)
            
            if term or trunc:
                break
        
        env.close()
        
        # Save locally within the worker
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
        
    print(f"Launching {NUM_ENVS} parallel workers for {NUM_EPISODES} episodes...")
    
    # Using ProcessPoolExecutor to run episodes in parallel
    with ProcessPoolExecutor(max_workers=NUM_ENVS) as executor:
        results = list(tqdm(executor.map(collect_single_episode, range(NUM_EPISODES)), total=NUM_EPISODES))
        
    print(f"Data collection complete. Success rate: {sum(results)}/{NUM_EPISODES}")

if __name__ == "__main__":
    collect_data_parallel()

