import gymnasium as gym
import numpy as np
import time
import argparse
import os
import multiprocessing as mp
import cv2
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

# --- Settings (Defaults) ---
NUM_WORKERS = 12        # Default, but can be lower
NUM_EPISODES = 500      # Total episodes needed
MAX_STEPS = 1000
DATA_DIR = "data/rollouts"
IMG_SIZE = 64

def collect_episode(seed):
    try:
        # Unique seed per process
        np.random.seed(seed)
        
        # Create Env
        # We use render_mode="rgb_array" to get pixels
        env = gym.make("CarRacing-v3", render_mode="rgb_array")
        obs, _ = env.reset()
        
        obs_seq = []
        action_seq = []
        reward_seq = []
        done_seq = []
        
        for t in range(MAX_STEPS):
            # Resize Observation to 64x64
            obs_small = cv2.resize(obs, (IMG_SIZE, IMG_SIZE))
            obs_seq.append(obs_small)
            
            # Action Strategy: Brownian Noise
            # Action: [Steer (-1, 1), Gas (0, 1), Brake (0, 1)]
            if t == 0:
                action = np.array([0.0, 0.0, 0.0])
            else:
                # Previous action + noise
                noise = np.random.randn(3) * 0.1
                action = action_seq[-1] + noise
                
            # Clip
            action[0] = np.clip(action[0], -1.0, 1.0)
            action[1] = np.clip(action[1], 0.0, 1.0)
            action[2] = np.clip(action[2], 0.0, 1.0)
            
            action_seq.append(action)
            
            # Step
            obs, reward, term, trunc, _ = env.step(action)
            reward_seq.append(reward)
            done_seq.append(term or trunc)
            
            if term or trunc:
                break
        
        env.close()
        
        # Save Data
        # Use seed as unique ID
        save_path = os.path.join(DATA_DIR, f"ep_{seed}.npz")
        np.savez_compressed(save_path,
                            obs=np.array(obs_seq),
                            actions=np.array(action_seq),
                            rewards=np.array(reward_seq),
                            dones=np.array(done_seq))
        return True
    except Exception as e:
        print(f"Worker {seed} failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Collect Data (Random Policy)")
    parser.add_argument("--episodes", type=int, default=NUM_EPISODES, help="Total episodes to collect")
    parser.add_argument("--workers", type=int, default=NUM_WORKERS, help="Number of parallel workers")
    args = parser.parse_args()

    episodes = args.episodes
    workers = min(args.workers, mp.cpu_count()) # Don't exceed physical cores

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    
    print(f"Starting Data Collection: {episodes} episodes with {workers} workers.")
    
    # Use a set of seeds as task IDs
    seeds = list(range(int(time.time()), int(time.time()) + episodes))
    
    # Parallel Execution
    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(tqdm(executor.map(collect_episode, seeds), total=episodes))
        
    success_count = sum(results)
    print(f"Collection Complete. Success: {success_count}/{episodes}")

if __name__ == "__main__":
    main()