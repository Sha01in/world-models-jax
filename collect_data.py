import gymnasium as gym
import numpy as np
import time
import argparse
import os
import multiprocessing as mp
import cv2
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from src.config import get_config
from src.env_utils import make_env

# --- Settings (Defaults) ---
NUM_WORKERS = 12        # Default, but can be lower
NUM_EPISODES = 500      # Total episodes needed
MAX_STEPS = 2100
DATA_DIR_BASE = "data/rollouts"
IMG_SIZE = 64

def collect_episode(args_tuple):
    seed, env_name = args_tuple
    try:
        # Unique seed per process
        np.random.seed(seed)
        
        config = get_config(env_name)
        
        # Create Env
        env = make_env(env_name, render_mode="rgb_array")
        obs, _ = env.reset()
        
        obs_seq = []
        action_seq = []
        reward_seq = []
        done_seq = []
        
        for t in range(MAX_STEPS):
            # Observation is already resized by wrapper if needed, but let's be safe
            if obs.shape[:2] != (IMG_SIZE, IMG_SIZE):
                 obs_small = cv2.resize(obs, (IMG_SIZE, IMG_SIZE))
            else:
                 obs_small = obs
                 
            obs_seq.append(obs_small)
            
            # Action Strategy: Random / Brownian
            if config.is_doom:
                 # For Doom, we just sample random continuous actions [-1, 1]
                 action = np.random.uniform(-1, 1, size=(config.action_dim,))
            else:
                # CarRacing: Brownian Noise
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
        # Save to 'random' subdirectory to match new structure
        data_dir = os.path.join(DATA_DIR_BASE, env_name, "random")
        if not os.path.exists(data_dir):
            os.makedirs(data_dir, exist_ok=True)
            
        save_path = os.path.join(data_dir, f"ep_{seed}.npz")
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
    parser.add_argument("--env", type=str, default="CarRacing-v3", help="Environment name")
    args = parser.parse_args()

    episodes = args.episodes
    workers = min(args.workers, mp.cpu_count()) # Don't exceed physical cores
    env_name = args.env
    
    data_dir = os.path.join(DATA_DIR_BASE, env_name, "random")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    print(f"Starting Data Collection for {env_name}: {episodes} episodes with {workers} workers.")
    
    # Use a set of seeds as task IDs
    seeds = list(range(int(time.time()), int(time.time()) + episodes))
    
    # Prepare args for map
    map_args = [(s, env_name) for s in seeds]
    
    # Parallel Execution
    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(tqdm(executor.map(collect_episode, map_args), total=episodes))
        
    success_count = sum(results)
    print(f"Collection Complete. Success: {success_count}/{episodes}")

if __name__ == "__main__":
    main()