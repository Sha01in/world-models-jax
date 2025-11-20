import gymnasium as gym
import numpy as np
import os
from tqdm import tqdm

# Settings
NUM_ENVS = 16           # Parallel workers
NUM_EPISODES = 1000     # Total episodes
MAX_STEPS = 600 
DATA_DIR = "data/rollouts_random"
IMG_SIZE = 64

def make_env():
    """Helper to create a single environment with resizing baked in."""
    env = gym.make("CarRacing-v3", render_mode="rgb_array")
    env = gym.wrappers.ResizeObservation(env, (IMG_SIZE, IMG_SIZE))
    return env

def collect_data():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    print(f"Launching {NUM_ENVS} parallel environments...")
    envs = gym.vector.AsyncVectorEnv([make_env for _ in range(NUM_ENVS)])
    
    print(f"Collecting {NUM_EPISODES} smooth random episodes (Brownian Noise)...")
    
    # Buffers
    obs_buffers = [[] for _ in range(NUM_ENVS)]
    action_buffers = [[] for _ in range(NUM_ENVS)]
    reward_buffers = [[] for _ in range(NUM_ENVS)]
    done_buffers = [[] for _ in range(NUM_ENVS)]
    
    # State tracking for smooth noise
    current_actions = np.zeros((NUM_ENVS, 3))
    
    saved_count = 0
    obs, _ = envs.reset()
    
    pbar = tqdm(total=NUM_EPISODES)
    
    while saved_count < NUM_EPISODES:
        # 1. Generate Smooth Random Actions
        targets = envs.action_space.sample()
        
        # Apply smoothing: 0.9 * old + 0.1 * new
        current_actions = 0.9 * current_actions + 0.1 * targets
        
        # 2. Step
        next_obs, rewards, terminated, truncated, _ = envs.step(current_actions)
        dones = np.logical_or(terminated, truncated)
        
        # 3. Store
        for i in range(NUM_ENVS):
            obs_buffers[i].append(obs[i])
            action_buffers[i].append(current_actions[i])
            reward_buffers[i].append(rewards[i])
            done_buffers[i].append(dones[i])
            
            if dones[i] or len(obs_buffers[i]) >= MAX_STEPS:
                if saved_count < NUM_EPISODES:
                    np.savez_compressed(
                        os.path.join(DATA_DIR, f"random_rollout_{saved_count}.npz"),
                        obs=np.array(obs_buffers[i]),
                        actions=np.array(action_buffers[i]),
                        rewards=np.array(reward_buffers[i]),
                        dones=np.array(done_buffers[i])
                    )
                    saved_count += 1
                    pbar.update(1)
                
                # Reset buffer & noise state for this env
                obs_buffers[i] = []
                action_buffers[i] = []
                reward_buffers[i] = []
                done_buffers[i] = []
                current_actions[i] = np.zeros(3) # Reset action momentum
        
        obs = next_obs

    envs.close()
    print("Parallel random collection complete.")

if __name__ == "__main__":
    collect_data()
