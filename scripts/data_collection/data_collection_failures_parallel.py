import gymnasium as gym
import numpy as np
import os
from tqdm import tqdm
import cv2

# Settings
NUM_ENVS = 16           # Number of parallel workers (adjust to your CPU cores)
TOTAL_EPISODES = 300    # Total bad episodes needed
MAX_STEPS = 300         # Short episodes for crashing
DATA_DIR = "data/rollouts_bad"
IMG_SIZE = 64

def make_env():
    """Helper to create a single environment with resizing baked in."""
    env = gym.make("CarRacing-v3", render_mode="rgb_array")
    # Resize inside the worker process to save bandwidth
    env = gym.wrappers.ResizeObservation(env, (IMG_SIZE, IMG_SIZE))
    return env

def collect_data():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    print(f"Launching {NUM_ENVS} parallel environments...")
    # AsyncVectorEnv runs each env in a separate process
    envs = gym.vector.AsyncVectorEnv([make_env for _ in range(NUM_ENVS)])
    
    print(f"Collecting {TOTAL_EPISODES} failure episodes...")
    
    # Storage for each active environment
    # We keep lists of lists: buffers[env_index] = [frame1, frame2, ...]
    obs_buffers = [[] for _ in range(NUM_ENVS)]
    action_buffers = [[] for _ in range(NUM_ENVS)]
    reward_buffers = [[] for _ in range(NUM_ENVS)]
    done_buffers = [[] for _ in range(NUM_ENVS)]
    
    saved_count = 0
    obs, _ = envs.reset()
    
    pbar = tqdm(total=TOTAL_EPISODES)
    
    while saved_count < TOTAL_EPISODES:
        # 1. Generate Random Actions (Biased towards gas to cause crashes)
        # Shape: (NUM_ENVS, 3)
        actions = envs.action_space.sample()
        
        # Bias: High gas (0.5 to 1.0) for everyone
        gas_bias = np.random.uniform(0.5, 1.0, size=NUM_ENVS)
        actions[:, 1] = gas_bias
        
        # 2. Step all environments at once
        next_obs, rewards, terminated, truncated, _ = envs.step(actions)
        dones = np.logical_or(terminated, truncated)
        
        # 3. Store data
        for i in range(NUM_ENVS):
            # Append current step data
            obs_buffers[i].append(obs[i])
            action_buffers[i].append(actions[i])
            reward_buffers[i].append(rewards[i])
            done_buffers[i].append(dones[i])
            
            # 4. Check if Episode Finished
            # Note: VectorEnv auto-resets. If done[i] is True, 'next_obs[i]' is actually
            # the first frame of the NEW episode. The final frame of the OLD episode 
            # is lost unless we look at info, but for VAE training, missing 1 frame is fine.
            if dones[i] or len(obs_buffers[i]) >= MAX_STEPS:
                # Only save if we haven't hit the limit yet
                if saved_count < TOTAL_EPISODES:
                    np.savez_compressed(
                        os.path.join(DATA_DIR, f"bad_rollout_{saved_count}.npz"),
                        obs=np.array(obs_buffers[i]),
                        actions=np.array(action_buffers[i]),
                        rewards=np.array(reward_buffers[i]),
                        dones=np.array(done_buffers[i])
                    )
                    saved_count += 1
                    pbar.update(1)
                
                # Clear buffer for the next episode (which has already started)
                obs_buffers[i] = []
                action_buffers[i] = []
                reward_buffers[i] = []
                done_buffers[i] = []
        
        obs = next_obs

    envs.close()
    print("Parallel collection complete.")

if __name__ == "__main__":
    collect_data()