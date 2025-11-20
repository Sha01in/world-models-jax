import gymnasium as gym
import numpy as np
import os
import cv2
from tqdm import tqdm

# Settings
NUM_EPISODES = 300     # Increased slightly to get more "good" examples
MAX_STEPS = 1000       
DATA_DIR = "data/rollouts"
IMG_SIZE = 64

def get_brownian_action(action, noise_scale=0.1):
    """
    Adds small noise to the previous action instead of sampling a new one completely.
    This creates smooth, continuous driving behavior.
    """
    # Action: [steering, gas, brake]
    # Steering: -1 to 1
    # Gas: 0 to 1
    # Brake: 0 to 1
    
    noise = np.random.randn(3) * noise_scale
    action += noise
    
    # Clip to valid range
    action[0] = np.clip(action[0], -1.0, 1.0) # Steering
    action[1] = np.clip(action[1], 0.0, 1.0)  # Gas
    action[2] = np.clip(action[2], 0.0, 1.0)  # Brake
    
    # Bias towards moving forward (keep gas > 0, brake low)
    # If gas is too low, boost it
    if action[1] < 0.2:
        action[1] += 0.1
        
    return action

def collect_data():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    env = gym.make("CarRacing-v3", render_mode="rgb_array")
    
    print(f"Collecting {NUM_EPISODES} episodes using Brownian Policy...")
    
    positive_reward_frames = 0
    total_frames = 0
    
    for episode in tqdm(range(NUM_EPISODES)):
        obs, _ = env.reset()
        
        # Initialize random action
        action = env.action_space.sample()
        action[1] = 1.0 # Start with full gas
        action[2] = 0.0 # No brake
        
        obs_sequence = []
        action_sequence = []
        reward_sequence = []
        done_sequence = []
        
        for t in range(MAX_STEPS):
            obs_small = cv2.resize(obs, (IMG_SIZE, IMG_SIZE))
            obs_sequence.append(obs_small)
            
            # Update action smoothly
            action = get_brownian_action(action)
            action_sequence.append(action)
            
            obs, reward, terminated, truncated, _ = env.step(action)
            
            if reward > 0:
                positive_reward_frames += 1
            total_frames += 1
            
            reward_sequence.append(reward)
            done = terminated or truncated
            done_sequence.append(done)
            
            if done:
                break
                
        np.savez_compressed(
            os.path.join(DATA_DIR, f"rollout_{episode}.npz"),
            obs=np.array(obs_sequence),
            actions=np.array(action_sequence),
            rewards=np.array(reward_sequence),
            dones=np.array(done_sequence)
        )
        
    env.close()
    print(f"Data collection complete.")
    print(f"Quality Check: {positive_reward_frames / total_frames * 100:.2f}% of frames had positive reward.")
    print("(Target is > 5-10% to give the RNN something to learn)")

if __name__ == "__main__":
    collect_data() 