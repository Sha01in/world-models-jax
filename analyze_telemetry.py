import numpy as np
import matplotlib.pyplot as plt

def analyze_episode(ep_num):
    try:
        data = np.load(f"telemetry/ep_{ep_num}.npz")
    except FileNotFoundError:
        print(f"Episode {ep_num} not found.")
        return

    actions = data['actions'] # (T, 3)
    rewards = data['rewards'] # (T,)
    z = data['z']             # (T, 32)
    surprise = data['surprise'] # (T,)
    r_pred = data['r_pred']     # (T,)

    print(f"\n--- Analysis of Episode {ep_num} ---")
    print(f"Total Score: {np.sum(rewards):.2f}")
    print(f"Avg Surprise: {np.mean(surprise):.4f}")
    
    slice_len = 200
    print(f"\nFrames 0-{slice_len} Stats:")
    print(f"Avg Steer:  {np.mean(actions[:slice_len, 0]):.3f}")
    print(f"Avg Reward: {np.mean(rewards[:slice_len]):.3f}")
    print(f"Avg Pred R: {np.mean(r_pred[:slice_len]):.3f}")
    
    # Check for Hallucination
    # Find frames where actual reward is negative but predicted reward is positive
    hallucinations = 0
    delusion_diffs = []
    
    for i in range(len(rewards)):
        if rewards[i] < -0.05: # Actual crash/grass
            diff = r_pred[i] - rewards[i]
            delusion_diffs.append(diff)
            if r_pred[i] > 0.0:
                hallucinations += 1
            
    print(f"Hallucination Frames (Actual < -0.05, Pred > 0): {hallucinations} / {len(rewards)}")
    if len(delusion_diffs) > 0:
        print(f"Avg Reward Delusion (Pred - Actual) when Crashing: {np.mean(delusion_diffs):.4f}")
    else:
        print("No crashes detected in this episode.")
    
    # Check for Helplessness
    # Find frames where predicted reward is negative, but agent fails to steer?
    # Or just check if pred reward goes negative at all.
    neg_pred = np.sum(r_pred < 0)
    print(f"Negative Prediction Frames (Pred < 0): {neg_pred} / {len(rewards)}")

if __name__ == "__main__":
    for i in range(1, 6):
        analyze_episode(i)

