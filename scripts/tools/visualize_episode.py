import numpy as np
import matplotlib.pyplot as plt
import cv2
import argparse

import os

# Default Settings
DEFAULT_EPISODE_NUM = 4

def get_paths(episode_num):
    return {
        "video": f"videos/final_agent_ep{episode_num}.mp4",
        "telemetry": f"telemetry/ep_{episode_num}.npz",
        "output": f"diagnostics/debug_grid_ep{episode_num}.png"
    }

# Grid Settings
ROWS = 5
COLS = 4
TOTAL_FRAMES_TO_SHOW = ROWS * COLS

def create_annotated_grid(episode_num):
    paths = get_paths(episode_num)
    video_path = paths["video"]
    telemetry_path = paths["telemetry"]
    output_path = paths["output"]
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. Load Data
    try:
        data = np.load(telemetry_path)
        rewards = data['rewards']
        r_pred = data['r_pred']
        surprise = data['surprise']
        actions = data['actions']
    except Exception as e:
        print(f"Failed to load telemetry: {e}")
        return

    # 2. Open Video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return
    
    total_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Processing Episode {episode_num} | Video Frames: {total_frames_video} | Data Points: {len(rewards)}")

    # 3. Select Key Frames
    # We want to see a mix of:
    # - High Surprise moments (RNN confusion)
    # - Large discrepancies between Reward and Predicted Reward (Delusion)
    # - Regular intervals
    
    # Find indices of top 5 surprise moments
    top_surprise_idx = np.argsort(surprise)[-5:]
    
    # Find indices of top 5 "Delusion" moments (Crash but predicted safe)
    # Delusion = (Pred > 0) AND (Actual < 0) magnitude
    delusion_score = np.zeros_like(rewards)
    mask = (r_pred > 0) & (rewards < -0.05)
    delusion_score[mask] = r_pred[mask] - rewards[mask]
    top_delusion_idx = np.argsort(delusion_score)[-5:]
    
    # Regular intervals for the rest
    remaining_slots = TOTAL_FRAMES_TO_SHOW - len(top_surprise_idx) - len(top_delusion_idx)
    step = max(1, len(rewards) // remaining_slots)
    regular_idx = np.arange(0, len(rewards), step)[:remaining_slots]
    
    # Combine and sort unique indices
    indices_to_plot = np.unique(np.concatenate([regular_idx, top_surprise_idx, top_delusion_idx]))
    indices_to_plot.sort()
    indices_to_plot = indices_to_plot[:TOTAL_FRAMES_TO_SHOW] # Cap at limit

    # 4. Create Plot
    fig, axes = plt.subplots(ROWS, COLS, figsize=(20, 16))
    axes = axes.flatten()
    
    for i, ax in enumerate(axes):
        if i >= len(indices_to_plot):
            ax.axis('off')
            continue
            
        idx = indices_to_plot[i]
        
        # Get Frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
            
        # Frame is BGR, convert to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Display
        ax.imshow(frame)
        ax.axis('off')
        
        # Annotate
        t_reward = rewards[idx]
        t_r_pred = r_pred[idx] if idx < len(r_pred) else 0.0
        t_surprise = surprise[idx] if idx < len(surprise) else 0.0
        t_action = actions[idx] if idx < len(actions) else [0,0,0]
        
        # Color code border based on status
        # Red = Crash, Green = Good, Yellow = Delusion
        status_color = 'black'
        if t_reward < -0.05:
            status_color = "red"
            if t_r_pred > 0:
                status_color = "orange"  # Delusion
        elif t_reward > 0.1:
            status_color = 'green'
            
        # Title
        title = f"T={idx}\nAct:[{t_action[0]:.2f}, {t_action[1]:.2f}, {t_action[2]:.2f}]\n"
        title += f"R: {t_reward:.2f} | Pred: {t_r_pred:.2f}\nSurprise: {t_surprise:.4f}"
        
        ax.set_title(title, fontsize=9, color=status_color, fontweight='bold')
        
        # Add border
        plt.setp(ax.spines.values(), color=status_color, linewidth=3)

    plt.tight_layout()
    plt.suptitle(f"Episode {episode_num} Diagnostics: Real | Recon | Dream", fontsize=16, y=1.02)
    plt.savefig(output_path, bbox_inches='tight')
    cap.release()
    print(f"Saved diagnostic grid to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create Diagnostic Grid from Episode")
    parser.add_argument("--episode", type=int, default=DEFAULT_EPISODE_NUM, help="Episode number to visualize")
    args = parser.parse_args()
    create_annotated_grid(args.episode)
