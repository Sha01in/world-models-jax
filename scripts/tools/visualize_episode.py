import numpy as np
import matplotlib.pyplot as plt
import cv2

# Settings
EPISODE_NUM = 4
VIDEO_PATH = f"videos/final_agent_ep{EPISODE_NUM}.mp4"
TELEMETRY_PATH = f"telemetry/ep_{EPISODE_NUM}.npz"
OUTPUT_PATH = f"debug_grid_ep{EPISODE_NUM}.png"

# Grid Settings
ROWS = 5
COLS = 4
TOTAL_FRAMES_TO_SHOW = ROWS * COLS

def create_annotated_grid():
    # 1. Load Data
    try:
        data = np.load(TELEMETRY_PATH)
        rewards = data['rewards']
        r_pred = data['r_pred']
        surprise = data['surprise']
        actions = data['actions']
    except Exception as e:
        print(f"Failed to load telemetry: {e}")
        return

    # 2. Open Video
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Failed to open video: {VIDEO_PATH}")
        return
    
    total_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Processing Episode {EPISODE_NUM} | Video Frames: {total_frames_video} | Data Points: {len(rewards)}")

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
    plt.suptitle(f"Episode {EPISODE_NUM} Diagnostics: Real | Recon | Dream", fontsize=16, y=1.02)
    plt.savefig(OUTPUT_PATH, bbox_inches='tight')
    cap.release()
    print(f"Saved diagnostic grid to {OUTPUT_PATH}")

if __name__ == "__main__":
    create_annotated_grid()
