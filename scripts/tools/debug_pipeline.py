import numpy as np
import glob
import jax
import jax.numpy as jnp
import equinox as eqx
from src.rnn import MDNRNN

# Settings
DATA_DIR = "data/rollouts/*.npz"  # Raw data
SERIES_DIR = "data/series/*.npz"  # Processed data for RNN
RNN_PATH = "checkpoints/rnn.eqx"
LATENT_DIM = 32
HIDDEN_SIZE = 256
ACTION_DIM = 3

def check_data_quality():
    print("=== 1. CHECKING DATA QUALITY ===")
    files = glob.glob(DATA_DIR)
    if not files:
        print("No data found!")
        return
    
    total_reward = []
    positive_frames = 0
    total_frames = 0
    
    # Check first 50 files to save time
    for f in files[:50]:
        with np.load(f) as data:
            if 'rewards' not in data:
                print(f"File {f} missing 'rewards' key! Re-run Step 01.")
                return
            
            r = data['rewards']
            total_reward.append(np.sum(r))
            positive_frames += np.sum(r > 0)
            total_frames += len(r)
            
    avg_score = np.mean(total_reward)
    max_score = np.max(total_reward)
    pos_ratio = positive_frames / total_frames
    
    print(f"Analyzed {min(len(files), 50)} episodes.")
    print(f"Average Episode Score: {avg_score:.2f}")
    print(f"Best Episode Score:    {max_score:.2f}")
    print(f"Frames with +Reward:   {pos_ratio*100:.2f}%")
    
    if max_score < 50:
        print("\n[CRITICAL WARNING] Your training data is too poor.")
        print("The random policy never learned to drive, so the RNN has no 'good memories' to dream about.")
        print("The RNN likely predicts negative reward for everything.")
    else:
        print("\n[OK] Data looks sufficient.")

def check_rnn_predictions():
    print("\n=== 2. CHECKING RNN PREDICTIONS ===")
    # Load RNN
    key = jax.random.PRNGKey(0)
    rnn = MDNRNN(latent_dim=LATENT_DIM, action_dim=ACTION_DIM, 
                 hidden_size=HIDDEN_SIZE, key=key)
    try:
        rnn = eqx.tree_deserialise_leaves(RNN_PATH, rnn)
        print("RNN loaded successfully.")
    except Exception as e:
        print(f"Could not load RNN: {e}")
        return

    # Load one processed series
    files = glob.glob(SERIES_DIR)
    if not files:
        print("No series data found.")
        return
    
    with np.load(files[0]) as data:
        mu = data['mu']         # (1000, 32)
        actions = data['actions'] # (1000, 3)
        rewards = data['rewards'] # (1000,)
        dones = data['dones']     # (1000,)

    # Run RNN on this sequence (Open Loop)
    # We feed it real data and see what it predicts
    h = jnp.zeros(HIDDEN_SIZE)
    c = jnp.zeros(HIDDEN_SIZE)
    
    pred_rewards = []
    pred_dones = []
    
    print("Running inference on 100 steps of real data...")
    for t in range(100):
        z_t = jnp.array(mu[t])
        a_t = jnp.array(actions[t])
        
        rnn_input = jnp.concatenate([z_t, a_t], axis=0)
        (log_pi, mu_pred, log_sigma, r_pred, d_logit), (h, c) = rnn(rnn_input, (h, c))
        
        pred_rewards.append(r_pred.item())
        pred_dones.append(jax.nn.sigmoid(d_logit).item())

    # Compare
    print(f"{'Step':<5} | {'Real R':<10} | {'Pred R':<10} | {'Real Done':<10} | {'Pred Done Prob':<15}")
    print("-" * 60)
    for t in range(0, 100, 10): # Print every 10th step
        print(f"{t:<5} | {rewards[t+1]:<10.4f} | {pred_rewards[t]:<10.4f} | {str(dones[t+1]):<10} | {pred_dones[t]:<15.4f}")

    avg_pred_r = np.mean(pred_rewards)
    avg_pred_d = np.mean(pred_dones)
    
    print("\nDiagnosis:")
    if avg_pred_d > 0.5:
        print("[PROBLEM] The RNN predicts 'Death' (Done) constantly.")
        print("Fix: Bias the Done Head output layer to start negative.")
    elif avg_pred_r < 0.0:
        print("[PROBLEM] The RNN is depressed (Predicts negative reward constantly).")
        print("Fix: Better data collection or weighting positive rewards higher.")
    else:
        print("[OK] Predictions look reasonable. The issue might be in the Dream loop logic.")

if __name__ == "__main__":
    check_data_quality()
    check_rnn_predictions()