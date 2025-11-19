import numpy as np
import glob

files = glob.glob("data/rollouts/*.npz")
print(f"Checking {len(files)} files...")

gas_values = []
scores = []

for f in files[:20]: # Check first 20
    with np.load(f) as data:
        actions = data['actions']
        rewards = data['rewards']
        
        # Actions are [Steer, Gas, Brake]
        gas = actions[:, 1] 
        avg_gas = np.mean(gas)
        total_score = np.sum(rewards)
        
        gas_values.append(avg_gas)
        scores.append(total_score)

print(f"Average Gas Input in Data: {np.mean(gas_values):.4f}")
print(f"Average Episode Score:     {np.mean(scores):.4f}")

if np.mean(gas_values) < 0.1:
    print("CRITICAL: Your training data has almost zero gas. The teacher was broken.")
elif np.mean(scores) < 0:
    print("CRITICAL: The teacher drove, but got negative scores. The agent learned to do nothing to avoid pain.")
else:
    print("Data looks healthy.")