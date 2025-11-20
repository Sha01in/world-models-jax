import os
import sys
import time

def run_command(command):
    print(f"\n[TEST] Running: {command}")
    start_time = time.time()
    ret = os.system(command)
    if ret != 0:
        print(f"[FAIL] Command failed: {command}")
        sys.exit(1)
    print(f"[PASS] Completed in {time.time() - start_time:.2f}s")

def main():
    print("="*60)
    print("Running Full Pipeline Integration Test (Tiny)")
    print("="*60)
    
    # Ensure python path is set
    os.environ["PYTHONPATH"] = os.getcwd()

    # 1. Collect Data (2 Episodes)
    run_command("python collect_data.py --episodes 2 --workers 1")
    
    # 2. Train VAE (1 Epoch, small batch)
    run_command("python run_vae_training.py --epochs 1 --batch_size 32")
    
    # 3. Process Data
    run_command("python process_data.py")
    
    # 4. Train RNN (1 Epoch)
    run_command("python train_rnn.py --epochs 1 --batch_size 32")
    
    # 5. Train Dream (1 Generation, small pop)
    run_command("python train_dream.py --generations 1 --pop_size 4 --dream_length 50")

    # 6. Visualize Agent (1 Episode)
    run_command("python test_agent.py --episodes 1")

    print("\n" + "="*60)
    print("[SUCCESS] Pipeline integration test passed!")
    print("Check 'videos/' for the agent performance video.")
    print("="*60)

if __name__ == "__main__":
    main()