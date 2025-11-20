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
    # Using data_collection_random.py via the generic collect_data wrapper won't work easily 
    # because collect_data.py was refactored to BE the random collector.
    # So we call collect_data.py directly.
    run_command("python collect_data.py --episodes 2 --workers 1")
    
    # 2. Train VAE (1 Epoch, small batch)
    # Ensure we have enough data (2 episodes might be small for batch 128, 
    # but VAE loads chunks. Let's see if batch size < samples works. 
    # Episode len ~1000 steps. 2 eps = 2000 steps. Batch 32 is safe.
    run_command("python run_vae_training.py --epochs 1 --batch_size 32")
    
    # 3. Process Data
    run_command("python process_data.py")
    
    # 4. Train RNN (1 Epoch)
    run_command("python train_rnn.py --epochs 1 --batch_size 32")
    
    # 5. Train Dream (1 Generation, small pop)
    run_command("python train_dream.py --generations 1 --pop_size 4 --dream_length 50")

    print("\n" + "="*60)
    print("[SUCCESS] Pipeline integration test passed!")
    print("="*60)

if __name__ == "__main__":
    main()