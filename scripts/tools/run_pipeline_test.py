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
    
    # Dependency Check
    try:
        import gymnasium
        import jax
    except ImportError as e:
        print(f"\n[ERROR] Missing dependency: {e.name}")
        print("You seem to be running with a Python interpreter that doesn't have the dependencies installed.")
        print(f"Current Python: {sys.executable}")
        print("\nFix this by running with 'uv run':")
        print("    uv run python scripts/tools/run_pipeline_test.py")
        sys.exit(1)

    # Ensure python path is set
    os.environ["PYTHONPATH"] = os.getcwd()

    # 1. Collect Data (2 Episodes)
    run_command(f"{sys.executable} collect_data.py --episodes 2 --workers 1")
    
    # 2. Train VAE (1 Epoch, small batch)
    run_command(f"{sys.executable} run_vae_training.py --epochs 1 --batch_size 32")
    
    # 3. Process Data
    run_command(f"{sys.executable} process_data.py")
    
    # 4. Train RNN (1 Epoch)
    run_command(f"{sys.executable} train_rnn.py --epochs 1 --batch_size 32")
    
    # 5. Train Dream (1 Generation, small pop)
    run_command(f"{sys.executable} train_dream.py --generations 1 --pop_size 4 --dream_length 50")

    # 6. Visualize Agent (1 Episode)
    run_command(f"{sys.executable} test_agent.py --episodes 1")

    # 7. Generate Debug Grid
    run_command(f"{sys.executable} scripts/tools/visualize_episode.py --episode 1")

    print("\n" + "="*60)
    print("[SUCCESS] Pipeline integration test passed!")
    print("Check 'videos/' for the agent performance video.")
    print("Check 'diagnostics/' for debug filmstrips and grids.")
    print("="*60)

if __name__ == "__main__":
    main()