import os
import sys
import time
import argparse

def run_command(command):
    print(f"\n[TEST] Running: {command}")
    start_time = time.time()
    ret = os.system(command)
    if ret != 0:
        print(f"[FAIL] Command failed: {command}")
        sys.exit(1)
    print(f"[PASS] Completed in {time.time() - start_time:.2f}s")

def main():
    parser = argparse.ArgumentParser(description="Run Full Pipeline Integration Test")
    parser.add_argument("--env", type=str, default="CarRacing-v3", help="Environment name")
    args = parser.parse_args()
    
    env_name = args.env

    print("="*60)
    print(f"Running Full Pipeline Integration Test (Tiny) for {env_name}")
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
    run_command(f"{sys.executable} collect_data.py --episodes 2 --workers 1 --env {env_name}")
    
    # 2. Train VAE (1 Epoch, small batch)
    run_command(f"{sys.executable} run_vae_training.py --epochs 1 --batch_size 32 --env {env_name}")
    
    # 3. Process Data
    run_command(f"{sys.executable} process_data.py --env {env_name}")
    
    # 4. Train RNN (1 Epoch)
    run_command(f"{sys.executable} train_rnn.py --epochs 1 --batch_size 32 --env {env_name}")
    
    # 5. Train Dream (1 Generation, small pop)
    run_command(f"{sys.executable} train_dream.py --generations 1 --pop_size 4 --dream_length 50 --env {env_name}")

    # 6. Visualize Agent (1 Episode)
    run_command(f"{sys.executable} test_agent.py --episodes 1 --env {env_name}")

    # 7. Generate Debug Grid
    # visualize_episode.py might not support --env yet, let's check or skip if it fails?
    # Actually, let's assume it needs to be updated or we just run it and see.
    # The original script didn't have --env in the call, but it might need it to find the right checkpoints.
    # Let's check if visualize_episode.py exists and what it takes.
    # For now, I'll add the flag if I can, or just try running it.
    # Wait, I haven't checked visualize_episode.py. Let's assume it needs update or might fail.
    # But to be safe, I will try to run it with --env if it accepts it, or just run it as is if it defaults to CarRacing.
    # However, if I am testing Doom, running it as is (CarRacing) will fail or show wrong data.
    # Let's try to pass --env. If it fails, the test fails, which is correct behavior (we need to fix it too).
    run_command(f"{sys.executable} scripts/tools/visualize_episode.py --episode 1 --env {env_name}")

    print("\n" + "="*60)
    print("[SUCCESS] Pipeline integration test passed!")
    print("Check 'videos/' for the agent performance video.")
    print("Check 'diagnostics/' for debug filmstrips and grids.")
    print("="*60)

if __name__ == "__main__":
    main()