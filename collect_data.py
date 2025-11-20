import sys
import os
import subprocess

def print_menu():
    print("\n=== World Models Data Collection ===")
    print("-- Phase 1: Initial Bootstrap --")
    print("1. Random (Brownian Noise)     [Baseline]")
    print("2. Smart Random (Score > 5)    [Better Baseline]")
    print("3. Heuristic (Good Data)       [Teacher/Expert]")
    print("4. Failures (Crash Data)       [Negative Examples]")
    
    print("\n-- Phase 2: Robustness --")
    print("5. Aggressive (Speed Limits)   [Friction Limits]")
    print("6. Recovery (Heuristic + Noise)[Recovering from Errors]")
    
    print("\n-- Phase 3: Active Learning --")
    print("7. Iterative Failures          [Previous Agent Failures]")
    print("8. On-Policy                   [Current Agent Failures]")
    
    print("\n0. Exit")
    print("====================================")

def run_script(script_name):
    script_path = os.path.join("scripts", "data_collection", script_name)
    if not os.path.exists(script_path):
        print(f"Error: Script not found at {script_path}")
        return
    
    print(f"\n>>> Launching {script_name}...\n")
    try:
        # Run the script as a subprocess
        # Ensure the root directory is in PYTHONPATH so imports like 'from src...' work
        env = os.environ.copy()
        env["PYTHONPATH"] = os.getcwd() + os.pathsep + env.get("PYTHONPATH", "")
        
        subprocess.run([sys.executable, script_path], check=True, env=env)
    except subprocess.CalledProcessError as e:
        print(f"\nError: Data collection failed with code {e.returncode}")
    except KeyboardInterrupt:
        print("\nData collection interrupted.")

def main():
    while True:
        print_menu()
        choice = input("Select Data Mode (0-8): ").strip()
        
        if choice == '0':
            print("Exiting.")
            break
        elif choice == '1':
            run_script("data_collection_random.py")
        elif choice == '2':
            run_script("data_collection_smart.py")
        elif choice == '3':
            run_script("data_collection_heuristic.py")
        elif choice == '4':
            run_script("data_collection_failures.py")
        elif choice == '5':
            run_script("data_collection_aggressive.py")
        elif choice == '6':
            run_script("data_collection_recovery.py")
        elif choice == '7':
            run_script("data_collection_iterative.py")
        elif choice == '8':
            run_script("data_collection_on_policy.py")
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
