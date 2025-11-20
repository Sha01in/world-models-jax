import sys
import os
import subprocess

def print_menu():
    print("\n=== World Models Data Collection ===")
    print("1. Random (Brownian Noise) [Initial Training]")
    print("2. Iterative Failures [Previous Agent Failures]")
    print("3. Recovery [Heuristic + Noise]")
    print("4. Aggressive [Heuristic Speed]")
    print("5. On-Policy [Current Agent Failures]")
    print("6. Heuristic [Good Data]")
    print("7. Smart Random [Score > 5]")
    print("8. Failures [Crash Data]")
    print("0. Exit")
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
        choice = input("Select Data Mode (0-5): ").strip()
        
        if choice == '0':
            print("Exiting.")
            break
        elif choice == '1':
            run_script("data_collection_random.py")
        elif choice == '2':
            run_script("data_collection_iterative.py")
        elif choice == '3':
            run_script("data_collection_recovery.py")
        elif choice == '4':
            run_script("data_collection_aggressive.py")
        elif choice == '5':
            run_script("data_collection_on_policy.py")
        elif choice == '6':
            run_script("data_collection_heuristic.py")
        elif choice == '7':
            run_script("data_collection_smart.py")
        elif choice == '8':
            run_script("data_collection_failures.py")
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()

