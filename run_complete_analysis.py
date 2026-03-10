"""
Complete pipeline to run the entire analysis.
Usage:
  python run_complete_analysis.py              # fetch + parse + train
  python run_complete_analysis.py --skip-fetch # parse + train (use existing JSON files)
"""
import subprocess
import sys
import argparse


def run_script(script_name):
    try:
        print(f"\n{'='*50}")
        print(f"Running {script_name}")
        print(f"{'='*50}")

        result = subprocess.run([sys.executable, script_name],
                                capture_output=False, check=True)
        print(f"{script_name} completed successfully")

    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}: {e}")
        return False

    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-fetch', action='store_true',
                        help='Skip SSB API data fetch (use existing JSON files)')
    args = parser.parse_args()

    scripts = []
    if not args.skip_fetch:
        scripts.append('fetch_ssb_data.py')
    scripts.extend(['data_parser.py', 'train_model.py'])

    print("Starting complete property market analysis...")

    for script in scripts:
        success = run_script(script)
        if not success:
            print(f"Pipeline failed at {script}")
            return

    print("\nComplete analysis finished successfully!")
    print("\nFiles created:")
    print("- processed/processed_data.csv")
    print("- confusion_matrix.png")
    print("- feature_importance.png")


if __name__ == "__main__":
    main()
