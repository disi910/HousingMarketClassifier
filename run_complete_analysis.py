"""
Complete pipeline to run the entire analysis.
Usage:
  python run_complete_analysis.py              # fetch + parse + train
  python run_complete_analysis.py --skip-fetch # parse + train (use existing JSON files)
"""
import subprocess
import sys
import argparse
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent / "src"


def run_script(script_name):
    script_path = SRC_DIR / script_name
    try:
        print(f"\n{'='*50}")
        print(f"Running {script_name}")
        print(f"{'='*50}")

        subprocess.run([sys.executable, str(script_path)],
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
        if not run_script(script):
            print(f"Pipeline failed at {script}")
            return

    print("\nComplete analysis finished successfully!")
    print("\nFiles created:")
    print("- output/processed_data.csv")
    print("- output/confusion_matrix.png")
    print("- output/feature_importance.png")


if __name__ == "__main__":
    main()
