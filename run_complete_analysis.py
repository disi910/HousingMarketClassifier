"""
Complete pipeline to run the entire analysis
"""
import subprocess
import sys

def run_script(script_name):
    """Run a Python script and handle errors"""
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
    """Run complete analysis pipeline"""
    scripts = [
        'data_parser.py',
        'train_model.py'
    ]
    
    print("Starting complete property market analysis...")
    
    for script in scripts:
        success = run_script(script)
        if not success:
            print(f"Pipeline failed at {script}")
            return
    
    print("\nComplete analysis finished successfully!")
    print("\nFiles created:")
    print("- processed/processed_data.csv")
    print("- Feature importance plots")
    print("- Confusion matrix")
    print("- Backtesting results")

if __name__ == "__main__":
    main()