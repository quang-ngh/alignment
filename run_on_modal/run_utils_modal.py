import modal
import sys
import os

# Add src directory to path so we can import utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Create Modal app
app = modal.App("run-utils")

# Create volume for storage
volume = modal.Volume.from_name("fifa-data", create_if_missing=True)

# Define image with required packages
image = modal.Image.debian_slim(python_version="3.11").pip_install([
    "datasets",
    "pillow",
    "pyarrow"
]).add_local_dir("../src", "/root/src")

@app.function(
    image=image,
    volumes={"/data": volume},
    gpu=None,
    timeout=3600
)
def run_utils():
    # Import shared utilities
    from utils import process_fifa_dataset
    
    # Run the dataset processing using shared code
    process_fifa_dataset(
        data_path="/data/fifa/FiFA-1k",  # Use local volume data
        save_dir="/data/1k_data",
        n_samples=1000,
        split_ratio=0.5
    )
    
    print("Done!")
    volume.commit()

@app.local_entrypoint()
def main():
    run_utils.remote()
