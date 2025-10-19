import modal

# Create Modal app
app = modal.App("download-fifa")

# Create volume for storage
volume = modal.Volume.from_name("fifa-data", create_if_missing=True)

# Define image with required packages
image = modal.Image.debian_slim(python_version="3.11").pip_install([
    "huggingface_hub",
    "datasets"
])

@app.function(
    image=image,
    volumes={"/data": volume}
)
def download_fifa():
    from huggingface_hub import snapshot_download
    
    # Download the dataset
    snapshot_download(
        repo_id="Dragonjinny/FiFA-pickapic-v2",
        local_dir="/data/fifa/",
        repo_type="dataset",
        allow_patterns=["FiFA-1k/*"]
    )
    
    print("Download complete!")
    
    # Save to volume
    volume.commit()

@app.local_entrypoint()
def main():
    download_fifa.remote()
