import modal
import os
import json
from pathlib import Path
import subprocess
import sys

# Configuration
MODEL_NAME = "sd15_corrupted_dpo_1000"

# Create Modal image with minimal dependencies
eval_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .run_commands([
        "apt-get update",
        "apt-get install -y git build-essential cmake",
        "pip install --upgrade pip setuptools wheel"
    ])
    .pip_install([
        "torch==2.3.1", "torchvision==0.18.1", "torchaudio==2.3.1",
        "numpy", "packaging", "ninja", "setuptools", "wheel",
        "pillow", "tqdm", "huggingface_hub", "gitpython", "pandas", "omegaconf",
        "open_clip_torch"
    ])
    .add_local_file("basic_evaluator.py", "/root/basic_evaluator.py")
    .add_local_dir("../datasets/eval_prompts", "/root/datasets/eval_prompts")
)

app = modal.App("eval-basic")
volume = modal.Volume.from_name("fifa-data", create_if_missing=False)

@app.function(
    image=eval_image,
    volumes={"/data": volume},
    gpu="A100",
    timeout=3600,
    secrets=[
        modal.Secret.from_dict({"WANDB_API_KEY": "0c2416138832b33d254a444d26384582d70420e4"}),
        modal.Secret.from_name("huggingface-token")
    ]
)
def evaluate_images_basic_only():
    import torch
    import os
    from pathlib import Path
    import json
    import subprocess
    import sys
    
    # Set environment variables
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    
    print(f"Starting basic evaluation for: {MODEL_NAME}")
    
    # Create output directories
    images_dir = Path("/data/evaluation_output_1k/generated_images")
    results_dir = Path("/data/evaluation_output_1k/evaluation_results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if images exist
    if not images_dir.exists():
        print(f"Error: Images directory {images_dir} does not exist!")
        return
    
    # Find image categories
    categories = [d.name for d in images_dir.iterdir() if d.is_dir()]
    print(f"Found image categories: {categories}")
    
    print("=" * 60)
    print("EVALUATING IMAGES")
    print("=" * 60)
    
    # Import our basic evaluator
    from basic_evaluator import evaluate_images_basic
    
    # Process each category
    for category in categories:
        print(f"\nEvaluating {category}...")
        
        # Load prompts for this category
        prompt_file = f"/root/datasets/eval_prompts/hpsv2_{category}.json"
        if not os.path.exists(prompt_file):
            print(f"Warning: No prompt file found for {category}")
            continue
            
        with open(prompt_file, 'r') as f:
            prompts = json.load(f)
        
        # Get image files
        category_dir = images_dir / category
        image_files = sorted([f for f in category_dir.glob("*.jpg")])
        
        if len(image_files) != len(prompts):
            print(f"Warning: Mismatch between images ({len(image_files)}) and prompts ({len(prompts)}) for {category}")
            min_len = min(len(image_files), len(prompts))
            image_files = image_files[:min_len]
            prompts = prompts[:min_len]
        
        print(f"Evaluating {len(image_files)} images for {category}")
        
        # Run evaluation
        output_file = results_dir / f"{category}_results.csv"
        try:
            results = evaluate_images_basic(
                images=[str(f) for f in image_files],
                prompts=prompts,
                output_file=str(output_file),
                device="cuda"
            )
            print(f"Successfully evaluated {category}")
        except Exception as e:
            print(f"Error evaluating {category}: {e}")
            continue
    
    print("=" * 60)
    print("EVALUATION COMPLETE!")
    print("=" * 60)
    print(f"Generated images: {images_dir}")
    print(f"Evaluation results: {results_dir}")
    print("\nGenerated files:")
    for f in results_dir.glob("*.csv"):
        print(f"  - {f.name}")

@app.local_entrypoint()
def main():
    print("Starting basic evaluation...")
    print(f"Model: {MODEL_NAME}")
    
    evaluate_images_basic_only.remote()
    
    print("\nTo download results, run:")
    print("modal volume get fifa-data /data/evaluation_output_1k/evaluation_results ./local_eval_results")
