#!/usr/bin/env python3
"""
Step 1: Generate images only from UNet checkpoint.

Usage:
    modal run generate_only.py
"""

import modal
import os
import json

# Configuration
CHECKPOINT_PATH = "/data/training_runs/sd15_corrupted_dpo/checkpoint-1000"

# Modal image setup
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .run_commands([
        "apt-get update",
        "apt-get install -y git build-essential cmake",
        "pip install --upgrade pip setuptools wheel"
    ])
    .pip_install([
        "torch==2.3.1", "torchvision==0.18.1", "torchaudio==2.3.1",
        "numpy", "packaging", "ninja", "setuptools", "wheel",
        "accelerate", "diffusers", "transformers", "datasets",
        "pillow", "tqdm", "xformers", "huggingface_hub", "gitpython", "pandas", "omegaconf"
    ])
    .pip_install("flash-attn==2.6.2", extra_options="--no-build-isolation")
    # Add local files
    .add_local_file("generate_with_unet.py", "/root/generate_with_unet.py")
    .add_local_dir("../datasets/eval_prompts", "/root/datasets/eval_prompts")
)

app = modal.App("generate-only")
volume = modal.Volume.from_name("fifa-data", create_if_missing=False)

@app.function(
    image=image,
    volumes={"/data": volume},
    gpu="A100",
    timeout=3600
)
def generate_images_only():
    """Generate images from UNet checkpoint."""
    import subprocess
    import sys
    import os
    from pathlib import Path
    
    sys.path.insert(0, "/root")
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    
    print(f"Starting image generation...")
    print(f"Using checkpoint: {CHECKPOINT_PATH}")
    
    # Create output directory
    output_dir = Path("/data/evaluation_output/generated_images")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get available prompt files
    prompt_dir = "/root/datasets/eval_prompts"
    prompt_files = [f for f in os.listdir(prompt_dir) if f.endswith('.json')]
    print(f"Available prompt files: {prompt_files}")
    
    # Generate images for each category
    for prompt_file in prompt_files:
        category = prompt_file.replace('.json', '').replace('hpsv2_', '')
        prompt_path = os.path.join(prompt_dir, prompt_file)
        category_images_dir = output_dir / category
        category_images_dir.mkdir(exist_ok=True)
        
        print(f"\nGenerating images for {category}...")
        
        # Use the shared generate script
        cmd = [
            "python", "/root/generate_with_unet.py",
            "--checkpoint_path", CHECKPOINT_PATH,
            "--prompt_file", prompt_path,
            "--output_dir", str(category_images_dir)
        ]
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"Generated images for {category}")
            print(f"Output: {result.stdout[-200:]}")  # Show last 200 chars
        except subprocess.CalledProcessError as e:
            print(f"Error generating images for {category}: {e}")
            if e.stderr:
                print(f"Error details: {e.stderr[-500:]}")
            continue
    
    print(f"\nImage generation completed!")
    print(f"Images saved to: {output_dir}")
    
    # List generated files
    print("\nGenerated files:")
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            if file.endswith('.jpg'):
                file_path = os.path.join(root, file)
                print(f"  - {file_path}")

@app.local_entrypoint()
def main():
    """Run image generation on Modal."""
    print("Starting image generation...")
    print(f"Using checkpoint: {CHECKPOINT_PATH}")
    
    generate_images_only.remote()
    
    print("\nTo download generated images, run:")
    print("modal volume get fifa-data /data/evaluation_output/generated_images ./local_generated_images")
