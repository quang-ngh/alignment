#!/usr/bin/env python3
"""
Step 2: Evaluate generated images using HPSv2, PickScore, and ImageReward.

Usage:
    modal run eval_only.py
"""

import modal
import os
import json
from pathlib import Path

# Configuration
MODEL_NAME = "sd15_corrupted_dpo_1000"

# Modal image setup
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
        "accelerate", "diffusers", "transformers", "datasets",
        "pillow", "tqdm", "xformers", "huggingface_hub", "gitpython", "pandas", "omegaconf"
    ])
    .pip_install("flash-attn==2.6.2", extra_options="--no-build-isolation")
    .run_commands([
        "pip install fire",
        "pip install git+https://github.com/yuvalkirstain/PickScore.git"
    ])
    .run_commands([
        "mkdir -p /root/checkpoints",
        "cd /root && python -c \"from huggingface_hub import snapshot_download; snapshot_download('laion/CLIP-ViT-H-14-laion2B-s32B-b79K', local_dir='/root/checkpoints/CLIP-ViT-H-14-laion2B-s32B-b79K')\"",
        "cd /root && python -c \"from huggingface_hub import snapshot_download; snapshot_download('yuvalkirstain/PickScore_v1', local_dir='/root/checkpoints/pickscore_v1')\""
    ])
    # Add local files
    .add_local_file("../evaluator.py", "/root/evaluator.py")
    .add_local_dir("../datasets/eval_prompts", "/root/datasets/eval_prompts")
    .add_local_dir("../HPSv2", "/root/HPSv2")
    .add_local_dir("../ImageReward", "/root/ImageReward")
)

app = modal.App("eval-only")
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
def evaluate_images_only():
    """Evaluate generated images using all metrics."""
    import subprocess
    import sys
    import os
    from pathlib import Path
    import torch
    from tqdm import tqdm
    
    sys.path.insert(0, "/root")
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    
    print(f"Starting evaluation for: {MODEL_NAME}")
    
    # Install local packages after files are available
    print("Installing HPSv2 and ImageReward...")
    import subprocess
    subprocess.run(["cd", "/root/HPSv2", "&&", "pip", "install", "-r", "requirements.txt", "&&", "pip", "install", "-e", "."], shell=True)
    subprocess.run(["cd", "/root/ImageReward", "&&", "python", "setup.py", "develop"], shell=True)
    print("Local packages installed successfully")
    
    # Create output directories
    images_dir = Path("/data/evaluation_output/generated_images")
    results_dir = Path("/data/evaluation_output/evaluation_results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if images exist
    if not images_dir.exists():
        print(f"Error: Images directory {images_dir} does not exist!")
        print("Please run generate_only.py first to generate images.")
        return
    
    # Get available image categories
    image_categories = [d for d in os.listdir(images_dir) if os.path.isdir(images_dir / d)]
    print(f"Found image categories: {image_categories}")
    
    if not image_categories:
        print("No image categories found!")
        return
    
    # Evaluate images
    print("\n" + "="*60)
    print("EVALUATING IMAGES")
    print("="*60)
    
    try:
        from evaluator import evaluate_all
        import pandas as pd
        
        all_results = {}
        
        for category in image_categories:
            image_dir = images_dir / category
            
            # Load corresponding prompts
            prompt_file = f"hpsv2_{category}.json"
            prompt_path = f"/root/datasets/eval_prompts/{prompt_file}"
            
            if not os.path.exists(prompt_path):
                print(f"Prompt file {prompt_file} not found, skipping {category}")
                continue
            
            # Load prompts
            with open(prompt_path, 'r') as f:
                prompts = json.load(f)
            
            # Count available images
            image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
            if len(image_files) == 0:
                print(f"No images found in {image_dir}, skipping {category}")
                continue
            
            # Limit to available images
            num_images = min(len(prompts), len(image_files))
            prompts = prompts[:num_images]
            
            print(f"Evaluating {num_images} image-prompt pairs for {category}...")
            
            # Run individual image evaluations
            detailed_results = []
            for i in tqdm(range(num_images), desc=f"Evaluating {category}"):
                try:
                    image_path = os.path.join(image_dir, f"image_{i}.jpg")
                    if not os.path.exists(image_path):
                        continue
                    
                    # Evaluate with all metrics
                    scores = evaluate_all(
                        image=image_path,
                        prompt=prompts[i],
                        hps_version="v2.1"
                    )
                    
                    detailed_results.append({
                        'image_id': i,
                        'prompt': prompts[i],
                        'image_path': image_path,
                        **scores
                    })
                    
                except Exception as e:
                    print(f"Error evaluating image {i}: {e}")
                    continue
            
            # Save detailed results
            if detailed_results:
                df = pd.DataFrame(detailed_results)
                detailed_output_path = results_dir / f"{MODEL_NAME}_{category}_detailed.csv"
                df.to_csv(detailed_output_path, index=False)
                
                # Calculate summary statistics
                summary = {
                    'category': category,
                    'num_samples': len(detailed_results),
                    'hpsv2_mean': df['hpsv2'].mean(),
                    'hpsv2_std': df['hpsv2'].std(),
                    'pickscore_mean': df['pickscore'].mean(),
                    'pickscore_std': df['pickscore'].std(),
                    'imagereward_mean': df['imagereward'].mean(),
                    'imagereward_std': df['imagereward'].std(),
                }
                
                all_results[category] = summary
                
                print(f"Category: {category}")
                print(f"  Samples: {summary['num_samples']}")
                print(f"  HPSv2: {summary['hpsv2_mean']:.4f} ± {summary['hpsv2_std']:.4f}")
                print(f"  PickScore: {summary['pickscore_mean']:.4f} ± {summary['pickscore_std']:.4f}")
                print(f"  ImageReward: {summary['imagereward_mean']:.4f} ± {summary['imagereward_std']:.4f}")
        
        # Save overall summary
        if all_results:
            summary_df = pd.DataFrame.from_dict(all_results, orient='index')
            summary_output_path = results_dir / f"{MODEL_NAME}_summary.csv"
            summary_df.to_csv(summary_output_path)
            print(f"\nOverall summary saved to {summary_output_path}")
        
        print("Evaluation completed successfully")
    except Exception as e:
        print(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION COMPLETE!")
    print("="*60)
    
    print(f"Generated images: {images_dir}")
    print(f"Evaluation results: {results_dir}")
    
    # List generated files
    print("\nGenerated files:")
    for root, dirs, files in os.walk(results_dir):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                print(f"  - {file_path}")

@app.local_entrypoint()
def main():
    """Run evaluation on Modal."""
    print("Starting evaluation...")
    print(f"Model: {MODEL_NAME}")
    
    evaluate_images_only.remote()
    
    print("\nTo download results, run:")
    print("modal volume get fifa-data /data/evaluation_output/evaluation_results ./local_eval_results")
