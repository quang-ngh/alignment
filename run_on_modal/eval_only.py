#!/usr/bin/env python3
"""
Evaluate generated images using HPSv2 and PickScore only.

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
    .add_local_file("../evaluator.py", "/root/evaluator.py")
    .add_local_dir("../datasets/eval_prompts", "/root/datasets/eval_prompts")
    .add_local_dir("../HPSv2", "/root/HPSv2")
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
    """Evaluate generated images using PickScore and HPSv2 only."""
    import sys
    import subprocess
    import pandas as pd
    from tqdm import tqdm
    
    # Setup
    sys.path.insert(0, "/root")
    sys.path.insert(0, "/root/HPSv2")
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    
    print(f"Starting evaluation for: {MODEL_NAME}")
    
    # Install HPSv2
    print("Installing HPSv2...")
    try:
        # Install requirements first
        subprocess.run(["pip", "install", "-r", "/root/HPSv2/requirements.txt"], 
                      capture_output=True, text=True, check=True)
        print("HPSv2 requirements installed successfully")
        
        # Install HPSv2 package
        subprocess.run(["pip", "install", "-e", "/root/HPSv2"], 
                      capture_output=True, text=True, check=True)
        print("HPSv2 package installed successfully")
        
        # Verify installation
        import hpsv2
        print("HPSv2 import successful")
        
    except subprocess.CalledProcessError as e:
        print(f"HPSv2 installation failed: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return
    except ImportError as e:
        print(f"HPSv2 import failed: {e}")
        print("Trying to install HPSv2 from GitHub...")
        try:
            subprocess.run(["pip", "install", "git+https://github.com/tgxs002/HPSv2.git"], 
                          capture_output=True, text=True, check=True)
            import hpsv2
            print("HPSv2 installed from GitHub successfully")
        except Exception as e2:
            print(f"Failed to install HPSv2 from GitHub: {e2}")
            print("Trying to use local HPSv2 module...")
            try:
                # Try to import directly from the local directory
                import sys
                sys.path.insert(0, "/root/HPSv2/hpsv2")
                import hpsv2
                print("HPSv2 imported from local directory successfully")
            except Exception as e3:
                print(f"Failed to import HPSv2 from local directory: {e3}")
                return
    
    # Setup directories
    images_dir = Path("/data/evaluation_output_1k/generated_images")
    results_dir = Path("/data/evaluation_output_1k_2/evaluation_results_1k_2")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    if not images_dir.exists():
        print(f"Error: Images directory {images_dir} does not exist!")
        print("Please run generate_only.py first to generate images.")
        return
    
    # Get image categories
    image_categories = [d for d in os.listdir(images_dir) if os.path.isdir(images_dir / d)]
    print(f"Found image categories: {image_categories}")
    
    if not image_categories:
        print("No image categories found!")
        return
    
    # Import evaluators
    from evaluator import evaluate_pickscore, evaluate_hpsv2
    
    print("\n" + "="*60)
    print("EVALUATING IMAGES WITH PICKSORE AND HPSV2")
    print("="*60)
    
    all_results = {}
    
    for category in image_categories:
        image_dir = images_dir / category
        prompt_path = f"/root/datasets/eval_prompts/hpsv2_{category}.json"
        
        if not os.path.exists(prompt_path):
            print(f"Prompt file hpsv2_{category}.json not found, skipping {category}")
            continue
        
        # Load prompts
        with open(prompt_path, 'r') as f:
            prompts = json.load(f)
        
        # Get available images
        image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
        if not image_files:
            print(f"No images found in {image_dir}, skipping {category}")
            continue
        
        # Limit to available images
        num_images = min(len(prompts), len(image_files))
        prompts = prompts[:num_images]
        
        print(f"Evaluating {num_images} image-prompt pairs for {category}...")
        
        # Evaluate images
        detailed_results = []
        for i in tqdm(range(num_images), desc=f"Evaluating {category}"):
            image_path = os.path.join(image_dir, f"image_{i}.jpg")
            if not os.path.exists(image_path):
                continue
            
            try:
                pickscore = evaluate_pickscore(image=image_path, prompt=prompts[i])
                hpsv2_score = evaluate_hpsv2(image=image_path, prompt=prompts[i], hps_version="v2.1")
                
                detailed_results.append({
                    'image_id': i,
                    'prompt': prompts[i],
                    'image_path': image_path,
                    'pickscore': pickscore,
                    'hpsv2': hpsv2_score
                })
            except Exception as e:
                print(f"Error evaluating image {i}: {e}")
                continue
        
        # Save results
        if detailed_results:
            df = pd.DataFrame(detailed_results)
            detailed_output_path = results_dir / f"{MODEL_NAME}_{category}_detailed_1k_2.csv"
            df.to_csv(detailed_output_path, index=False)
            
            # Calculate summary
            summary = {
                'category': category,
                'num_samples': len(detailed_results),
                'hpsv2_mean': df['hpsv2'].mean(),
                'hpsv2_std': df['hpsv2'].std(),
                'pickscore_mean': df['pickscore'].mean(),
                'pickscore_std': df['pickscore'].std(),
            }
            
            all_results[category] = summary
            
            print(f"Category: {category}")
            print(f"  Samples: {summary['num_samples']}")
            print(f"  HPSv2: {summary['hpsv2_mean']:.4f} ± {summary['hpsv2_std']:.4f}")
            print(f"  PickScore: {summary['pickscore_mean']:.4f} ± {summary['pickscore_std']:.4f}")
    
    # Save overall summary
    if all_results:
        summary_df = pd.DataFrame.from_dict(all_results, orient='index')
        summary_output_path = results_dir / f"{MODEL_NAME}_summary_1k_2.csv"
        summary_df.to_csv(summary_output_path)
        print(f"\nOverall summary saved to {summary_output_path}")
    
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
                print(f"  - {os.path.join(root, file)}")

@app.local_entrypoint()
def main():
    """Run evaluation on Modal."""
    print("Starting evaluation...")
    print(f"Model: {MODEL_NAME}")
    
    evaluate_images_only.remote()
    
    print("\nTo download results, run:")
    print("modal volume get fifa-data /data/evaluation_output_1k_2/evaluation_results_1k_2 ./local_eval_results")