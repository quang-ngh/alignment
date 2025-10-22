import os
import json
import pandas as pd
from pathlib import Path
from PIL import Image
import torch
from tqdm import tqdm
import sys

# ImageReward removed due to shape mismatch errors

def evaluate_pickscore_only(images, prompts, device="cuda"):
    """Evaluate using only PickScore"""
    try:
        from PickScore import PickScore
        scorer = PickScore(device=device)
        
        scores = []
        for image, prompt in tqdm(zip(images, prompts), desc="PickScore evaluation", total=len(images)):
            try:
                if isinstance(image, str):
                    image = Image.open(image).convert("RGB")
                score = scorer.score(image, prompt)
                scores.append(score)
            except Exception as e:
                print(f"Error in PickScore evaluation: {e}")
                scores.append(0.0)
        
        return scores
    except Exception as e:
        print(f"Failed to initialize PickScore: {e}")
        return [0.0] * len(images)

# ImageReward evaluation removed due to shape mismatch errors

def evaluate_images_simple(images, prompts, output_file, device="cuda"):
    """Simple evaluation using only PickScore"""
    print(f"Evaluating {len(images)} images with {len(prompts)} prompts")
    
    # Evaluate with PickScore
    print("Running PickScore evaluation...")
    pickscore_scores = evaluate_pickscore_only(images, prompts, device)
    
    # Create results
    results = []
    for i, (image_path, prompt) in enumerate(zip(images, prompts)):
        results.append({
            "image_path": image_path,
            "prompt": prompt,
            "pickscore": pickscore_scores[i]
        })
    
    # Save results
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    
    # Print summary
    print(f"\nEvaluation Summary:")
    print(f"PickScore - Mean: {df['pickscore'].mean():.4f}, Std: {df['pickscore'].std():.4f}")
    
    return results
