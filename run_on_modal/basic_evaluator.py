import os
import json
import pandas as pd
from pathlib import Path
from PIL import Image
import torch
from tqdm import tqdm
import sys

def evaluate_clip_only(images, prompts, device="cuda"):
    """Evaluate using OpenCLIP similarity"""
    try:
        import open_clip
        
        # Load CLIP model
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k', device=device)
        tokenizer = open_clip.get_tokenizer('ViT-B-32')
        
        scores = []
        for image, prompt in tqdm(zip(images, prompts), desc="CLIP evaluation", total=len(images)):
            try:
                if isinstance(image, str):
                    image = Image.open(image).convert("RGB")
                
                # Preprocess image
                image_tensor = preprocess(image).unsqueeze(0).to(device)
                
                # Encode image and text
                with torch.no_grad():
                    image_features = model.encode_image(image_tensor)
                    text_features = model.encode_text(tokenizer([prompt]).to(device))
                    
                    # Normalize features
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                    
                    # Compute similarity
                    similarity = torch.cosine_similarity(image_features, text_features)
                    scores.append(similarity.item())
                    
            except Exception as e:
                print(f"Error in CLIP evaluation: {e}")
                scores.append(0.0)
        
        return scores
    except Exception as e:
        print(f"Failed to initialize CLIP: {e}")
        return [0.0] * len(images)

def evaluate_images_basic(images, prompts, output_file, device="cuda"):
    """Basic evaluation using only CLIP"""
    print(f"Evaluating {len(images)} images with {len(prompts)} prompts")
    
    # Evaluate with CLIP
    print("Running CLIP evaluation...")
    clip_scores = evaluate_clip_only(images, prompts, device)
    
    # Create results
    results = []
    for i, (image_path, prompt) in enumerate(zip(images, prompts)):
        results.append({
            "image_path": image_path,
            "prompt": prompt,
            "clip_similarity": clip_scores[i]
        })
    
    # Save results
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    
    # Print summary
    print(f"\nEvaluation Summary:")
    print(f"CLIP Similarity - Mean: {df['clip_similarity'].mean():.4f}, Std: {df['clip_similarity'].std():.4f}")
    
    return results
