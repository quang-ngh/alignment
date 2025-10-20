#!/usr/bin/env python3
"""
Generate images using UNet weights from Modal checkpoint.

Usage:
    python generate_with_unet.py \
        --checkpoint_path /data/training_runs/sd15_corrupted_dpo/checkpoint-1000 \
        --prompt_file /root/datasets/eval_prompts/hpsv2_photo.json \
        --output_dir /data/evaluation_output/generated_images/photo
"""

import argparse
import json
import os
import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from PIL import Image
from tqdm import tqdm


def load_pipeline_with_unet(checkpoint_path: str, device: str = "cuda"):
    """Load pipeline with custom UNet from checkpoint."""
    print(f"Loading UNet from: {checkpoint_path}")
    
    # Load base pipeline
    pipeline = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False
    )
    
    # Load custom UNet
    unet_path = os.path.join(checkpoint_path, "unet")
    custom_unet = UNet2DConditionModel.from_pretrained(unet_path, torch_dtype=torch.float16)
    pipeline.unet = custom_unet
    pipeline = pipeline.to(device)
    
    print("✓ Pipeline loaded with custom UNet")
    return pipeline


def generate_images(pipeline, prompts, output_dir, seed=42):
    """Generate images for prompts."""
    os.makedirs(output_dir, exist_ok=True)
    generator = torch.Generator(device=pipeline.device).manual_seed(seed)
    
    for idx, prompt in enumerate(tqdm(prompts, desc="Generating")):
        try:
            result = pipeline(
                prompt=prompt,
                num_images_per_prompt=1,
                guidance_scale=7.5,
                num_inference_steps=50,
                generator=generator
            )
            result.images[0].save(os.path.join(output_dir, f"image_{idx}.jpg"), "JPEG", quality=95)
        except Exception as e:
            print(f"Error generating image {idx}: {e}")
            Image.new("RGB", (512, 512), color="white").save(os.path.join(output_dir, f"image_{idx}.jpg"), "JPEG")
    
    print(f"✓ Generated images saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--prompt_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    
    # Load pipeline
    pipeline = load_pipeline_with_unet(args.checkpoint_path)
    
    # Load prompts
    with open(args.prompt_file, 'r') as f:
        prompts = json.load(f)
    
    # Generate images
    generate_images(pipeline, prompts, args.output_dir)
    print("Image generation completed!")


if __name__ == "__main__":
    main()
