import os
import json 
import numpy as np
import torch
from diffusers import AutoencoderKL, StableDiffusionPipeline, UNet2DConditionModel, DPMSolverMultistepScheduler
import random
from tqdm import tqdm

def generate_noise(n_samples=500, size=(4,64,64), seed=999):
    torch.manual_seed(seed)
    size = (n_samples, *size)
    random_noise = torch.randn(size, device="cuda", dtype=torch.bfloat16, generator=torch.Generator(device="cuda").manual_seed(seed))
    return random_noise


def get_sd_model(model_path, unet_path=None, version="sd15", device="cuda"):
    if unet_path is None:
        unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet", torch_dtype=torch.float16).to(device)
    else:
        unet = UNet2DConditionModel.from_pretrained(unet_path, torch_dtype=torch.float16).to(device)

    scheduler = DPMSolverMultistepScheduler.from_config(model_path, subfolder="scheduler")

    if version == "sd15":        
        pipeline = StableDiffusionPipeline.from_pretrained(
            model_path, 
            unet=unet,
            torch_dtype=torch.float16,
            safety_checker=None,
            scheduler=scheduler,
        ).to(device)
        return pipeline

    vae = AutoencoderKL.from_pretrained("checkpoints", subfolder="sdxl_vae_fp16_fix", torch_dtype=torch.float16).to(device)
    if version == "sdxl":
        from diffusers import StableDiffusionXLPipeline
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            model_path, 
            unet=unet, 
            vae=vae,
            torch_dtype=torch.float16, 
            safety_checker=None,
            scheduler=scheduler
        ).to(device)
        return pipeline
    
@torch.inference_mode()
def generate_hpsv2(
    pipeline, 
    noise_path: str = "datasets/hpsv2_noise.pt", 
    json_path: str = "datasets/hpsv2_anime.json", 
    save_dir: str = "./output",
    inference_steps: int = 20,
    guidance_scale: float = 7.5,   
    batch_size: int = 32,
    start_idx: int=0,
    end_idx: int=-1,
    dtype=torch.float16,
):
    basename = json_path.split("/")[-1].split(".")[0]
    save_dir = os.path.join(save_dir, basename)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    
    with open(json_path, "r") as f:
        list_prompts = json.load(f)
    f.close()

    #   Generate images
    pre_sample_latents = torch.load(noise_path, map_location="cpu")
    for i in tqdm(range(0, len(list_prompts), batch_size)):
        batch_prompts = list_prompts[i:i+batch_size]
        batch_noise = pre_sample_latents[i:i+batch_size].to(pipeline.device, dtype=dtype)

        #   check the existence of path
        gen_kwargs = {
            "prompt": batch_prompts,
            "latents": batch_noise,
            "num_inference_steps": inference_steps,
            "guidance_scale": guidance_scale, 
            "output_type": "pil",
        }
        batch_images = pipeline(**gen_kwargs).images
        for j, image in enumerate(batch_images):
            image.save(os.path.join(save_dir, f"image_{i+j}.jpg"))

@torch.inference_mode()
def generate_pickapic_test(
    pipeline,
    noise_path: str = "datasets/pickapic_test_noise.pt",
    save_dir: str = "./output",
    json_path: str = "datasets/eval_prompts/pickapic_test_prompts.json",
    inference_steps: int = 20,
    guidance_scale: float = 4.5,   
    batch_size: int = 32,
    start_idx: int=0,
    end_idx: int=-1,
    dtype=torch.float16,
):
    list_prompts = json.load(open(json_path, "r"))

    if noise_path != "":
        print("Using pre-generated noise latents")
        pre_sample_latents = torch.load(noise_path, map_location="cpu")
        pre_sample_latents = pre_sample_latents.to(pipeline.device, dtype=dtype)
        random_seed = 999

    # else:
    # random_seed = args.random_seed
    # pre_sample_latents = generate_noise(
    #     n_samples=len(list_prompts),
    #     size=(4,128,128),
    #     seed=random_seed,
    # )
    # pre_sample_latents = pre_sample_latents.to(pipeline.device, dtype=dtype)
    save_dir = os.path.join(save_dir, f"random_seed_{random_seed}")
    print(f"Using random seed {random_seed} for noise generation, generated {len(list_prompts)} noise latents")
    print(f"Saving images to {save_dir}")

    
    if end_idx < 0:
        end_idx = len(list_prompts)
    
    start_idx = max(start_idx, 0)
    os.makedirs(save_dir, exist_ok=True)
    total_prompts = end_idx - start_idx
    check_generated = {}

    for i in tqdm(range(0, total_prompts, batch_size)):
        _run = True       #   check if the batch is already generated

        batch_prompts = list_prompts[start_idx + i:start_idx + i + batch_size]
        batch_latents = pre_sample_latents[start_idx + i:start_idx + i + batch_size]
        if len(batch_prompts) != batch_size:
            batch_prompts = batch_prompts + [batch_prompts[-1]] * (batch_size - len(batch_prompts))
            batch_latents = torch.cat([batch_latents, batch_latents[-1].unsqueeze(0).repeat(batch_size - len(batch_latents), *([1]*(batch_latents.dim()-1)))], dim=0)

        for j in range(len(batch_prompts)):
            check_idx = start_idx + i + j
            check_path = os.path.join(save_dir, f"image_{check_idx}.jpg")
            if not os.path.exists(check_path):
                _run = False
                break
        
        if not _run:
            print(f"Generating batch {i}")
            gen_kwargs = {
                "prompt": batch_prompts,
                "latents": batch_latents.to(pipeline.device, dtype=dtype),
                "num_inference_steps": inference_steps,
                "guidance_scale": guidance_scale,
                "output_type": "pil",
                "eta": 0.0,
            }
            batch_images = pipeline(**gen_kwargs).images
            for j, image in enumerate(batch_images):
                idx = start_idx + i + j
                image.save(os.path.join(save_dir, f"image_{idx}.jpg"))
        else:
            print(f"Batch {i} is already generated")

def main(args):

    unet_path = args.unet_path
    if unet_path=="":
        unet_path = None # Original model
    else:
        unet_path = unet_path
    print(f"Using unet from {unet_path}")

    pipeline = get_sd_model(model_path=args.model_path, unet_path=unet_path, version=args.version)
    pipeline.set_progress_bar_config(disable=True)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    print(f"Saved images to {args.save_dir}")

    if args.gen_type == "hpsv2":
        list_hpsv2_json = [
            "datasets/eval_prompts/hpsv2_anime.json",
            "datasets/eval_prompts/hpsv2_concept_art.json",
            "datasets/eval_prompts/hpsv2_paintings.json",
            "datasets/eval_prompts/hpsv2_photo.json",
        ]
        for json_path in list_hpsv2_json:
            generate_hpsv2(
                pipeline, 
                save_dir=args.save_dir,
                noise_path=args.noise_path, 
                json_path=json_path, 
                batch_size=args.batch_size,
                guidance_scale=args.guidance_scale,
                inference_steps=args.inference_steps
            )
    elif args.gen_type == "pickapic_test":
        generate_pickapic_test(
            pipeline,
            save_dir=args.save_dir,
            noise_path=args.noise_path,
            json_path=args.json_path,
            batch_size=args.batch_size,
            start_idx=args.start_idx,
            end_idx=args.end_idx,
            guidance_scale=args.guidance_scale,
            inference_steps=args.inference_steps
        )
    elif args.gen_type == "partipromps":
        raise NotImplementedError("Participating prompts generation is not implemented yet")

if __name__ == "__main__":
    # noise_sd15 = generate_noise(
    #     n_samples=1632, # 800 prompts for each category
    #     size=(4,128,128), # 64 for sd15, 128 for sdxl, channel=4
    #     seed=999,
    # )
    # torch.save(noise_sd15.detach().cpu(), "datasets/partiprompts_noise_sdxl.pt")

    from omegaconf import OmegaConf
    args = OmegaConf.from_cli()
    main(args)