

"""Generate an image given a prompt using a trained model."""
import os

import functools
from argparse import ArgumentParser  # pylint: disable=g-importing-member
import random
import json
from diffusers import DDIMScheduler, DPMSolverMultistepScheduler # pylint: disable=g-importing-member
from diffusers import StableDiffusionPipeline  # pylint: disable=g-importing-member
import numpy as np
import torch
from tqdm import tqdm
import os
import ImageReward as imagereward
from transformers import CLIPModel  # pylint: disable=g-multiple-import
from transformers import CLIPTokenizer  # pylint: disable=g-multiple-import
from accelerate import PartialState
from diffusers import AutoencoderKL, StableDiffusionPipeline, UNet2DConditionModel, StableDiffusionXLPipeline
from PIL import Image

# torch._inductor.config.conv_1x1_as_mm = True
# torch._inductor.config.coordinate_descent_tuning = True
# torch._inductor.config.epilogue_fusion = False
# torch._inductor.config.coordinate_descent_check_all_directions = True


state = PartialState()



# calculate reward image reward
def _calculate_reward_ir(
        pipe,
        args,
        reward_tokenizer,
        tokenizer,
        weight_dtype,
        reward_clip_model,
        image_reward,
        imgs,
        prompts,
        test_flag=True,
):
    from utils.imagereward import image_reward_get_reward
    
    """Computes reward using ImageReward model."""
    if test_flag:
        image_pil = imgs
    else:
        image_pil = pipe.numpy_to_pil(imgs)[0]
    blip_reward, _ = image_reward_get_reward(
            image_reward, image_pil, prompts, weight_dtype
    )
    if args.reward_filter == 1:
        blip_reward = torch.clamp(blip_reward, min=0)
    inputs = reward_tokenizer(
            prompts,
            max_length=tokenizer.model_max_length,
            padding="do_not_pad",
            truncation=True,
    )
    input_ids = inputs.input_ids
    padded_tokens = reward_tokenizer.pad(
            {"input_ids": input_ids}, padding=True, return_tensors="pt"
    )
    txt_emb = reward_clip_model.get_text_features(
            input_ids=padded_tokens.input_ids.to("cuda").unsqueeze(0)
    )
    return blip_reward.cpu().squeeze(0).squeeze(0), txt_emb.squeeze(0)



def main(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)

    weight_dtype = torch.float16
    device = state.device

    print("==============================", args.version, "==============================")

    model_path = args.model_path

    print(f"Model path: {model_path}")
    print(f"Pretrained model: {args.pretrained_model_name_or_path}")
    
    # Debug: Check if model path exists and what's in it
    if model_path:
        if os.path.exists(model_path):
            print(f"Model path exists: {model_path}")
            print(f"Contents: {os.listdir(model_path)}")
        else:
            print(f"WARNING: Model path does not exist: {model_path}")

    # OPTION 0
    if "xl" in args.pretrained_model_name_or_path:
        unet = UNet2DConditionModel.from_pretrained(
                    model_path if model_path else args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
            )
        # use_safetensors=True, 
                    # variant="fp16"
        unet.to(device, dtype=weight_dtype)
    else:
        unet = UNet2DConditionModel.from_pretrained(
                    model_path if model_path else args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
            )
        unet.to(device, dtype=weight_dtype)
    
    
    if "xl" in args.pretrained_model_name_or_path:
        vae = AutoencoderKL.from_pretrained(
                                "madebyollin/sdxl-vae-fp16-fix",
                                subfolder=None,
                                revision=args.revision,
                                torch_dtype=weight_dtype,
                        )
        pipe = StableDiffusionXLPipeline.from_pretrained(
                    args.pretrained_model_name_or_path,
                    unet=unet,
                    vae=vae,
                    revision=args.revision,
                    safety_checker=None,
                    torch_dtype=weight_dtype,
                    use_safetensors=True, 
                    variant="fp16"
            )
        pipe = pipe.to(device)
    else:
        pipe = StableDiffusionPipeline.from_pretrained(
                    args.pretrained_model_name_or_path,
                    unet=unet,
                    revision=args.revision,
                    safety_checker=None,
            )

    # pipe.unet.to(memory_format=torch.channels_last)
    # pipe.vae.to(memory_format=torch.channels_last)
    # pipe.unet = torch.compile(
    #     pipe.unet, mode="max-autotune", fullgraph=True
    # )
    # pipe.unet.compile_repeated_blocks(fullgraph=True)

    # pipe.vae.decode = torch.compile(
    #     pipe.vae.decode,
    #     mode="max-autotune",
    #     fullgraph=True
    # )
    # pipe.vae.decode.compile_repeated_blocks(fullgraph=True)

    pipe.set_progress_bar_config(disable=True)
    # pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)


    if "xl" not in args.pretrained_model_name_or_path:
        pipe.unet.to(device, dtype=weight_dtype)
        pipe.vae.to(device, dtype=weight_dtype)

        pipe.text_encoder.to(device, dtype=weight_dtype)
    
    pipe.enable_xformers_memory_efficient_attention()
    

    # read prompt json path and then get prompt list
    # keys of json files are the prompts
    if args.prompts_path is not None:
        with open(args.prompts_path, "r") as f:
            prompt_list = json.load(f)

    if type(prompt_list) == dict:
        prompt_list = list(prompt_list.keys()) if isinstance(prompt_list, dict) else prompt_list
    elif type(prompt_list) == list:
        pass
    else:
        raise ValueError("Invalid prompt list")
    

    tokenizer = CLIPTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
    )

    if args.reward_type == "imagereward":
        image_reward = imagereward.load("ImageReward-v1.0")
        image_reward.requires_grad_(False)
        image_reward.to(device, dtype=weight_dtype)

        # reward models
        reward_clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14", use_fast=True).to(device, dtype=weight_dtype)

        reward_tokenizer = CLIPTokenizer.from_pretrained(
                "openai/clip-vit-large-patch14"
        )

        calculate_reward = functools.partial(
                    _calculate_reward_ir,
                    pipe,
                    args,
                    reward_tokenizer,
                    tokenizer,
                    weight_dtype,
                    reward_clip_model,
                    image_reward,
            )
    elif args.reward_type == "pickscore":
        from utils.pickscore_utils import Selector
        selector = Selector(device)
        calculate_reward = functools.partial(
                    selector.score,
            )
    elif args.reward_type == "aesthetic":
        from utils.aes_utils import Selector
        selector = Selector(device)
        calculate_reward = functools.partial(
                    selector.score,
            )
    elif args.reward_type == "clipscore":
        from utils.clip_utils import Selector
        selector = Selector(device)
        calculate_reward = functools.partial(
                    selector.score,
            )
    elif args.reward_type == "hpsv2":
        from utils.hps_utils import Selector
        selector = Selector(device)
        calculate_reward = functools.partial(
                    selector.score,
            )

    # check the output directory
    if state.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Wait for main process to create directory
    state.wait_for_everyone()

    # if already exist, remove the file
    if args.model_path is None:
        args.version = "pretrain_sd15"
        if 'xl' in args.pretrained_model_name_or_path:
            args.version = "pretrain_sdxl"

    # Set up simplified directory structure
    output_dir = os.path.join(args.output_dir, args.version, args.dataset)
    image_folder = os.path.join(output_dir, "images")
    output_file = os.path.join(output_dir, f"{args.reward_type}.json")
    
    # Create image directory upfront (all prompts will use the same directory)
    if state.is_main_process:
        os.makedirs(image_folder, exist_ok=True)
    
    state.wait_for_everyone()

    num_imgs_per_prompt = args.num_imgs_per_prompt

    p_to_idx = {prompt: idx for idx, prompt in enumerate(prompt_list)}

    rewards = {}

    if args.model_path is None:
        args.version = "pretrain_sd15"
        if 'xl' in args.pretrained_model_name_or_path:
            args.version = "pretrain_sdxl"

    # Check if we should skip processing
    if args.overwrite == 0:
        # first check the existence of the file
        if os.path.exists(output_file):
            try:
                with open(output_file, "r") as f:
                    data = json.load(f)
                    if len(data) == len(prompt_list):
                        print("All prompts already processed. Exiting.")
                        return
            except (json.JSONDecodeError, FileNotFoundError):
                print(f"Error reading existing file {output_file}. Will overwrite.")
    
    # Create empty file to mark start of processing (only on main process)
    if state.is_main_process:
        with open(output_file, "w") as f:
            json.dump({}, f)
        print(f"Created output file: {output_file}")


    def subroutine(sub_prompts):
        # Initialize rewards structure
        for prompt in sub_prompts:
            if prompt not in rewards:
                rewards[prompt] = {}
                rewards[prompt]['rewards'] = []
        
        generator = torch.Generator(device).manual_seed(args.seed)
        
        # Create list of all (prompt, image_idx) pairs
        all_tasks = []
        for prompt in sub_prompts:
            prompt_idx = p_to_idx[prompt]  # Get original prompt index
            for img_idx in range(num_imgs_per_prompt):
                img_path = f"{image_folder}/{prompt_idx}_{prompt[:50]}_{args.seed}_{img_idx}.jpg"
                all_tasks.append((prompt, img_idx, img_path))
        
        print(f"Process {state.process_index}: Total tasks: {len(all_tasks)}")
        
        # Check which images already exist
        tasks_to_generate = []
        existing_images = {}
        
        for prompt, img_idx, img_path in all_tasks:
            if os.path.exists(img_path) and (args.overwrite == 0 or args.overwrite == 1):
                existing_images[(prompt, img_idx)] = img_path
            else:
                tasks_to_generate.append((prompt, img_idx, img_path))
        
        print(f"Process {state.process_index}: Existing images: {len(existing_images)}, Need to generate: {len(tasks_to_generate)}")
        
        # Generate all missing images in batches
        if tasks_to_generate:
            for batch_start in tqdm(range(0, len(tasks_to_generate), args.batch_size), 
                                  desc=f"Process {state.process_index} generating images"):
                batch_end = min(batch_start + args.batch_size, len(tasks_to_generate))
                batch_tasks = tasks_to_generate[batch_start:batch_end]
                
                # Extract prompts for this batch
                batch_prompts = [task[0] for task in batch_tasks]
                
                with torch.no_grad():
                    img_results = pipe(batch_prompts, eta=0.0, generator=generator, num_inference_steps=args.num_inference_steps).images
                
                # Save images
                for (prompt, img_idx, img_path), img_result in zip(batch_tasks, img_results):
                    assert img_results is not None, f"Image result is None for prompt: {prompt}, img_idx: {img_idx}, img_path: {img_path}"
                    img_result.save(img_path)
                    # Store in existing_images for later processing
                    existing_images[(prompt, img_idx)] = img_path
        
        # Calculate rewards for all images (existing + newly generated)
        for prompt, img_idx, img_path in tqdm(all_tasks, desc=f"Process {state.process_index} calculating rewards"):
            if (prompt, img_idx) in existing_images:
                img_path = existing_images[(prompt, img_idx)]
                img_result = Image.open(img_path)
                
                result = calculate_reward(img_result, prompt)
                if isinstance(result, tuple):
                    reward, _ = result
                else:
                    reward = result
                rewards[prompt]['rewards'].append(reward)
    
    
    with state.split_between_processes(prompt_list) as sub_prompts:
        print(f"Process {state.process_index}: Processing {len(sub_prompts)} prompts")
        subroutine(sub_prompts)
        
        # Calculate statistics for this process's rewards
        for prompt, v in rewards.items():
            if v['rewards']:  # Only calculate if there are rewards
                rewards[prompt]['mean'] = torch.mean(torch.tensor(v['rewards'])).item()
                rewards[prompt]['std'] = torch.std(torch.tensor(v['rewards'])).item()
                rewards[prompt]['rewards'] = [x.item() if hasattr(x, 'item') else x for x in v['rewards']]
        
        print(f"Process {state.process_index}: Completed {len(rewards)} prompts")

    # Gather rewards from all processes and prepare final_rewards
    # output_file and output_dir already defined above
    
    if state.num_processes > 1:
        # Multi-process: save each process's results to separate files, then merge
        process_output_file = os.path.join(output_dir, f"scores_process_{state.process_index}.json")
        with open(process_output_file, "w") as f:
            json.dump(rewards, f, indent=4)
        print(f"Process {state.process_index}: Saved {len(rewards)} prompts to {process_output_file}")
        
        # Wait for all processes to finish saving
        state.wait_for_everyone()
        
        # Only the main process merges all files and saves final results
        if not state.is_main_process:
            return
        
        final_rewards = {}
        
        # Load existing data if file exists
        if os.path.exists(output_file):
            with open(output_file, "r") as f:
                existing_data = json.load(f)
            final_rewards.update(existing_data)
        
        # Merge all process files
        for process_idx in range(state.num_processes):
            process_file = os.path.join(output_dir, f"scores_process_{process_idx}.json")
            if os.path.exists(process_file):
                with open(process_file, "r") as f:
                    process_data = json.load(f)
                final_rewards.update(process_data)
                print(f"Merged {len(process_data)} prompts from process {process_idx}")
                os.remove(process_file)
    else:
        # Single process: load existing data if exists
        final_rewards = rewards
        if os.path.exists(output_file):
            with open(output_file, "r") as f:
                existing_data = json.load(f)
            final_rewards.update(existing_data)
    
    # Shared code: Calculate overall mean and save final results
    prompt_means = []
    for prompt, data in final_rewards.items():
        if isinstance(data, dict) and 'mean' in data:
            prompt_means.append(data['mean'])
    
    overall_mean = None
    if prompt_means:
        overall_mean = sum(prompt_means) / len(prompt_means)
        print(f"Overall mean reward across all {len(prompt_means)} prompts: {overall_mean:.6f}")
        final_rewards['overall_mean'] = overall_mean
    
    # Save the final results
    with open(output_file, "w") as f:
        json.dump(final_rewards, f, indent=4)
    num_prompts = len([k for k in final_rewards.keys() if k != 'overall_mean'])
    print(f"Saved final results with {num_prompts} prompts to {output_file}")


    del pipe
    torch.cuda.empty_cache()

    


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--prompts_path", default=None, type=str)
    parser.add_argument("--model-path", default=None, type=str)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--output-dir", default="outputs", type=str)
    parser.add_argument("--version", default="trained", type=str)
    parser.add_argument(
            "--pretrained_model_name_or_path",
            type=str,
            default="runwayml/stable-diffusion-v1-5",
            # required=True,
            help=(
                    "Path to pretrained model or model identifier from"
                    " huggingface.co/models."
            ),
    )
    parser.add_argument(
            "--revision",
            type=str,
            default=None,
            required=False,
            help=(
                    "Revision of pretrained model identifier from huggingface.co/models."
            ),
    )
    parser.add_argument(
            "--reward_filter",
            type=int,
            default=0,
            help="0: raw value, 1: took positive",
    )
    parser.add_argument(
            "--dataset",
            type=str,
            default="pickscore"
    )
    parser.add_argument("--reward_type", default="pickscore", type=str)
    parser.add_argument(
                "--unet_init", type=str, default='', help="Initialize start of run from unet (not compatible w/ checkpoint load)"
        )
    parser.add_argument(
                "--num_imgs_per_prompt", type=int, default=4, help="Initialize start of run from unet (not compatible w/ checkpoint load)"
        )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size for image generation"
    )
    parser.add_argument(
        "--num_inference_steps", type=int, default=20, help="Number of denoising steps"
    )
    
    parser.add_argument(
        "--overwrite", type=int, default=0, help=
        """
            Overwrite existing files.
            0 : Do not overwrite everything 
            1 : Do not overwrite images but overwrite the json file
            Else : Overwrite everything
        """
    )

    main(parser.parse_args())