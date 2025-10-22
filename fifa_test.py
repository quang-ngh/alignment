

"""Generate an image given a prompt using a trained model."""
import os

import functools
from argparse import ArgumentParser  # pylint: disable=g-importing-member
import random
import json
from diffusers import DDIMScheduler  # pylint: disable=g-importing-member
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


    pipe.set_progress_bar_config(disable=True)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)


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
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    

    num_imgs_per_prompt = args.num_imgs_per_prompt

    p_to_idx = {prompt: idx for idx, prompt in enumerate(prompt_list)}
    seeds = list(range(args.num_seeds))


    rewards = {}

    # if already exist, remove the file
    if args.model_path is None:
        args.version = "pretrain_sd15"
        if 'xl' in args.pretrained_model_name_or_path:
            args.version = "pretrain_sdxl"

    if args.overwrite == 0:
        # first check the existenc of the file
        if os.path.exists(f"{args.output_dir}/{args.version}_{args.dataset}.json"):
            with open(f"{args.output_dir}/{args.version}_{args.dataset}.json", "r") as f:
                data = json.load(f)
                if len(data) == len(prompt_list):
                    raise ValueError("Already finished")
            
    with open(f"{args.output_dir}/{args.version}_{args.dataset}.json", "w") as f:
        # save the empty dictionary
        json.dump({}, f)


    batch = 4
    def subroutine(prompt, seeds):
        pidx = p_to_idx[prompt]
        if prompt not in rewards:
            rewards[prompt] = {}
            rewards[prompt]['rewards'] = []
        for seed in seeds:
            generator = torch.Generator(device).manual_seed(seed)
            ### GENERATE IMAGE
            for idx in range((num_imgs_per_prompt + batch - 1) // batch):
                offset = idx * batch
                num_imgs = min(batch, num_imgs_per_prompt - offset)
                # Continue if all images for the batch already generated.

                # first check that img_paths are already generated
                if args.reward_type == "aesthetic" or args.reward_type == "clipscore":
                    image_folder = f"{args.output_dir}/{args.version}_{args.dataset}/images"
                else:
                    image_folder = f"{args.output_dir}/{args.reward_type}/{args.version}_{args.dataset}/images"
                

                
                img_paths = [f"{image_folder}/{prompt[:20]}_{seed}_{iidx + offset}.jpg" for iidx in range(num_imgs)]

                # if images are already generated, then skip
                if all([os.path.exists(img_path) for img_path in img_paths]) and (args.overwrite == 0 or args.overwrite == 1):
                    img_results = [None] * num_imgs

                else:
                    with torch.no_grad():
                        img_results = pipe([prompt] * num_imgs, eta=0.0, generator=generator).images

                for iidx, img_result in enumerate(img_results):

                    if not os.path.exists(f"{image_folder}"):
                        os.makedirs(f"{image_folder}")
                    img_path = f"{image_folder}/{prompt[:20]}_{seed}_{iidx + offset}.jpg"
                    # save img result
                    if img_result is not None:
                        try:
                            img_result.save(img_path)
                        except:
                            img_path = f"{image_folder}/{prompt[:20]}_{seed}_{iidx + offset}.jpg"
                            img_result.save(img_path.replace(":", "_"))

                    # if img_result is None, then load the image
                    if img_result is None:
                        img_result = Image.open(img_path)
                     

                    result = calculate_reward(img_result, prompt)
                    if isinstance(result, tuple):
                        reward, _ = result
                    else:
                        reward = result
                    rewards[prompt]['rewards'].append(reward)
    
    
    with state.split_between_processes(prompt_list) as sub_prompts:
        for prompt in tqdm(sub_prompts):
            subroutine(prompt, seeds)
        for prompt, v in rewards.items():
            rewards[prompt]['mean'] = torch.mean(torch.tensor(v['rewards'])).item()
            rewards[prompt]['std'] = torch.std(torch.tensor(v['rewards'])).item()
            rewards[prompt]['rewards'] = [x.item() if hasattr(x, 'item') else x for x in v['rewards']]
        # save file, but if exists, add more data


    # Todo : Current code requires two gpus for saving the file
    # If you want to use only one or more than two gpus, then you need to change the code
    with state.main_process_first():
        with open(f"{args.output_dir}/{args.version}_{args.dataset}.json", "r") as f:
            print(f"{args.output_dir}/{args.version}_{args.dataset}.json")
            data = json.load(f)

        with open(f"{args.output_dir}/{args.version}_{args.dataset}.json", "w") as f:
            data.update(rewards)
            json.dump(data, f, indent=4)

    state.wait_for_everyone()


    del pipe
    torch.cuda.empty_cache()

    


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--prompts_path", default=None, type=str)
    parser.add_argument("--model-path", default=None, type=str)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--output-dir", default="outputs", type=str)
    parser.add_argument("--version", default="trained", type=str)
    parser.add_argument("--num_seeds", default=1, type=int)
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
        "--overwrite", type=int, default=0, help=
        """
            Overwrite existing files.
            0 : Do not overwrite everything 
            1 : Do not overwrite images but overwrite the json file
            Else : Overwrite everything
        """
    )

    main(parser.parse_args())