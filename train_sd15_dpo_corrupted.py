

import argparse
import io
import logging
import math
import os
import random
import json

import accelerate
import datasets
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from transformers.utils import ContextManagers

import diffusers
from diffusers.optimization import get_scheduler
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel, StableDiffusionXLPipeline
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
import copy
from src.utils import *

if is_wandb_available():
    import wandb

    
    
## SDXL
from transformers import AutoTokenizer, PretrainedConfig
from src.dataset import BaseDataset, DubiousDataset


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.20.0")

logger = get_logger(__name__, log_level="INFO")

DATASET_NAME_MAPPING = {
    "yuvalkirstain/pickapic_v1": ("jpg_0", "jpg_1", "label_0", "caption"),
    "yuvalkirstain/pickapic_v2": ("jpg_0", "jpg_1", "label_0", "caption"),
    "ymhao/HPDv2": ("jpg_0", "jpg_1", "label_0", "caption"),
}

        
def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--input_perturbation", type=float, default=0, help="The scale of input perturbation. Recommended 0.1."
    )
    parser.add_argument(
        "--good_manifest", type=str, default='', help="Good manifest"
    )
    parser.add_argument(
        "--dubious_manifest", type=str, default='', help="Dubious manifest"
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing an image."
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="caption",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-finetuned",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None,
                        # was random for submission, need to test that not distributing same noise etc across devices
                        help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=None,
        help=(
            "The resolution for input images, all the images in the dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--random_crop",
        default=False,
        action="store_true",
        help=(
            "If set the images will be randomly"
            " cropped (instead of center). The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--no_hflip",
        action="store_true",
        help="whether to supress horizontal flipping",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=2000,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-8,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant_with_warmup",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--use_adafactor", action="store_true", help="Whether or not to use adafactor (should save mem)"
    )
    # Bram Note: Haven't looked @ this yet
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--flip_percentage", type=float, default=0.0, help="Flip percentage")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default='', # latest
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="tuning",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )

    ## SDXL
    parser.add_argument(
        "--pretrained_vae_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained VAE model with better numerical stability. ",
    )
    parser.add_argument("--sdxl", action='store_true', help="Train sdxl")
    
    ## DPO
    parser.add_argument("--sft", action='store_true', help="Run Supervised Fine-Tuning instead of Direct Preference Optimization")
    parser.add_argument("--beta_dpo", type=float, default=5000, help="The beta DPO temperature controlling strength of KL penalty")
    parser.add_argument(
        "--hard_skip_resume", action="store_true", help="Load weights etc. but don't iter through loader for loader resume, useful b/c resume takes forever"
    )
    parser.add_argument(
        "--unet_init", type=str, default='', help="Initialize start of run from unet (not compatible w/ checkpoint load)"
    )
    parser.add_argument(
        "--proportion_empty_prompts",
        type=float,
        default=0.2,
        help="Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement).",
    )
    parser.add_argument(
        "--split", type=str, default='train', help="Datasplit"
    )
    parser.add_argument(
        "--choice_model", type=str, default='', help="Model to use for ranking (override dataset PS label_0/1). choices: aes, clip, hps, pickscore"
    )
    parser.add_argument(
        "--dreamlike_pairs_only", action="store_true", help="Only train on pairs where both generations are from dreamlike"
    )

    parser.add_argument(
        "--train_data_path", type=str, default=''
    )
    parser.add_argument(
        "--image_path", type=str, default='', help="Path to image folder"
    )
    parser.add_argument(
        "--image_to_reward_path", type=str, default='', help="Path to image folder"
    )
    parser.add_argument(
        "--use_new_label_0",
        default=False,
        action="store_true",
        help="Use new label_0 for pickapic_v2, not the original label_0",
    )
    parser.add_argument(
        "--lr_scheduler_rule", type=str, default=None, help="scheduler rule"
    )
    parser.add_argument(
        "--min_lr", type=float, default=0.0, help="minimum learning rate"
    )
    parser.add_argument(
        "--soft_label_mode", action="store_true", help="Use soft label (reward-weighted dpo loss) mode"
    )
    
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Sanity checks
    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Need either a dataset name or a training folder.")

    ## SDXL
    if args.sdxl:
        print("Running SDXL")
    if args.resolution is None:
        if args.sdxl:
            args.resolution = 1024
        else:
            args.resolution = 512
            
    args.train_method = 'sft' if args.sft else 'dpo'
    return args


# Adapted from pipelines.StableDiffusionXLPipeline.encode_prompt
def encode_prompt_sdxl(batch, text_encoders, tokenizers, proportion_empty_prompts, caption_column, is_train=True):
    prompt_embeds_list = []
    prompt_batch = batch[caption_column]

    captions = []
    for caption in prompt_batch:
        if random.random() < proportion_empty_prompts:
            captions.append("")
        elif isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(random.choice(caption) if is_train else caption[0])

    with torch.no_grad():
        for tokenizer, text_encoder in zip(tokenizers, text_encoders):
            text_inputs = tokenizer(
                captions,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            prompt_embeds = text_encoder(
                text_input_ids.to('cuda'),
                output_hidden_states=True,
            )

            # We are only ALWAYS interested in the pooled output of the final text encoder
            pooled_prompt_embeds = prompt_embeds[0]
            prompt_embeds = prompt_embeds.hidden_states[-2]
            bs_embed, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
            prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
    return {"prompt_embeds": prompt_embeds, "pooled_prompt_embeds": pooled_prompt_embeds}




def main():
    
    args = parse_args()
    
    #### START ACCELERATOR BOILERPLATE ###
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed + accelerator.process_index) # added in + term, untested

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    ### END ACCELERATOR BOILERPLATE
    
    
    ### START DIFFUSION BOILERPLATE ###
    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, 
                                                    subfolder="scheduler",
                                                    prediction_type="epsilon")
    def enforce_zero_terminal_snr(scheduler):
        # Modified from https://arxiv.org/pdf/2305.08891.pdf
        # Turbo needs zero terminal SNR to truly learn from noise
        # Turbo: https://static1.squarespace.com/static/6213c340453c3f502425776e/t/65663480a92fba51d0e1023f/1701197769659/adversarial_diffusion_distillation.pdf
        # Convert betas to alphas_bar_sqrt
        alphas = 1 - scheduler.betas
        alphas_bar = alphas.cumprod(0)
        alphas_bar_sqrt = alphas_bar.sqrt()

        # Store old values.
        alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
        alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()
        # Shift so last timestep is zero.
        alphas_bar_sqrt -= alphas_bar_sqrt_T
        # Scale so first timestep is back to old value.
        alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)

        alphas_bar = alphas_bar_sqrt ** 2
        alphas = alphas_bar[1:] / alphas_bar[:-1]
        alphas = torch.cat([alphas_bar[0:1], alphas])
    
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        scheduler.alphas_cumprod = alphas_cumprod
        return 

    if 'turbo' in args.pretrained_model_name_or_path:
        enforce_zero_terminal_snr(noise_scheduler)

    #   Init model
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
    )
    ref_unet = copy.deepcopy(unet)


    # Freeze vae, text_encoder(s), reference unet
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    ref_unet.requires_grad_(False)
    unet.requires_grad_(True)


    # xformers efficient attention
    if is_xformers_available():
        import xformers

        xformers_version = version.parse(xformers.__version__)
        if xformers_version == version.parse("0.0.16"):
            logger.warn(
                "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
            )
        unet.enable_xformers_memory_efficient_attention()

    # BRAM NOTE: We're using >=0.16.0. Below was a bit of a bug hive. I hacked around it, but ideally ref_unet wouldn't
    # be getting passed here
    # 
    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            
            if len(models) > 1:
                assert args.train_method == 'dpo' # 2nd model is just ref_unet in DPO case
            models_to_save = models[:1]
            for i, model in enumerate(models_to_save):
                model.save_pretrained(os.path.join(output_dir, "unet"))

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

        def load_model_hook(models, input_dir):

            if len(models) > 1:
                assert args.train_method == 'dpo' # 2nd model is just ref_unet in DPO case
            models_to_load = models[:1]
            for i in range(len(models_to_load)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing or args.sdxl: #  (args.sdxl and ('turbo' not in args.pretrained_model_name_or_path) ):
        print("Enabling gradient checkpointing, either because you asked for this or because you're using SDXL")
        unet.enable_gradient_checkpointing()

    # Bram Note: haven't touched
    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    resolution = (512,512) # sd1.5
    good_dataset = BaseDataset(manifest=args.good_manifest, image_dir=args.train_data_dir, resolution=resolution)
    dubious_dataset = DubiousDataset(manifest=args.dubious_manifest, image_dir=args.train_data_dir, flip_percentage=args.flip_percentage, resolution=resolution)

    # DataLoaders creation:
    good_dataloader = torch.utils.data.DataLoader(
        good_dataset,
        shuffle=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )
    dubious_dataloader = torch.utils.data.DataLoader(
        dubious_dataset,
        shuffle=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )
    ##### END BIG OLD DATASET BLOCK #####
    
    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(good_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    
    #### START ACCELERATOR PREP ####
    unet, optimizer, good_dataloader, dubious_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, good_dataloader, dubious_dataloader, lr_scheduler
    )

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

        
    # Move text_encode and vae to gpu and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    ref_unet.to(accelerator.device, dtype=weight_dtype)
    ### END ACCELERATOR PREP ###
    
    
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(good_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, tracker_config)

    # Training initialization
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num good examples = {len(good_dataset)}")
    logger.info(f"  Num dubious examples = {len(dubious_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size * 4}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size * 4}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0


    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)
        

    # Bram Note: This was pretty janky to wrangle to look proper but works to my liking now
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")


    #### START MAIN TRAINING LOOP #####
    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        train_loss = 0.0
        implicit_acc_accumulated = 0.0
        for step, (good_batch, dubious_batch) in enumerate(zip(good_dataloader, dubious_dataloader)):
            # Skip steps until we reach the resumed step 
            with accelerator.accumulate(unet):

                # Get data from good dataset
                from_good_prompts = good_batch["prompt"]
                from_good_win_images = good_batch["win_image"]
                from_good_lose_images = good_batch["lose_image"]
                from_good_ids = good_batch["refer_id"]
                use_latent_from_good = good_batch.get("use_latent", False)
                if use_latent_from_good:
                    from_good_win_latents = good_batch["win_latent"]
                    from_good_lose_latents = good_batch["lose_latent"]
                else:
                    with torch.no_grad():
                        from_good_win_latents = vae.encode(from_good_win_images.to(weight_dtype)).latent_dist.sample() * vae.config.scaling_factor
                        from_good_lose_latents = vae.encode(from_good_lose_images.to(weight_dtype)).latent_dist.sample() * vae.config.scaling_factor

                # Get data from dubious dataset
                from_dubious_prompts = dubious_batch["prompt"]
                from_dubious_win_images = dubious_batch["win_image"]
                from_dubious_lose_images = dubious_batch["lose_image"]
                from_dubious_ids = dubious_batch["refer_id"]
                use_latent_from_dubious = dubious_batch.get("use_latent", False)
                if use_latent_from_dubious:
                    from_dubious_win_latents = dubious_batch["win_latent"]
                    from_dubious_lose_latents = dubious_batch["lose_latent"]
                else:
                    with torch.no_grad():
                        from_dubious_win_latents = vae.encode(from_dubious_win_images.to(weight_dtype)).latent_dist.sample() * vae.config.scaling_factor
                        from_dubious_lose_latents = vae.encode(from_dubious_lose_images.to(weight_dtype)).latent_dist.sample() * vae.config.scaling_factor

                # Get the actual batch size if two datasets are not the same size
                bsz = min(
                    len(from_good_prompts),
                    len(from_dubious_prompts),
                )

                win_latents = torch.cat(
                    [
                        from_good_win_latents[:bsz],
                        from_dubious_win_latents[:bsz],
                    ],
                    dim=0
                )
                lose_latents = torch.cat(
                    [
                        from_good_lose_latents[:bsz],
                        from_dubious_lose_latents[:bsz],
                    ],
                    dim=0
                )

                latents = torch.cat(
                    [win_latents, lose_latents],
                    dim=0
                ).to(accelerator.device, dtype=weight_dtype)
                assert latents.shape[0] == 4 * bsz, "Latents should be twice the batch size"

                noise = torch.randn_like(latents) 
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()
                # only first 20% timesteps for SDXL refiner
                if 'refiner' in args.pretrained_model_name_or_path:
                    timesteps = timesteps % 200
                elif 'turbo' in args.pretrained_model_name_or_path:
                    timesteps_0_to_3 = timesteps % 4
                    timesteps = 250 * timesteps_0_to_3 + 249
                
                timesteps = timesteps.chunk(2)[0].repeat(2)
                noise = noise.chunk(2)[0].repeat(2, 1, 1, 1) 
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                input_prompts = from_good_prompts[:bsz] + from_dubious_prompts[:bsz]
                input_ids = tokenizer(input_prompts, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt").input_ids
                with torch.no_grad():
                    encoder_hidden_states = text_encoder(input_ids.to(accelerator.device))[0]
                    encoder_hidden_states = encoder_hidden_states.repeat(2, 1, 1)   #   duplicate for DPO

                ### START PREP BATCH ###
                # if args.sdxl:
                #     # Get the text embedding for conditioning
                #     with torch.no_grad():
                       
                #         if 'refiner' in args.pretrained_model_name_or_path:
                #             add_time_ids = torch.tensor([args.resolution, 
                #                                          args.resolution,
                #                                          0,
                #                                          0,
                #                                           6.0], 
                #                                          dtype=weight_dtype,
                #                                          device=accelerator.device)[None, :].repeat(timesteps.size(0), 1)
                #         else: # SDXL-base
                #             add_time_ids = torch.tensor([args.resolution, 
                #                                          args.resolution,
                #                                          0,
                #                                          0,
                #                                           args.resolution, 
                #                                          args.resolution],
                #                                          dtype=weight_dtype,
                #                                          device=accelerator.device)[None, :].repeat(timesteps.size(0), 1)
                #         prompt_batch = encode_prompt_sdxl(batch, 
                #                                           text_encoders,
                #                                            tokenizers,
                #                                            args.proportion_empty_prompts, 
                #                                           caption_column=caption_column,
                #                                            is_train=True,
                #                                           )
                #     if args.train_method == 'dpo':
                #         prompt_batch["prompt_embeds"] = prompt_batch["prompt_embeds"].repeat(2, 1, 1)
                #         prompt_batch["pooled_prompt_embeds"] = prompt_batch["pooled_prompt_embeds"].repeat(2, 1)
                #     unet_added_conditions = {"time_ids": add_time_ids,
                #                             "text_embeds": prompt_batch["pooled_prompt_embeds"]}
                # else: # sd1.5
                #     # Get the text embedding for conditioning
                #     encoder_hidden_states = text_encoder(batch["input_ids"])[0]
                #     if args.train_method == 'dpo':
                #         encoder_hidden_states = encoder_hidden_states.repeat(2, 1, 1)
                #### END PREP BATCH ####

                        
                assert noise_scheduler.config.prediction_type == "epsilon"
                target = noise
               
                # Make the prediction from the model we're learning
                model_batch_args = (noisy_latents,
                                    timesteps, 
                                    encoder_hidden_states)
                added_cond_kwargs = None
                
                model_pred = unet(
                                *model_batch_args,
                                  added_cond_kwargs = added_cond_kwargs
                                 ).sample
                #### START LOSS COMPUTATION ####
                # model_pred and ref_pred will be (2 * LBS) x 4 x latent_spatial_dim x latent_spatial_dim
                # losses are both 2 * LBS
                # 1st half of tensors is preferred (y_w), second half is unpreferred
                model_losses = (model_pred - target).pow(2).mean(dim=[1,2,3])
                model_losses_w, model_losses_l = model_losses.chunk(2)
                # below for logging purposes
                raw_model_loss = 0.5 * (model_losses_w.mean() + model_losses_l.mean())
                
                model_diff = model_losses_w - model_losses_l # These are both LBS (as is t)
                # (e_w - e_theta(x_w)^2)
                # (e_l - e_theta(x_l)^2)
                
                with torch.no_grad(): # Get the reference policy (unet) prediction
                    ref_pred = ref_unet(
                                *model_batch_args,
                                    added_cond_kwargs = added_cond_kwargs
                                    ).sample.detach()
                    ref_losses = (ref_pred - target).pow(2).mean(dim=[1,2,3])
                    ref_losses_w, ref_losses_l = ref_losses.chunk(2)
                    ref_diff = ref_losses_w - ref_losses_l
                    raw_ref_loss = ref_losses.mean()    
                    
                scale_term = -0.5 * args.beta_dpo
                inside_term = scale_term * (model_diff - ref_diff)
                implicit_acc = (inside_term > 0).sum().float() / inside_term.size(0)
                
                # if args.soft_label_mode:
                #     # prepare reward values
                #     loss1 = -1 * F.logsigmoid(inside_term)
                #     loss2 = -1 * F.logsigmoid(-inside_term)
                #     batch_reward = batch["reward_values"].to(dtype=weight_dtype,
                #                                         device=accelerator.device)
                #     # softmax over rewards dim=-1
                #     T = 0.01
                #     reward_softmax = F.softmax(batch_reward/T, dim=-1)
                #     eps_1 = reward_softmax[:,0]
                #     eps_2 = reward_softmax[:,1]

                #     # print(f"eps_1: {eps_1}, eps_2: {eps_2}")

                #     loss = (eps_1 * loss1 + eps_2 * loss2).mean()

                # else:
                loss = -1 * F.logsigmoid(inside_term).mean()
                #### END LOSS COMPUTATION ###
                    
                # Gather the losses across all processes for logging 
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps
                # Also gather:
                # - model MSE vs reference MSE (useful to observe divergent behavior)
                # - Implicit accuracy
                avg_model_mse = accelerator.gather(raw_model_loss.repeat(args.train_batch_size)).mean().item()
                avg_ref_mse = accelerator.gather(raw_ref_loss.repeat(args.train_batch_size)).mean().item()
                avg_acc = accelerator.gather(implicit_acc).mean().item()
                implicit_acc_accumulated += avg_acc / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    if not args.use_adafactor: # Adafactor does itself, maybe could do here to cut down on code
                        accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            torch.cuda.synchronize()  # Wait for all operations to finish


            # Checks if the accelerator has just performed an optimization step, if so do "end of batch" logging
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                accelerator.log({"model_mse_unaccumulated": avg_model_mse}, step=global_step)
                accelerator.log({"ref_mse_unaccumulated": avg_ref_mse}, step=global_step)
                accelerator.log({"implicit_acc_accumulated": implicit_acc_accumulated}, step=global_step)
                train_loss = 0.0
                implicit_acc_accumulated = 0.0

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")
                        logger.info("Pretty sure saving/loading is fixed but proceed cautiously")

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            if args.train_method == 'dpo':
                logs["implicit_acc"] = avg_acc
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break


    # Create the pipeline using the trained modules and save it.
    # This will save to top level of output_dir instead of a checkpoint directory
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        pipeline = StableDiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            unet=unet,
            revision=args.revision,
        )
        pipeline.save_pretrained(args.output_dir)


    accelerator.end_training()


if __name__ == "__main__":
    main()
