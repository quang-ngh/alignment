

import argparse
import io
import logging
import math
import os
import random
import json
import time

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
# from diffusers.optimization import get_scheduler
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel, StableDiffusionXLPipeline
from diffusers.schedulers import DPMSolverMultistepScheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from sklearn.metrics import precision_score, recall_score, f1_score
import copy
from src.utils import *
from utils import get_scheduler


# flags related to compilation
# torch._inductor.config.conv_1x1_as_mm = True
# torch._inductor.config.coordinate_descent_tuning = True
# torch._inductor.config.epilogue_fusion = False
# torch._inductor.config.coordinate_descent_check_all_directions = True
# torch._inductor.config.force_fuse_int_mm_with_mul = True
# torch._inductor.config.use_mixed_mm = True

# from accelerate.utils import TorchDynamoPlugin
# Configure the compilation backend
# dynamo_plugin = TorchDynamoPlugin(
#     use_regional_compilation=True,
#     backend="inductor",  # Options: "inductor", "aot_eager", "aot_nvfuser", etc.
#     mode="max-autotune",      # Options: "default", "reduce-overhead", "max-autotune"
#     fullgraph=True,
#     dynamic=False
# )

def safe_precision_score(y_true, y_pred):
    """
    Wrapper for precision_score that handles edge cases where metrics would be undefined.
    Returns 0.0 for edge cases (all predictions 0 or all labels 0).
    
    Args:
        y_true: True labels (numpy array)
        y_pred: Predicted labels (numpy array)
    
    Returns:
        float: Precision score, or 0.0 for edge cases
    """
    num_pos_predictions = y_pred.sum()
    num_pos_labels = y_true.sum()
    
    if num_pos_predictions == 0 or num_pos_labels == 0:
        return 0.0
    
    try:
        return float(precision_score(y_true, y_pred, zero_division=0))
    except (ValueError, ZeroDivisionError):
        return 0.0


def safe_recall_score(y_true, y_pred):
    """
    Wrapper for recall_score that handles edge cases where metrics would be undefined.
    Returns 0.0 for edge cases (all predictions 0 or all labels 0).
    
    Args:
        y_true: True labels (numpy array)
        y_pred: Predicted labels (numpy array)
    
    Returns:
        float: Recall score, or 0.0 for edge cases
    """
    num_pos_predictions = y_pred.sum()
    num_pos_labels = y_true.sum()
    
    if num_pos_predictions == 0 or num_pos_labels == 0:
        return 0.0
    
    try:
        return float(recall_score(y_true, y_pred, zero_division=0))
    except (ValueError, ZeroDivisionError):
        return 0.0


def safe_f1_score(y_true, y_pred):
    """
    Wrapper for f1_score that handles edge cases where metrics would be undefined.
    Returns 0.0 for edge cases (all predictions 0 or all labels 0).
    
    Args:
        y_true: True labels (numpy array)
        y_pred: Predicted labels (numpy array)
    
    Returns:
        float: F1 score, or 0.0 for edge cases
    """
    num_pos_predictions = y_pred.sum()
    num_pos_labels = y_true.sum()
    
    if num_pos_predictions == 0 or num_pos_labels == 0:
        return 0.0
    
    try:
        return float(f1_score(y_true, y_pred, zero_division=0))
    except (ValueError, ZeroDivisionError):
        return 0.0

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
        "--manifest", type=str, default='', help="Manifest"
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
        "--hard_pseudo_label", default=False, action="store_true", help="Use hard pseudo-label (0/1) instead of soft pseudo-label (0-1)",
    )
    parser.add_argument(
        "--curriculum", choices=['none', 'linear', 'quadratic'], default='none', help="Curriculum",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.0, help="Threshold for pseudo-label. default 0 leads to no masking."
    )
    parser.add_argument(
        "--no-dr", action="store_true", help="No DR"
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
        # dynamo_plugin=dynamo_plugin,
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
    # Compile
    unet = unet.to(memory_format=torch.channels_last)
    unet.compile_repeated_blocks(mode='max-autotune', fullgraph=True)

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

    resolution = (512,512) # sd1.5
    dataset = BaseDataset(manifest=args.manifest, image_dir=args.train_data_dir, resolution=resolution)

    # scale learning rate
    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * 2 * accelerator.num_processes
            # double the batch size for generated data
        )

    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # DataLoaders creation:
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        drop_last=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )
    ##### END BIG OLD DATASET BLOCK #####
    
    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    # lr_scheduler = get_scheduler(
    #     args.lr_scheduler,
    #     optimizer=optimizer,
    #     num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
    #     num_training_steps=args.max_train_steps * accelerator.num_processes,
    # )
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        step_rules=args.lr_scheduler_rule,
        min_lr=args.min_lr,
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
    
    #### START ACCELERATOR PREP ####
    unet, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, dataloader, lr_scheduler
    )

    pipe = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        unet=accelerator.unwrap_model(unet),
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        revision=args.revision,
        safety_checker=None,
        torch_dtype=weight_dtype,
    )

    # pipe.set_progress_bar_config(disable=True)
    # pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    # pipe.unet.to(accelerator.device, dtype=weight_dtype)
    # pipe.vae.to(accelerator.device, dtype=weight_dtype)
    # pipe.text_encoder.to(accelerator.device, dtype=weight_dtype)
    ### END ACCELERATOR PREP ###
    
    
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(dataloader) / args.gradient_accumulation_steps)
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
    total_batch_size = args.train_batch_size * 2 * accelerator.num_processes * args.gradient_accumulation_steps # double the batch size for generated data

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size} (x2 for number of images)")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size} (x2 for number of images)")
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
        # Initialize accumulated metrics dictionary - easy to add new metrics here!
        metrics_accumulated = {
            "train_loss": 0.0,
            "model_mse_unaccumulated": 0.0,
            "ref_mse_unaccumulated": 0.0,
            "implicit_acc_accumulated": 0.0,
            "labeled_margin_accumulated": 0.0,
            "unlabeled_margin_accumulated": 0.0,
            "image_generation_time_accumulated": 0.0,
        }
        for step, batch in enumerate(dataloader):
            # Skip steps until we reach the resumed step
            with accelerator.accumulate(unet):

                # Get data from good dataset
                prompts = batch["prompt"]
                labeled_images_0 = batch["image_0"]
                labeled_images_1 = batch["image_1"]
                labeled_prefs = batch["preference"].to(accelerator.device, dtype=weight_dtype)

                bsz = len(prompts)

                # Get the text embedding for conditioning
                input_ids = tokenizer(prompts, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt").input_ids
                with torch.no_grad():
                    encoder_hidden_states = text_encoder(input_ids.to(accelerator.device))[0]
                
                    labeled_latents_0 = vae.encode(labeled_images_0.to(accelerator.device, dtype=weight_dtype)).latent_dist.sample() * vae.config.scaling_factor
                    labeled_latents_1 = vae.encode(labeled_images_1.to(accelerator.device, dtype=weight_dtype)).latent_dist.sample() * vae.config.scaling_factor    

                    # Time the image generation
                    gen_start_time = time.time()
                    # should do nothing if pass by reference
                    # pipe.unet = accelerator.unwrap_model(unet)
                    with torch.autocast("cuda", dtype=weight_dtype):
                        # duplicate for image0 and image1
                        unlabeled_latents = pipe(prompt_embeds=encoder_hidden_states.repeat(2, 1, 1), num_inference_steps=20, output_type="latent").images.to(weight_dtype)
                        unlabeled_latents_0, unlabeled_latents_1 = unlabeled_latents.chunk(2)
                    gen_end_time = time.time()
                    gen_time = gen_end_time - gen_start_time

                    # decoded = vae.decode(unlabeled_latents_0 / vae.config.scaling_factor).sample
                    # # Denormalize from [-1, 1] to [0, 1]
                    # second_latents = vae.encode(decoded).latent_dist.sample() * vae.config.scaling_factor
                    # # debug: check if latent from decode is the same as the latent from the model
                    # # debug what is the scaling vae.config.scaling_factor
                    # breakpoint()
                    
                latents_0 = torch.cat(
                    [
                        labeled_latents_0,
                        unlabeled_latents_0,
                    ],
                    dim=0
                )
                latents_1 = torch.cat(
                    [
                        labeled_latents_1,
                        unlabeled_latents_1,
                    ],
                    dim=0
                )

                latents = torch.cat(
                    [latents_0, latents_1],
                    dim=0
                ).to(accelerator.device, dtype=weight_dtype)

                noise = torch.randn_like(latents)
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device)
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

                assert noise_scheduler.config.prediction_type == "epsilon"
                target = noise

                # Make the prediction from the model we're learning
                model_batch_args = (noisy_latents,
                                    timesteps,
                                    encoder_hidden_states.repeat(4, 1, 1))   #   duplicate for labeled and unlabeled, image0 and image1
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
                model_losses_0, model_losses_1 = model_losses.chunk(2)
                # below for logging purposes
                raw_model_loss = 0.5 * (model_losses_0.mean() + model_losses_1.mean())
                
                # model_diff = model_losses_0 - model_losses_1 # These are both LBS (as is t)
                # (e_0 - e_theta(x_0)^2)
                # (e_1 - e_theta(x_1)^2)
                model_diff = model_losses_1 - model_losses_0

                with torch.no_grad(): # Get the reference policy (unet) prediction
                    ref_pred = ref_unet(
                                *model_batch_args,
                                    added_cond_kwargs = added_cond_kwargs
                                    ).sample.detach()
                    ref_losses = (ref_pred - target).pow(2).mean(dim=[1,2,3])
                    ref_losses_0, ref_losses_1 = ref_losses.chunk(2)
                    ref_diff = ref_losses_1 - ref_losses_0
                    raw_ref_loss = ref_losses.mean()

                scale_term = -0.5 * args.beta_dpo
                logits = scale_term * (model_diff - ref_diff)
                labeled_logits, unlabeled_logits = logits[:bsz], logits[bsz:]

                soft_pseudo_prefs = torch.sigmoid(logits)
                hard_pseudo_prefs = (soft_pseudo_prefs > 0.5).float()
                labeled_pseudo_prefs, unlabeled_pseudo_prefs = hard_pseudo_prefs[:bsz], hard_pseudo_prefs[bsz:]

                # for logging purposes
                margins = torch.abs(soft_pseudo_prefs * 2 - 1)
                labeled_margin = margins[:bsz].mean()
                unlabeled_margin = margins[bsz:].mean()

                masks = ((soft_pseudo_prefs - 0.5).abs() > args.threshold).float()
                labeled_masks = masks[:bsz]
                unlabeled_masks = masks[bsz:]

                implicit_preds = (logits > 0).float()
                labeled_preds, unlabeled_preds = implicit_preds[:bsz], implicit_preds[bsz:]
                labeled_acc = (labeled_preds == labeled_prefs).sum().float() / labeled_preds.size(0)

                from torch.nn.functional import binary_cross_entropy_with_logits

                if args.curriculum == 'linear':
                    curriculum_weight = global_step / args.max_train_steps
                elif args.curriculum == 'quadratic':
                    curriculum_weight = (global_step / args.max_train_steps) ** 2
                else:
                    curriculum_weight = 1.0

                pseudo_labeled_loss = (
                    binary_cross_entropy_with_logits(
                        labeled_logits,
                        labeled_pseudo_prefs,
                        reduction='none'
                    ) * labeled_masks
                ).mean()
                pseudo_unlabeled_loss = (
                    binary_cross_entropy_with_logits(
                        unlabeled_logits,
                        unlabeled_pseudo_prefs,
                        reduction='none'
                    ) * unlabeled_masks
                ).mean()
                labeled_loss = binary_cross_entropy_with_logits(labeled_logits, labeled_prefs, reduction='sum')
                if args.no_dr:
                    loss = (labeled_loss + pseudo_unlabeled_loss) / (bsz + bsz)
                else:
                    loss = (pseudo_labeled_loss + pseudo_unlabeled_loss) / (bsz + bsz) + curriculum_weight * (labeled_loss - pseudo_labeled_loss) / bsz
                #### END LOSS COMPUTATION ###
                    
                # Gather metrics across all processes (these are already averaged over batch)
                # Note: loss is already a scalar, no need to repeat
                avg_loss = accelerator.gather(loss.detach()).mean().item()
                avg_model_mse = accelerator.gather(raw_model_loss.detach()).mean().item()
                avg_ref_mse = accelerator.gather(raw_ref_loss.detach()).mean().item()
                avg_labeled_acc = accelerator.gather(labeled_acc.detach()).mean().item()
                avg_labeled_margin = accelerator.gather(labeled_margin.detach()).mean().item()
                avg_unlabeled_margin = accelerator.gather(unlabeled_margin.detach()).mean().item()
                # Convert generation time to tensor for gathering
                gen_time_tensor = torch.tensor(gen_time, device=accelerator.device, dtype=torch.float32)
                avg_gen_time = accelerator.gather(gen_time_tensor).mean().item()

                # Accumulate metrics across gradient accumulation steps
                # Add new metrics here by adding to the dictionary below
                current_metrics = {
                    "train_loss": avg_loss,
                    "model_mse_unaccumulated": avg_model_mse,
                    "ref_mse_unaccumulated": avg_ref_mse,
                    "implicit_acc_accumulated": avg_labeled_acc,
                    "labeled_margin_accumulated": avg_labeled_margin,
                    "unlabeled_margin_accumulated": avg_unlabeled_margin,
                    "image_generation_time_accumulated": avg_gen_time,
                }
                
                # Accumulate all metrics
                for key in metrics_accumulated:
                    metrics_accumulated[key] += current_metrics[key] / args.gradient_accumulation_steps

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
                
                # Log accumulated metrics (averaged across gradient accumulation steps)
                # Define which metrics go under "diagnostics/" prefix (for wandb/tensorboard grouping)
                diagnostic_metrics = {"labeled_margin_accumulated", "unlabeled_margin_accumulated", "image_generation_time_accumulated"}
                
                # Build log dict with appropriate prefixes
                log_dict = {}
                for key in metrics_accumulated:
                    log_key = f"diagnostics/{key}" if key in diagnostic_metrics else key
                    log_dict[log_key] = metrics_accumulated[key]
                log_dict["lr"] = lr_scheduler.get_last_lr()[0]
                
                accelerator.log(log_dict, step=global_step)
                
                # Reset accumulated metrics for next optimizer step
                metrics_accumulated = {key: 0.0 for key in metrics_accumulated}
                
                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")
                        logger.info("Pretty sure saving/loading is fixed but proceed cautiously")

                # # save some unlabeled image pairs
                if accelerator.is_main_process:
                    img_save_path = os.path.join(args.output_dir, "gen_images")
                    os.makedirs(img_save_path, exist_ok=True)
                    # Get first sample from batch and add batch dimension for decode
                    with torch.no_grad():
                        # Decode latents (pipeline outputs are already in the right format for VAE)
                        decoded_0 = vae.decode(unlabeled_latents_0[0:1] / vae.config.scaling_factor).sample
                        decoded_1 = vae.decode(unlabeled_latents_1[0:1] / vae.config.scaling_factor).sample
                        # Denormalize from [-1, 1] to [0, 1]
                        image_0 = (decoded_0 / 2 + 0.5).clamp(0, 1)
                        image_1 = (decoded_1 / 2 + 0.5).clamp(0, 1)
                        # Convert to numpy: CHW -> HWC, scale to 0-255, convert to uint8
                        image_0_np = (image_0[0].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                        image_1_np = (image_1[0].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                        # Determine win/lose based on pseudo preferences
                        # unlabeled_pseudo_prefs[0] == 0.0 means image_0 wins, image_1 loses
                        pref_ = unlabeled_pseudo_prefs[0].item()
                        # Get prompt text for filename (sanitize invalid filename characters)
                        prompt_text = prompts[0][:100] if isinstance(prompts[0], str) else str(prompts[0])[:100]
                        # Sanitize filename: remove invalid characters
                        invalid_chars = '<>:"/\\|?*'
                        prompt_safe = ''.join(c if c not in invalid_chars else '_' for c in prompt_text)
                        if pref_ == 1.0:
                            # image_1 wins, image_0 loses
                            win_image = image_1_np
                            lose_image = image_0_np
                            win_name = f"{global_step}_win_{prompt_safe}"
                            lose_name = f"{global_step}_lose_{prompt_safe}"
                        else:
                            # image_0 wins, image_1 loses
                            win_image = image_0_np
                            lose_image = image_1_np
                            win_name = f"{global_step}_win_{prompt_safe}"
                            lose_name = f"{global_step}_lose_{prompt_safe}"
                        # Convert to PIL Image and save with win/lose labels
                        Image.fromarray(win_image).save(os.path.join(img_save_path, f"{win_name}.png"))
                        Image.fromarray(lose_image).save(os.path.join(img_save_path, f"{lose_name}.png"))

            # Update progress bar with current step metrics (not accumulated)
            logs = {"step_loss": avg_loss, "lr": lr_scheduler.get_last_lr()[0], "image_generation_time": avg_gen_time}
            if args.train_method == 'dpo':
                logs["implicit_acc"] = avg_labeled_acc
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
