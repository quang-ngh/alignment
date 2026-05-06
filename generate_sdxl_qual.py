import argparse
import glob
import json
from pathlib import Path

import torch
from diffusers import (
    AutoencoderKL,
    DPMSolverMultistepScheduler,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
from tqdm import tqdm


DEFAULT_PROMPT_GLOB = "datasets/eval_prompts/hpsv2_*.json"
DEFAULT_OUTPUT_ROOT = "qual"
DEFAULT_VAE_ROOT = "checkpoints"
DEFAULT_VAE_SUBFOLDER = "sdxl_vae"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate qualitative HPSv2 images with SDXL and a trained UNet checkpoint."
    )
    parser.add_argument("--model_path", required=True, help="Base SDXL model path.")
    parser.add_argument(
        "--unet_path",
        default="",
        help="Path to the trained UNet checkpoint. Leave empty to use the base SDXL UNet.",
    )
    parser.add_argument("--seed", type=int, required=True, help="Base seed for generation.")
    parser.add_argument(
        "--prompt_glob",
        default=DEFAULT_PROMPT_GLOB,
        help="Glob pattern for HPSv2 prompt JSON files.",
    )
    parser.add_argument(
        "--output_root",
        default=DEFAULT_OUTPUT_ROOT,
        help="Root output directory. Final path is <output_root>/<unet_name>/<seed>/.",
    )
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for generation.")
    parser.add_argument(
        "--num_shards",
        type=int,
        default=1,
        help="Total number of prompt shards. Use 1 for single-process generation.",
    )
    parser.add_argument(
        "--shard_index",
        type=int,
        default=0,
        help="Zero-based shard index handled by this process.",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="Classifier-free guidance scale.",
    )
    parser.add_argument(
        "--inference_steps",
        type=int,
        default=20,
        help="Number of denoising steps.",
    )
    parser.add_argument("--height", type=int, default=1024, help="Output image height.")
    parser.add_argument("--width", type=int, default=1024, help="Output image width.")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device to run generation on.",
    )
    parser.add_argument(
        "--vae_root",
        default=DEFAULT_VAE_ROOT,
        help="Root path that contains the fixed SDXL VAE.",
    )
    parser.add_argument(
        "--vae_subfolder",
        default=DEFAULT_VAE_SUBFOLDER,
        help="Subfolder name for the fixed SDXL VAE.",
    )
    return parser.parse_args()


def resolve_unet_name(unet_path: str) -> str:
    if not unet_path:
        return "sdxl_base"

    path = Path(unet_path.rstrip("/"))
    name = path.name
    if name == "unet" and path.parent.name:
        name = path.parent.name
    return name or "sdxl_unet"


def get_prompt_files(prompt_glob: str) -> list[str]:
    prompt_files = sorted(glob.glob(prompt_glob))
    if not prompt_files:
        raise FileNotFoundError(f"No prompt files matched: {prompt_glob}")
    return prompt_files


def load_prompts(json_path: str) -> list[str]:
    with open(json_path, "r", encoding="utf-8") as handle:
        prompts = json.load(handle)

    if not isinstance(prompts, list) or not all(isinstance(prompt, str) for prompt in prompts):
        raise ValueError(f"Prompt file must contain a JSON list of strings: {json_path}")
    return prompts


def load_pipeline(args):
    torch_dtype = torch.float16 if args.device.startswith("cuda") else torch.float32

    if args.unet_path:
        unet = UNet2DConditionModel.from_pretrained(args.unet_path, torch_dtype=torch_dtype)
    else:
        unet = UNet2DConditionModel.from_pretrained(
            args.model_path,
            subfolder="unet",
            torch_dtype=torch_dtype,
        )

    scheduler = DPMSolverMultistepScheduler.from_pretrained(
        args.model_path,
        subfolder="scheduler",
    )
    vae = AutoencoderKL.from_pretrained(
        args.vae_root,
        subfolder=args.vae_subfolder,
        torch_dtype=torch_dtype,
    )

    pipeline = StableDiffusionXLPipeline.from_pretrained(
        args.model_path,
        unet=unet,
        vae=vae,
        scheduler=scheduler,
        torch_dtype=torch_dtype,
    ).to(args.device)
    pipeline.set_progress_bar_config(disable=True)

    if hasattr(pipeline, "watermarker"):
        pipeline.watermarker = None

    return pipeline


def build_generators(seed: int, prompt_indices: list[int], device: str):
    return [
        torch.Generator(device=device).manual_seed(seed + prompt_idx)
        for prompt_idx in prompt_indices
    ]


def validate_args(args):
    if args.batch_size < 1:
        raise ValueError("--batch_size must be at least 1")
    if args.num_shards < 1:
        raise ValueError("--num_shards must be at least 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard_index must be in [0, --num_shards)")


def get_shard_bounds(total_items: int, shard_index: int, num_shards: int) -> tuple[int, int]:
    base = total_items // num_shards
    remainder = total_items % num_shards

    start_idx = shard_index * base + min(shard_index, remainder)
    shard_size = base + (1 if shard_index < remainder else 0)
    end_idx = start_idx + shard_size
    return start_idx, end_idx


def write_metadata(output_dir: Path, prompt_files: list[str], args):
    metadata = {
        "model_path": args.model_path,
        "unet_path": args.unet_path,
        "unet_name": resolve_unet_name(args.unet_path),
        "seed": args.seed,
        "prompt_files": prompt_files,
        "guidance_scale": args.guidance_scale,
        "inference_steps": args.inference_steps,
        "batch_size": args.batch_size,
        "height": args.height,
        "width": args.width,
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


@torch.inference_mode()
def generate_prompt_file(
    pipeline: StableDiffusionXLPipeline,
    prompts: list[str],
    category_dir: Path,
    seed: int,
    batch_size: int,
    guidance_scale: float,
    inference_steps: int,
    height: int,
    width: int,
    start_idx: int,
    end_idx: int,
):
    category_dir.mkdir(parents=True, exist_ok=True)

    for batch_start_idx in tqdm(
        range(start_idx, end_idx, batch_size),
        desc=category_dir.name,
    ):
        batch_end_idx = min(batch_start_idx + batch_size, end_idx)
        batch_prompts = prompts[batch_start_idx:batch_end_idx]
        pending_indices = []
        pending_prompts = []

        for offset, prompt in enumerate(batch_prompts):
            prompt_idx = batch_start_idx + offset
            image_path = category_dir / f"image_{prompt_idx:04d}.jpg"
            if image_path.exists():
                continue
            pending_indices.append(prompt_idx)
            pending_prompts.append(prompt)

        if not pending_prompts:
            continue

        generators = build_generators(
            seed=seed,
            prompt_indices=pending_indices,
            device=str(pipeline.device),
        )
        images = pipeline(
            prompt=pending_prompts,
            num_inference_steps=inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            generator=generators,
            output_type="pil",
        ).images

        for prompt_idx, image in zip(pending_indices, images):
            image.save(category_dir / f"image_{prompt_idx:04d}.jpg")


def main():
    args = parse_args()
    validate_args(args)

    unet_name = resolve_unet_name(args.unet_path)
    output_dir = Path(args.output_root) / unet_name / str(args.seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt_files = get_prompt_files(args.prompt_glob)
    if args.shard_index == 0:
        write_metadata(output_dir, prompt_files, args)

    print(f"Using base model: {args.model_path}")
    print(f"Using UNet: {args.unet_path or 'base UNet'}")
    print(f"Using seed: {args.seed}")
    print(f"Using shard: {args.shard_index + 1}/{args.num_shards}")
    print(f"Saving images under: {output_dir}")

    pipeline = load_pipeline(args)

    for prompt_file in prompt_files:
        prompt_path = Path(prompt_file)
        prompt_name = prompt_path.stem
        prompt_output_dir = output_dir / prompt_name
        prompts = load_prompts(prompt_file)
        start_idx, end_idx = get_shard_bounds(
            total_items=len(prompts),
            shard_index=args.shard_index,
            num_shards=args.num_shards,
        )

        print(
            f"Generating prompts [{start_idx}, {end_idx}) out of {len(prompts)} "
            f"from {prompt_path.name}"
        )
        if start_idx == end_idx:
            continue
        generate_prompt_file(
            pipeline=pipeline,
            prompts=prompts,
            category_dir=prompt_output_dir,
            seed=args.seed,
            batch_size=args.batch_size,
            guidance_scale=args.guidance_scale,
            inference_steps=args.inference_steps,
            height=args.height,
            width=args.width,
            start_idx=start_idx,
            end_idx=end_idx,
        )


if __name__ == "__main__":
    main()
