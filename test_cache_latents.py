from src.pipe_sd15 import StableDiffusionPipeline
import torch
import numpy as np
from diffusers import AutoencoderKL
from PIL import Image

prompts = [
    "a photo of a cat",
]

pipeline = StableDiffusionPipeline.from_pretrained("./checkpoints/sd15", torch_dtype=torch.float16).to("cuda")
num_inference_steps = 20
guidance_scale = 7.5

gen_kwargs = {
    "num_inference_steps": num_inference_steps,
    "guidance_scale": guidance_scale,
    "return_all_latents": True,
    "return_dict": False,
}
output, is_nsfw, steps_latents = pipeline(
    prompts,
    **gen_kwargs,
)


#   Decode latents
decoded_images = []
for latent in steps_latents:
    if len(latent.shape) == 3:
        latent = latent.unsqueeze(0)
    decoded_image = pipeline.vae.decode(latent.to(device=pipeline.vae.device, dtype=pipeline.vae.dtype) / pipeline.vae.config.scaling_factor).sample
    decoded_image = (decoded_image / 2 + 0.5).clamp(0, 1).permute(0,2,3,1)
    decoded_image = (decoded_image[0] * 255).detach().cpu().numpy().astype(np.uint8)
    decoded_image = Image.fromarray(decoded_image)
    decoded_images.append(decoded_image)

    # INSERT_YOUR_CODE
    

    # decoded_images is a list of PIL Image objects (single-channel or 3-channel)
    # Ensure images are RGB for GIFs
decoded_images[0].save(
    "generation.gif",
    save_all=True,
    append_images=decoded_images[1:],
    duration=50,   # duration(ms) per frame, adjust if desired
    loop=1
)


