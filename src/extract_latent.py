from diffusers import AutoencoderKL
import torch
import json
import os
from PIL import Image
from torchvision import transforms
import numpy as np
from tqdm import tqdm

resolution = (512,512) #sd 1.5
# resolution = (1024,1024) #sd xl
train_transforms = transforms.Compose(
    [
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)

def extract_latent(
    vae, 
    json_path, 
    save_dir="/common/users/hn315/datasets/FiFA-100k/latent", 
    data_dir="/common/users/hn315/datasets/FiFA-100k/data/train",
):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    with open(json_path, "r") as f:
        manifest = json.load(f) 
    f.close()    

    for item in tqdm(manifest[:]):
        image_0 = Image.open(os.path.join(data_dir, item["image_0_basename"]))
        image_1 = Image.open(os.path.join(data_dir, item["image_1_basename"]))
        image_0 = train_transforms(image_0)
        image_1 = train_transforms(image_1)

        image_0_latent = vae.encode(image_0.unsqueeze(0).to(vae.device)).latent_dist.sample() * vae.config.scaling_factor
        image_1_latent = vae.encode(image_1.unsqueeze(0).to(vae.device)).latent_dist.sample() * vae.config.scaling_factor

        image_0_latent = image_0_latent.detach().cpu().numpy()
        image_1_latent = image_1_latent.detach().cpu().numpy()

        save_path_0 = os.path.join(save_dir, item["image_0_basename"].replace(".jpg", ".npz"))
        save_path_1 = os.path.join(save_dir, item["image_1_basename"].replace(".jpg", ".npz"))

        np.savez_compressed(save_path_0, image_0_latent)
        np.savez_compressed(save_path_1, image_1_latent)

if __name__ == "__main__":
    vae = AutoencoderKL.from_pretrained("./checkpoints/sd15", subfolder="vae").to("cuda")
    extract_latent(vae, "datasets/manifest/fifa_10k.json")

        