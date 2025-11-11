from PIL import Image
import os
import json
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import random
from diffusers import AutoencoderKL

class BaseDataset(Dataset):
    def __init__(self, manifest, image_dir, resolution=(512,512), latent_dir=None, transform=None, prompt_dir=None, **kwargs):
        self.manifest = manifest
        self.image_dir = image_dir
        self.latent_dir = latent_dir
        if transform is not None:
            self.transform = transform
        else:
            random_crop = kwargs.get("random_crop", False)
            no_hflip = kwargs.get("no_hflip", False)

            self.transform = transforms.Compose(
                [
                    transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                    transforms.RandomCrop(resolution[0]) if random_crop else transforms.CenterCrop(resolution[0]),
                    transforms.Lambda(lambda x: x) if no_hflip else transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]
            )
        self.resolution = resolution
        self.data_list = []
        self.prompt_dir = prompt_dir
        self.init()

    def init(self):
        #   Parse the manifest
        with open(self.manifest, "r") as f:
            annotations = json.load(f)
        f.close()

        if self.latent_dir is not None:
            print("Using precomputed latents")
        else:
            print("No latent directory provided. Will not use precomputed latents.")

        #   Run the sanity check    
        for item in tqdm(annotations, desc="Running sanity check"):
            image_paths = [os.path.join(self.image_dir, item["image_0_basename"]), os.path.join(self.image_dir, item["image_1_basename"])]
            if self.prompt_dir is not None:
                image_paths.append(os.path.join(self.prompt_dir, item["image_0_basename"].replace(".jpg", ".pt")))
            if not all(os.path.exists(path) for path in image_paths):
                raise ValueError(f"Image file not found: {image_paths}")
            else:
                self.data_list.append(item)

            if self.latent_dir is not None:
                latent_paths = [os.path.join(self.latent_dir, item["image_0_basename"].replace(".jpg", ".npz")), os.path.join(self.latent_dir, item["image_1_basename"].replace(".jpg", ".npz"))]
                if not all(os.path.exists(path) for path in latent_paths):
                    raise ValueError(f"Latent file not found: {latent_paths}")
        
        if len(self.data_list) == 0:
            raise ValueError("No data found in the annotation file")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        prompt = item["caption"]

        #   Get the prefer label
        prefer_label = float(item["refer_id"]) # in case refer_id is a string

        #   Load the images
        image_0 = Image.open(os.path.join(self.image_dir, item["image_0_basename"]))
        image_1 = Image.open(os.path.join(self.image_dir, item["image_1_basename"]))
        image_0_tensor = self.transform(image_0)
        image_1_tensor = self.transform(image_1)
        win_image = image_0_tensor if prefer_label == 0 else image_1_tensor
        lose_image = image_1_tensor if prefer_label == 0 else image_0_tensor

        data_dict = {
            "prompt": prompt,
            "win_image": win_image,
            "lose_image": lose_image,
            "refer_id": prefer_label
            # "image_0": image_0_tensor,
            # "image_1": image_1_tensor,
            # "preference": prefer_label,
        }

        #   Load latent if available
        if self.latent_dir is not None:
            image_0_latent = np.load(os.path.join(self.latent_dir, item["image_0_basename"].replace(".jpg", ".npz")))["arr_0"]
            image_1_latent = np.load(os.path.join(self.latent_dir, item["image_1_basename"].replace(".jpg", ".npz")))["arr_0"]
            image_0_latent = torch.from_numpy(image_0_latent).squeeze(0)
            image_1_latent = torch.from_numpy(image_1_latent).squeeze(0)
            win_latent = image_0_latent if prefer_label == 0 else image_1_latent
            lose_latent = image_1_latent if prefer_label == 0 else image_0_latent
            data_dict["win_latent"] = win_latent
            data_dict["lose_latent"] = lose_latent
            # data_dict["latent_0"] = image_0_latent
            # data_dict["latent_1"] = image_1_latent
            data_dict["use_latent"] = True

        if self.prompt_dir is not None:
            prompt_path = os.path.join(self.prompt_dir, f"{item['image_0_basename'].split('.')[0]}.pt")
            prompt_embeds, pooled_prompt_embeds = torch.load(prompt_path, map_location="cpu")
            data_dict["prompt_embeds"] = prompt_embeds.squeeze(0)
            data_dict["pooled_prompt_embeds"] = pooled_prompt_embeds.squeeze(0)
        
        return data_dict

class DubiousDataset(BaseDataset):
    def __init__(self, manifest, image_dir, flip_percentage=0.4, resolution=(512,512), latent_dir=None, transform=None):
        super().__init__(manifest, image_dir)
        # self.data_list = []
        self.flip_label(flip_percentage)

    def flip_label(self, percentage=0.4):
        new_data_list = []
        
        num_to_flip = int(len(self.data_list) * percentage)
        indices = list(range(len(self.data_list)))
        random.shuffle(indices)
        flip_indices = set(indices[:num_to_flip])

        for i, item in enumerate(self.data_list):
            flipped_item = item.copy() 
            if i in flip_indices:
                flipped_item["refer_id"] = 1 - item["refer_id"]
                flipped_item["is_flip"] = True
            else:
                flipped_item["is_flip"] = False
            new_data_list.append(flipped_item)

        self.data_list = new_data_list
    
    def __getitem__(self, idx):
        return super().__getitem__(idx)

    def __len__(self):
        return len(self.data_list) 

class DRDataset(BaseDataset):
    def __init__(
        self, 
        manifest, 
        image_dir, 
        resolution=(512,512), 
        latent_dir=None, 
        transform=None,
        pseudo_label_path: str = None,
        prompt_dir: str = None,
    ):
        super().__init__(manifest, image_dir, resolution, latent_dir, transform)
        if pseudo_label_path is None:
            raise ValueError("Pseudo label path must be provided")

        self.pseudo_label = json.load(open(pseudo_label_path, "r"))

        self.pseudo_mapping = self._construct_pseudo_mapping()
        self.prompt_dir = prompt_dir

    def _construct_pseudo_mapping(self):
        pseudo_mapping = {}
        for item in self.pseudo_label:
            filename = item["image_0_basename"]
            pseudo_mapping[filename] = float(item["refer_id"])

        return pseudo_mapping

    def __getitem__(self, idx):
        item = self.data_list[idx]
        prompt = item["caption"]

        #   Get the prefer label
        prefer_label = float(item["refer_id"]) # in case refer_id is a string

        #   Load the images
        image_0 = Image.open(os.path.join(self.image_dir, item["image_0_basename"]))
        image_1 = Image.open(os.path.join(self.image_dir, item["image_1_basename"]))
        image_0_tensor = self.transform(image_0)
        image_1_tensor = self.transform(image_1)
        # win_image = image_0_tensor if prefer_label == 0 else image_1_tensor
        # lose_image = image_1_tensor if prefer_label == 0 else image_0_tensor
        pseudo_label = self.pseudo_mapping.get(item["image_0_basename"], None)

        if pseudo_label is None:
            pseudo_label=1-prefer_label
            warning_msg = f"Pseudo label not found for {item['image_0_basename']}. Using {pseudo_label} as pseudo label."
            print(warning_msg)
            # raise ValueError(f"Pseudo label not found for {item['image_0_basename']}")

        data_dict = {
            "prompt": prompt,
            # "win_image": win_image,
            # "lose_image": lose_image,
            "image_0": image_0_tensor,
            "image_1": image_1_tensor,
            "preference": prefer_label,
            "pseudo_preference": pseudo_label,
        }

        #   Load latent if available
        if self.latent_dir is not None:
            image_0_latent = np.load(os.path.join(self.latent_dir, item["image_0_basename"].replace(".jpg", ".npz")))["arr_0"]
            image_1_latent = np.load(os.path.join(self.latent_dir, item["image_1_basename"].replace(".jpg", ".npz")))["arr_0"]
            image_0_latent = torch.from_numpy(image_0_latent).squeeze(0)
            image_1_latent = torch.from_numpy(image_1_latent).squeeze(0)
            # win_latent = image_0_latent if prefer_label == 0 else image_1_latent
            # lose_latent = image_1_latent if prefer_label == 0 else image_0_latent
            # data_dict["win_latent"] = win_latent
            # data_dict["lose_latent"] = lose_latent
            data_dict["latent_0"] = image_0_latent
            data_dict["latent_1"] = image_1_latent
            data_dict["use_latent"] = True

        if self.prompt_dir is not None:
            prompt_path = os.path.join(self.prompt_dir, f"{item['image_0_basename'].split('.')[0]}.pt")
            prompt_embeds, pooled_prompt_embeds = torch.load(prompt_path, map_location="cpu")
            data_dict["prompt_embeds"] = prompt_embeds.squeeze(0)
            data_dict["pooled_prompt_embeds"] = pooled_prompt_embeds.squeeze(0)

        return data_dict



if __name__ == "__main__":
    from torch.utils.data import DataLoader

    good_dataset = BaseDataset(manifest="datasets/manifest/set1/good_from_10k.json", image_dir="datasets/FiFA-100k/data/train")
    dubious_dataset = DubiousDataset(manifest="datasets/manifest/set1/dub_from_10k.json", image_dir="datasets/FiFA-100k/data/train", flip_percentage=0.4)
    vae = AutoencoderKL.from_pretrained("./checkpoints/sd15", subfolder="vae", torch_dtype=torch.bfloat16).to("cuda")
    good_dataloader = DataLoader(good_dataset, batch_size=4, shuffle=True)
    dubious_dataloader = DataLoader(dubious_dataset, batch_size=4, shuffle=True)

    for good_batch, dubious_batch in zip(good_dataloader, dubious_dataloader):
        win_images = good_batch["win_image"].to(vae.device, dtype=torch.bfloat16)
        lose_images = good_batch["lose_image"].to(vae.device, dtype=torch.bfloat16)

        win_latents = vae.encode(win_images.to(vae.device)).latent_dist.sample() * vae.config.scaling_factor
        lose_latents = vae.encode(lose_images.to(vae.device)).latent_dist.sample() * vae.config.scaling_factor

        #   Decode
        win_images_decoded = (vae.decode(win_latents/vae.config.scaling_factor).sample / 2 + 0.5).clamp(0, 1)
        lose_images_decoded = (vae.decode(lose_latents/vae.config.scaling_factor).sample / 2 + 0.5).clamp(0, 1)
        
        # INSERT_YOUR_CODE

        # Save the first win image (decoded) and first lose image (decoded) from the batch
        # Ensure directory exists for saves
        import os
        os.makedirs('./generated_test_images', exist_ok=True)
        from torchvision.utils import make_grid, save_image

        # Make a grid with win images decoded
        win_grid = make_grid(win_images_decoded.cpu(), nrow=2)
        save_image(win_grid, './generated_test_images/win_images_decoded_grid.png')

        # Make a grid with lose images decoded
        lose_grid = make_grid(lose_images_decoded.cpu(), nrow=2)
        save_image(lose_grid, './generated_test_images/lose_images_decoded_grid.png')

        print(good_batch["prompt"])
        break
