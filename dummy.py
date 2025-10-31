import numpy as np
import json
import os
import torch
from tqdm import tqdm

# # let's create ground truth binary data
# np.random.seed(42)
# ground_truth = np.random.randint(0, 2, size=(100000))

# # need to set aside a good-quality set. We know this set is good and the other set is DUBIOUS,
# #   meaning that some of it could be bad, but we dont know which.
# percent_good = 0.5

# # now we corrupt the dubious set (flip or not flip). if we want a 20% of the full data to be bad, we need to flip 40% of the dubious set.
# percent_flip = 0.4

# good_set = ground_truth[: int(percent_good * len(ground_truth))].copy()
# dubious_set = ground_truth[int(percent_good * len(ground_truth)):].copy()

# n_flip = int(percent_flip * len(dubious_set))
# flip_indices = np.random.sample(len(dubious_set)) < percent_flip
# print(dubious_set)
# dubious_set[flip_indices] = 1 - dubious_set[flip_indices]
# print(dubious_set)

# breakpoint()
# # final data
# data = np.concatenate([good_set, dubious_set])

# # check
# len(ground_truth), len(data), (data != ground_truth).mean()

# objs = json.load(open("datasets/eval_prompts/pickapic_test.json", "r"))
# n_noises = len(objs)
# seed = 999
# noise = torch.randn(n_noises, 4, 64, 64, device="cuda", generator=torch.Generator(device="cuda").manual_seed(seed))
# torch.save(noise.cpu(), "datasets/pickapic_test_noise.pt")

# prompts = []
# for prompt, _ in objs.items():
#     prompts.append(prompt)

# json.dump(prompts, open("datasets/eval_prompts/pickapic_test_prompts.json", "w"))
# good = []
# dub = []

# n_sampes = int(0.5 * len(objs))
# print(n_sampes)
# for i, obj in enumerate(objs[:]):
#     if i < n_sampes:
#         good.append(obj)
#     else:
#         dub.append(obj)

# save_dir = "datasets/manifest/from_100k"
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir, exist_ok=True)
# json.dump(good, open(os.path.join(save_dir, "good.json"), "w"))
# json.dump(dub, open(os.path.join(save_dir, "dubious.json"), "w"))

# good_manifest = json.load(open("datasets/manifest/from_5k/labeled.json", "r"))
# dubious_manifest = json.load(open("datasets/manifest/from_5k/unlabeled.json", "r"))

# total = good_manifest + dubious_manifest
# image_dir = "datasets/FiFA-100k/data/train"
# not_found = []
# for item in tqdm(total):
#     image_0 = os.path.join(image_dir, item["image_0_basename"])
#     image_1 = os.path.join(image_dir, item["image_1_basename"])
#     if not os.path.exists(image_0):
#         not_found.append(item)
#     if not os.path.exists(image_1):
#         not_found.append(item)


# if len(not_found) == 0:
#     with open("datasets/manifest/5k_high_margin.json", "w") as f:
#         json.dump(total, f)
#     f.close()

# print(len(total))

from diffusers import StableDiffusionXLPipeline
annotation = "./datasets/manifest/5k_high_margin.json"
output_dirs = "./datasets/precomputed_prompt_embeds/5k_high_margin"
if not os.path.exists(output_dirs):
    os.makedirs(output_dirs, exist_ok=True)

model_dir="/common/users/hn315/checkpoints/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b"
pipeline = StableDiffusionXLPipeline.from_pretrained(model_dir, unet=None, vae=None, torch_dtype=torch.bfloat16).to("cuda")

objs = json.load(open(annotation, "r"))
for item in tqdm(objs):
    prompt = item["caption"]
    basename = item["image_0_basename"].split(".")[0]
    output_path = os.path.join(output_dirs, f"{basename}.pt")
    prompt_embeds, _, pooled_prompt_embeds, _ = pipeline.encode_prompt(
        prompt=prompt,
        negative_prompt="",
        device="cuda",
        num_images_per_prompt=1,
        do_classifier_free_guidance=False,
    )
    torch.save(
        (prompt_embeds.detach().cpu(), pooled_prompt_embeds.detach().cpu()),
        output_path
    )
# breakpoint()

