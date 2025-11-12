from shlex import join
import numpy as np
import json
import os
import torch
from tqdm import tqdm
import random
import shutil
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

# from diffusers import StableDiffusionXLPipeline
# annotation = "./datasets/manifest_high_margin/5k_high_margin.json"
# output_dirs = "./datasets/precomputed_prompt_embeds/5k_high_margin_sorted"
# if not os.path.exists(output_dirs):
#     os.makedirs(output_dirs, exist_ok=True)

# model_dir="/common/users/hn315/checkpoints/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b"
# pipeline = StableDiffusionXLPipeline.from_pretrained(model_dir, unet=None, vae=None, torch_dtype=torch.bfloat16).to("cuda")

# objs = json.load(open(annotation, "r"))
# for item in tqdm(objs):
#     prompt = item["caption"]
#     basename = item["image_0_basename"].split(".")[0]
#     output_path = os.path.join(output_dirs, f"{basename}.pt")
#     prompt_embeds, _, pooled_prompt_embeds, _ = pipeline.encode_prompt(
#         prompt=prompt,
#         negative_prompt="",
#         device="cuda",
#         num_images_per_prompt=1,
#         do_classifier_free_guidance=False,
#     )
#     torch.save(
#         (prompt_embeds.detach().cpu(), pooled_prompt_embeds.detach().cpu()),
#         output_path
#     )
# # breakpoint()

# label = json.load(open("datasets/manifest_high_margin/from_5k_high_margin_25_75/labeled.json", "r"))
# pseudo_unlabeled = json.load(open("datasets/manifest_high_margin/from_5k_high_margin_25_75/pseudo_unlabeled_qwen.json", "r"))
# total = label + pseudo_unlabeled
# res = []
# for item in total:
#     res.append(item)

# json.dump(res, open("datasets/manifest_high_margin/labeled_and_pseudo_unlabeled_qwen.json", "w"))

# unlabel = json.load(open("datasets/manifest_high_margin/from_20k_high_margin_25_75/labeled.json", "r"))
# pseudo_unlabel = json.load(open("datasets/manifest_high_margin/from_20k_high_margin_25_75/pseudo_labeled_qwen.json", "r"))

# res=[]
# check = {}
# for item in unlabel:
#     check[item["image_0_basename"]] = item["refer_id"]

# for item in pseudo_unlabel:
#     check_name = item["image_0_basename"]

#     pseudo_label = float(item["refer_id"])
#     if pseudo_label != check[check_name]:
#         res.append(item)
# print(len(res) / len(pseudo_unlabel))

# # sd = ""
# # for item in res:
# #     sd += f"{item['caption']}\n"

# # with open("check_qwen.txt", "w") as f:
# #     f.write(sd)

# n_samples = 1632
# noise = torch.randn(n_samples, 4, 64, 64, device="cuda", generator=torch.Generator(device="cuda").manual_seed(999))
# torch.save(noise.detach().cpu(), "datasets/partiprompts_noise_sd15.pt")

# _dirs = {
#     "./datasets/manifest_high_margin/from_10k_high_margin_25_75": "./datasets/manifest_high_margin/10k_labaled_pseudo_unlabled.json",
#     "./datasets/manifest_high_margin/from_20k_high_margin_25_75": "./datasets/manifest_high_margin/20k_labaled_pseudo_unlabled.json",
#     "./datasets/manifest_high_margin/from_50k_high_margin_25_75": "./datasets/manifest_high_margin/50k_labaled_pseudo_unlabled.json",
#     "./datasets/manifest_high_margin/from_100k_high_margin_25_75": "./datasets/manifest_high_margin/100k_labaled_pseudo_unlabled.json",
# }

# for folder, save_path in _dirs.items():

#     label = json.load(open(os.path.join(folder, "labeled.json"), "r"))
#     pseudo_unlabeled = json.load(open(os.path.join(folder, "pseudo_unlabeled_qwen.json"), "r"))

#     total = label + pseudo_unlabeled
#     res = []
#     for item in total:
#         res.append(item)

#     print(len(res))
#     json.dump(res, open(save_path, "w"))

    
# pseudo_of_label = "datasets/manifest_high_margin/from_5k_high_margin_25_75/labeled.json"

# pseudo_of_unlabel_list = [
#     # "datasets/manifest_high_margin/from_5k_high_margin_25_75/pseudo_unlabeled_qwen.json",
#     "datasets/manifest_high_margin/from_10k_high_margin_25_75/pseudo_unlabeled_qwen.json",
#     # "datasets/manifest_high_margin/from_20k_high_margin_25_75/pseudo_unlabeled_qwen.json",
#     "datasets/manifest_high_margin/from_50k_high_margin_25_75/pseudo_unlabeled_qwen.json",
#     "datasets/manifest_high_margin/from_100k_high_margin_25_75/pseudo_unlabeled_qwen.json",
# ]
# label = json.load(open(pseudo_of_label, "r"))

# for pseudo_file in pseudo_of_unlabel_list:
#     unlabel = json.load(open(pseudo_file, "r"))
#     total = label + unlabel
#     res = []
#     for item in total:
#         res.append(item)
#     print(len(res))
    
#     n_samples = "10k" if "10k" in pseudo_file else "20k" if "20k" in pseudo_file else "50k" if "50k" in pseudo_file else "100k"
#     basename = f"ablate_scale_pseudo_{n_samples}_pseudo_of_unlabel.json"
#     json.dump(res, open(os.path.join("datasets/manifest_dpo_ablate2", basename), "w"))


# n_images = 200
# image_folders = {
#     "sdxl_base": "main_results/partiprompts/sdxl_base",
#     "sdxl_dpo_fifa5k": "main_results/partiprompts/sdxl_dpo_fifa5k_high_margin_100_v2",
#     "sdxl_dpo_25_75": "main_results/partiprompts/sdxl_dpo_fifa5k_high_margin_25",
#     "ours_25_75": "main_results/partiprompts/sdxl_dr_ots_high_margin_pseudo_qwen"
# }

# save_folder = "./main_results/qualitatives"
# if not os.path.exists(save_folder):
#     os.makedirs(save_folder)

# paritprompt = "./datasets/eval_prompts/partiprompts.json"
# objs = json.load(open(paritprompt))
# breakpoint()
# index = set([random.randint(0, len(objs) - 1) for _ in range(n_images)])

# for key in image_folders.keys():
#     save_dir = os.path.join(save_folder, key)
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)

# metadata = []
# for id in index:
#     prompt = objs[id]
#     image_name = f"image_{id}.jpg"
#     metadata.append({
#         "prompt": prompt,
#         "image_name": image_name,
#     })

#     for method, image_folder in image_folders.items():
#         source_image = os.path.join(image_folder, image_name)
#         dest_image = os.path.join(save_folder, method, image_name)
#         shutil.copy(source_image, dest_image)
# for key in image_folders:
#     print(len(os.listdir(os.path.join(save_folder, key))))

# f= open(os.path.join(save_folder, "metadata.json"), "w")
# json.dump(metadata, f)
# f.close()

# N = [5000, 10000, 20000, 50000]
# objs = json.load(open("datasets/manifest_hpdv2/100k.json", "r"))
# for n in N:
#     objs = objs[:n]
#     if n == 5000: name = "5k"
#     elif n == 10000: name = "10k"
#     elif n == 20000: name = "20k"
#     elif n == 50000: name = "50k"
#     json.dump(objs, open(f"datasets/manifest_hpdv2/{name}_hpdv2.json", "w"))

save_dir = "./datasets/hpdv2_5k/data/train"
data_dir = "./datasets/hpdv2_sorted/data/train"
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
objs = json.load(open("datasets/manifest_hpdv2/5k.json", "r"))
for item in tqdm(objs):
    image_0 = os.path.join(save_dir, item["image_0_basename"])
    image_1 = os.path.join(save_dir, item["image_1_basename"])
    if not os.path.exists(image_0):
        print(f"Image {image_0} does not exist")
    if not os.path.exists(image_1):
        print(f"Image {image_1} does not exist")
    # shutil.copy(image_0, os.path.join(save_dir, item["image_0_basename"]))
    # shutil.copy(image_1, os.path.join(save_dir, item["image_1_basename"]))



