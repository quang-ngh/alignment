import torch
import numpy as np
import json
from transformers import AutoProcessor, AutoModel
from PIL import Image
import os 

DEVICE="cuda" if torch.cuda.is_available() else "cpu"

processor_name_or_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
model_pretrained_name_or_path = "yuvalkirstain/PickScore_v1"

processor = AutoProcessor.from_pretrained(processor_name_or_path, use_fast=True)
model = AutoModel.from_pretrained(model_pretrained_name_or_path).eval().to(DEVICE)

manifest = json.load(open("datasets/manifest_high_margin/from_20k_high_margin_25_75/labeled.json", "r"))
objs = json.load(open("datasets/manifest_high_margin/from_20k_high_margin_25_75/pseudo_labeled_qwen.json", "r"))

for item in manifest:
    image_path = os.path.join("datasets/FiFA-100k-sorted/data/train", item["image_0_basename"])
    prompt = item["caption"]
    image = Image.open(image_path)
    image = processor(images=image, return_tensors="pt").to(DEVICE)
    prompt = processor(text=prompt, return_tensors="pt").to(DEVICE)
    breakpoint()