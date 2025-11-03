import os
import io
from PIL import Image
import concurrent.futures
from datasets import load_dataset
import json
from tqdm import tqdm

def process_example(args):
    idx, example, save_dir = args
    # print(example)
    caption = example["original_prompt.txt"]
    jpg_0 = example["jpg_0.jpg"]
    jpg_1 = example["jpg_1.jpg"]
    label_0 = int(float(example["label_0.txt"]))
    label_1 = int(float(example["label_1.txt"]))

    refer_id = 0 if label_0 > label_1 else 1
    # image_0 = Image.open(io.BytesIO(jpg_0)).convert("RGB")
    # image_1 = Image.open(io.BytesIO(jpg_1)).convert("RGB")
    image_0 = jpg_0.convert("RGB")
    image_1 = jpg_1.convert("RGB")

    # Fix tied label after refer_id is determined
    if label_0 == label_1:
        print(f"label_0 == label_1: {label_0} {label_1}")
        label_0 = 0
        label_1 = 1

    image_0_basename = f"{idx:06d}_{label_0}.jpg"
    image_1_basename = f"{idx:06d}_{label_1}.jpg"

    image_0_path = os.path.join(save_dir, image_0_basename)
    image_1_path = os.path.join(save_dir, image_1_basename)
    # print(image_0_path)
    image_0.save(image_0_path)
    image_1.save(image_1_path)
    return {
        "caption": caption,
        "image_0_basename": image_0_basename,
        "image_1_basename": image_1_basename,
        "refer_id": refer_id,
    }

def read_dataset(data_path, split="train", save_dir="/common/users/hn315/datasets/FiFA-100k/data", n_samples=100000):
    res = []
    dataset = load_dataset(data_path, split=split)

    # Make sure local save dir for this split exists
    split_dir = os.path.join(save_dir, split)
    if not os.path.exists(split_dir):
        os.makedirs(split_dir, exist_ok=True)

    # Prepare argument tuples
    args_iter = ((idx, example, split_dir) for idx, example in enumerate(dataset))

    max_workers = min(32, os.cpu_count() or 1)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for out in tqdm(executor.map(process_example, args_iter, chunksize=32)):
            res.append(out)

    with open("./datasets/manifest/fifa_new.json", "w") as f:
        json.dump(res, f)


def split_semi_supervised_dataset_by_ratio_and_seed(path_to_manifest_json, save_dir, labeled_ratio=0.5, seed=42):
    random.seed(seed)
    with open(path_to_manifest_json, "r") as f:
        dataset = json.load(f)

    random.shuffle(dataset)
    labeled_dataset = dataset[:int(len(dataset) * labeled_ratio)]
    unlabeled_dataset = dataset[int(len(dataset) * labeled_ratio):]

    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "labeled.json"), "w") as f:
        json.dump(labeled_dataset, f)
    with open(os.path.join(save_dir, "unlabeled.json"), "w") as f:
        json.dump(unlabeled_dataset, f)

    return labeled_dataset, unlabeled_dataset

if __name__ == "__main__":
    read_dataset(
        data_path="datasets/FiFA-new",
        split="train",
        save_dir="datasets/FiFA-new/data/",
    )