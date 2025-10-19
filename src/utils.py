import os
import io
import random
from PIL import Image
import concurrent.futures
from datasets import load_dataset
import json

def process_example(args):
    """Process a single example from the dataset."""
    idx, example, save_dir = args

    caption = example["caption"]
    jpg_0 = example["jpg_0"]
    jpg_1 = example["jpg_1"]
    label_0 = example["label_0"]
    label_1 = example["label_1"]

    refer_id = 0 if label_0 > label_1 else 1
    image_0 = Image.open(io.BytesIO(jpg_0)).convert("RGB")
    image_1 = Image.open(io.BytesIO(jpg_1)).convert("RGB")

    # Fix tied label after refer_id is determined
    if label_0 == label_1:
        print(f"label_0 == label_1: {label_0} {label_1}")
        label_0 = 0
        label_1 = 1

    image_0_basename = f"{idx:06d}_{int(label_0)}.jpg"
    image_1_basename = f"{idx:06d}_{int(label_1)}.jpg"

    image_0_path = os.path.join(save_dir, image_0_basename)
    image_1_path = os.path.join(save_dir, image_1_basename)
    print(image_0_path)
    image_0.save(image_0_path)
    image_1.save(image_1_path)
    return {
        "caption": caption,
        "image_0_basename": image_0_basename,
        "image_1_basename": image_1_basename,
        "refer_id": refer_id,
    }

def load_and_process_dataset(data_path, split="train", save_dir="/data", n_samples=None):
    """Load dataset and process examples with proper directory structure."""
    res = []
    dataset = load_dataset(data_path)

    # Debug: Check dataset structure
    print(f"Dataset type: {type(dataset)}")
    print(f"Dataset keys: {dataset.keys()}")
    
    # Get the actual dataset (it might be a DatasetDict)
    if hasattr(dataset, 'keys'):
        # It's a DatasetDict, get the train split
        train_dataset = dataset[split]
    else:
        # It's already a Dataset
        train_dataset = dataset
        
    print(f"Train dataset type: {type(train_dataset)}")
    print(f"Train dataset features: {train_dataset.features}")
    print(f"First example: {train_dataset[0] if len(train_dataset) > 0 else 'Empty'}")

    # Create the exact directory structure from DATASET.MD
    # /data/FiFA-100k/data/train/ for images
    image_dir = os.path.join(save_dir, "FiFA-100k", "data", "train")
    if not os.path.exists(image_dir):
        os.makedirs(image_dir, exist_ok=True)

    # Create manifest directory structure
    manifest_dir = os.path.join(save_dir, "manifest")
    from_100k_dir = os.path.join(manifest_dir, "from_100k")
    if not os.path.exists(from_100k_dir):
        os.makedirs(from_100k_dir, exist_ok=True)

    # Prepare argument tuples
    args_iter = ((idx, example, image_dir) for idx, example in enumerate(train_dataset))

    max_workers = min(32, os.cpu_count() or 1)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for out in executor.map(process_example, args_iter, chunksize=32):
            res.append(out)

    return res, manifest_dir, from_100k_dir

def save_manifests(res, manifest_dir, from_100k_dir, split_ratio=0.5):
    """Save manifests with optional train/test split."""
    # Save main manifest
    with open(os.path.join(manifest_dir, "fifa_100k.json"), "w") as f:
        json.dump(res, f)

    # Split into good and dub manifests (50/50 split by default)
    random.shuffle(res)
    mid_point = int(len(res) * split_ratio)
    
    good_data = res[:mid_point]
    dub_data = res[mid_point:]
    
    # Save good manifest
    with open(os.path.join(from_100k_dir, "good_from_100k.json"), "w") as f:
        json.dump(good_data, f)
    
    # Save dub manifest  
    with open(os.path.join(from_100k_dir, "dub_from_100k.json"), "w") as f:
        json.dump(dub_data, f)
    
    print(f"Created {len(good_data)} good samples and {len(dub_data)} dub samples")
    return good_data, dub_data

def read_dataset(data_path, split="train", save_dir="/common/users/hn315/datasets/FiFA-100k/data", n_samples=100000):
    """Legacy function for backward compatibility."""
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
        for out in executor.map(process_example, args_iter, chunksize=32):
            res.append(out)

    with open("./datasets/manifest/fifa_100k.json", "w") as f:
        json.dump(res, f)

    return res

def process_fifa_dataset(data_path, save_dir="/data", n_samples=None, split_ratio=0.5):
    """Main function to process FiFA dataset with proper structure."""
    # Load and process dataset
    res, manifest_dir, from_100k_dir = load_and_process_dataset(
        data_path=data_path,
        split="train", 
        save_dir=save_dir,
        n_samples=n_samples
    )
    
    # Save manifests
    good_data, dub_data = save_manifests(res, manifest_dir, from_100k_dir, split_ratio)
    
    return res, good_data, dub_data

if __name__ == "__main__":
    process_fifa_dataset(
        data_path="path to fifa 100k parquet",
        save_dir="path to save images",
    )