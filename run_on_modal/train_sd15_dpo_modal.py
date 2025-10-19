import modal

# Define a custom image based on a modern NVIDIA CUDA image
# This image includes the full CUDA toolkit necessary for building custom extensions.
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    # Install system dependencies needed for flash-attention compilation
    .run_commands([
        "apt-get update",
        "apt-get install -y git build-essential cmake",
        "pip install --upgrade pip setuptools wheel"
    ])
    # Install all your pip dependencies in one step.
    .pip_install([
        "torch==2.3.1",
        "torchvision==0.18.1", 
        "torchaudio==2.3.1",
        "numpy",
        "packaging",
        "ninja",
        "setuptools",
        "wheel",
        "accelerate",
        "diffusers",
        "transformers",
        "datasets",
        "pillow",
        "wandb",
        "tqdm",
        "xformers",
        "huggingface_hub",
        "gitpython"
    ])
    # Install flash-attn separately with --no-build-isolation
    # after the other dependencies to ensure torch is available.
    .pip_install("flash-attn==2.6.2", extra_options="--no-build-isolation")
    .add_local_file("../train_sd15_dpo_corrupted.py", "/root/train_sd15_dpo_corrupted.py")
    .add_local_dir("../configs", "/root/configs")
    .add_local_dir("../src", "/root/src")
)

# Create Modal app
app = modal.App("train-dpo")

# Create volume for storage
volume = modal.Volume.from_name("fifa-data", create_if_missing=False)


@app.function(
    image=image,
    volumes={"/data": volume},
    gpu="A100:2",  # 2 A100 GPUs (40GB each)
    timeout=86400,
    secrets=[
        modal.Secret.from_dict({"WANDB_API_KEY": "0c2416138832b33d254a444d26384582d70420e4"}),
        modal.Secret.from_name("huggingface-token")
    ]
)
def train_dpo(model_name: str = "sd15-dpo-trained"):
    import subprocess
    import sys
    import os
    
    sys.path.insert(0, "/root")
    
    # Memory optimizations with flash-attention
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "1"
    
    # Test flash-attention import
    print("Testing flash-attention import...")
    try:
        import flash_attn
        print(f"✓ Flash-attention {flash_attn.__version__} imported successfully")
    except Exception as e:
        print(f"⚠️ Flash-attention import failed: {e}")
        print("Continuing without flash-attention...")
    
    # Fix NCCL issues for 2 GPU setup
    os.environ["NCCL_DEBUG"] = "WARN"
    os.environ["NCCL_IB_DISABLE"] = "1"
    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ["NCCL_SHM_DISABLE"] = "1"
    os.environ["NCCL_SOCKET_IFNAME"] = "eth0"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    
    cmd = [
        "torchrun",
        "--nproc_per_node=2",
        "--master_addr=localhost",
        "--master_port=12355",
        "/root/train_sd15_dpo_corrupted.py",
        "--mixed_precision", "bf16",
        "--pretrained_model_name_or_path", "stable-diffusion-v1-5/stable-diffusion-v1-5",
        "--output_dir", "/data/training_runs/sd15_corrupted_dpo",
        "--good_manifest", "/data/1k_data/manifest/from_100k/good_from_100k.json",
        "--dubious_manifest", "/data/1k_data/manifest/from_100k/dub_from_100k.json", 
        "--train_data_dir", "/data/1k_data/FiFA-100k/data/train",
        "--flip_percentage", "0.4",
        "--train_batch_size", "8",
        "--gradient_accumulation_steps", "8",
        "--dataloader_num_workers", "4",
        "--max_train_steps", "1000",
        "--learning_rate", "1e-5",
        "--checkpointing_steps", "500",
        "--gradient_checkpointing",
        "--allow_tf32",
        "--report_to", "wandb",
        "--hf_repo_name", "tungnguyen/diffusion-alignment",
        "--hf_model_subfolder", model_name
    ]
    
    subprocess.run(cmd)

@app.local_entrypoint()
def main(model_name: str = "sd15-dpo-trained"):
    train_dpo.remote(model_name)
