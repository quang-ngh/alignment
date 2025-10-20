# Complete Modal Workflow

This directory contains the complete pipeline from dataset preparation to model evaluation.

## Files

- `download_fifa_simple.py` - Download FIFA dataset
- `run_utils_modal.py` - Process dataset and create manifests
- `train_sd15_dpo_modal.py` - Train DPO model
- `generate_with_unet.py` - Core functions for loading UNet and generating images
- `generate_only.py` - Generate images from trained UNet
- `eval_only.py` - Evaluate generated images

## Complete Workflow

### Step 1: Download Dataset
```bash
modal run download_fifa_simple.py
```

### Step 2: Process Dataset
```bash
modal run run_utils_modal.py
```

### Step 3: Train Model
```bash
modal run train_sd15_dpo_modal.py
```

### Step 4: Generate Images
```bash
modal run generate_only.py
```

### Step 5: Evaluate Images
```bash
modal run eval_only.py
```

### Download Results
```bash
# Download generated images
modal volume get fifa-data /data/evaluation_output/generated_images ./local_generated_images

# Download evaluation results
modal volume get fifa-data /data/evaluation_output/evaluation_results ./local_eval_results
```

## Output Locations

- Dataset: `/data/fifa/`
- Processed data: `/data/1k_data/`
- Trained model: `/data/training_runs/sd15_corrupted_dpo/`
- Generated images: `/data/evaluation_output/generated_images/`
- Evaluation results: `/data/evaluation_output/evaluation_results/`
