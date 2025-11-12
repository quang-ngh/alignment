import os
import json
import torch
from tqdm import tqdm
from torchmetrics.multimodal.clip_score import CLIPScore
from PIL import Image
from torchvision.transforms import ToTensor
import argparse
from transformers import  AutoTokenizer, AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ========== Strategy 1: CLIP Score ==========
class CLIPScoreStrategy:
    def __init__(self, model_name="openai/clip-vit-large-patch14"):
        print(f"Loading CLIP model: {model_name}")
        self.model = CLIPScore(model_name_or_path=model_name).to(_DEVICE)
        self.to_tensor = ToTensor()
        
    def compare_images(self, image_0_path, image_1_path, prompt):
        """
        Compare two images using CLIP score.
        Returns 1 if image_1 has higher score, 0 if image_0 has higher score.
        """
        # Load images
        image_0 = Image.open(image_0_path).convert("RGB")
        image_1 = Image.open(image_1_path).convert("RGB")
        
        # Convert to tensor
        img_0_tensor = self.to_tensor(image_0).unsqueeze(0).to(_DEVICE)
        img_1_tensor = self.to_tensor(image_1).unsqueeze(0).to(_DEVICE)
        
        # Compute scores
        with torch.no_grad():
            score_0 = self.model(img_0_tensor, prompt).item()
            score_1 = self.model(img_1_tensor, prompt).item()
        
        # Return 1 if image_1 is better, 0 if image_0 is better
        refer_id = 1 if score_1 > score_0 else 0
        
        return refer_id, score_0, score_1


# ========== Strategy 2: Qwen VLM ==========
class QwenVLMStrategy:
    def __init__(self, model_name="Qwen/Qwen2-VL-7B-Instruct"):
        print(f"Loading Qwen VLM model: {model_name}")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
        
    def compare_images(self, image_0_path, image_1_path, prompt):
        """
        Compare two images using Qwen VLM.
        Returns 1 if image_1 is better aligned, 0 if image_0 is better aligned.
        """
        # Create the prompt for VLM
        instruction = f"""Given the following text prompt: "{prompt}"

Please compare the two images and determine which one is better aligned with the text prompt. 
Consider aspects like:
- How well the image matches the description
- Quality and clarity
- Composition and aesthetics
- Accuracy of elements mentioned in the prompt

Please respond with ONLY "Image 1" or "Image 2" to indicate which image is better aligned with the prompt."""

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_0_path,
                    },
                    {
                        "type": "image",
                        "image": image_1_path,
                    },
                    {
                        "type": "text",
                        "text": instruction,
                    },
                ],
            }
        ]
        
        # Prepare for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)
        
        # Generate response
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        # Parse response
        output_text_lower = output_text.lower().strip()
        
        # Determine which image is better
        # Image 1 corresponds to image_0, Image 2 corresponds to image_1
        if "image 2" in output_text_lower or "second" in output_text_lower:
            refer_id = 1
        elif "image 1" in output_text_lower or "first" in output_text_lower:
            refer_id = 0
        else:
            # Default to 0 if unclear
            print(f"Warning: Unclear VLM response: {output_text}. Defaulting to image 0.")
            refer_id = 0
            
        return refer_id, output_text


# ========== Main Generation Function ==========
def generate_pseudo_labels(
    input_json_path,
    output_json_path,
    image_base_dir,
    strategy="clipscore",
    model_name=None,
    batch_size=1,
):
    """
    Generate pseudo labels for unlabeled data.
    
    Args:
        input_json_path: Path to the unlabeled JSON file
        output_json_path: Path to save the pseudo-labeled JSON file
        image_base_dir: Base directory containing the images
        strategy: "clipscore" or "qwen_vlm"
        model_name: Optional model name override
        batch_size: Batch size for processing (currently only supports 1)
    """
    # Load data
    print(f"Loading data from {input_json_path}")
    with open(input_json_path, "r") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} samples")
    
    # Initialize strategy
    if strategy == "clipscore":
        if model_name is None:
            model_name = "openai/clip-vit-base-patch32"
        strategy_model = CLIPScoreStrategy(model_name=model_name)
    elif strategy == "qwen_vlm":
        if model_name is None:
            model_name = "Qwen/Qwen2-VL-7B-Instruct"
        strategy_model = QwenVLMStrategy(model_name=model_name)
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Choose 'clipscore' or 'qwen_vlm'")
    
    # Process each sample
    results = []
    for idx, item in enumerate(tqdm(data, desc=f"Generating pseudo labels ({strategy})")):
        caption = item["caption"]
        image_0_basename = item["image_0_basename"]
        image_1_basename = item["image_1_basename"]
        
        # Construct full paths
        image_0_path = os.path.join(image_base_dir, image_0_basename)
        image_1_path = os.path.join(image_base_dir, image_1_basename)
        
        # Check if images exist
        if not os.path.exists(image_0_path):
            print(f"Warning: Image not found: {image_0_path}, skipping...")
            continue
        if not os.path.exists(image_1_path):
            print(f"Warning: Image not found: {image_1_path}, skipping...")
            continue
        
        try:
            # Compare images
            if strategy == "clipscore":
                refer_id, score_0, score_1 = strategy_model.compare_images(
                    image_0_path, image_1_path, caption
                )
                # Store result
                result = {
                    "caption": caption,
                    "image_0_basename": image_0_basename,
                    "image_1_basename": image_1_basename,
                    "refer_id": str(float(refer_id)),
                    "score_0": float(score_0),
                    "score_1": float(score_1),
                }
            elif strategy == "qwen_vlm":
                refer_id, vlm_response = strategy_model.compare_images(
                    image_0_path, image_1_path, caption
                )
                # Store result
                result = {
                    "caption": caption,
                    "image_0_basename": image_0_basename,
                    "image_1_basename": image_1_basename,
                    "refer_id": str(float(refer_id)),
                    "vlm_response": vlm_response,
                }
            
            results.append(result)
            
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            continue
    
    # Save results
    print(f"Saving {len(results)} pseudo-labeled samples to {output_json_path}")
    with open(output_json_path, "w") as f:
        json.dump(results, f, indent=4)
    
    # Print statistics
    if results:
        refer_1_count = sum(1 for r in results if r["refer_id"] == "1.0")
        refer_0_count = len(results) - refer_1_count
        print(f"\nStatistics:")
        print(f"  Total samples: {len(results)}")
        print(f"  Image 0 preferred: {refer_0_count} ({refer_0_count/len(results)*100:.2f}%)")
        print(f"  Image 1 preferred: {refer_1_count} ({refer_1_count/len(results)*100:.2f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate pseudo labels for image preference data")
    parser.add_argument(
        "--input_json",
        type=str,
        required=True,
        help="Path to the unlabeled JSON file"
    )
    parser.add_argument(
        "--output_json",
        type=str,
        required=True,
        help="Path to save the pseudo-labeled JSON file"
    )
    parser.add_argument(
        "--image_base_dir",
        type=str,
        required=True,
        help="Base directory containing the images"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["clipscore", "qwen_vlm"],
        default="clipscore",
        help="Strategy to use for generating pseudo labels"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Model name (optional, will use defaults if not provided)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for processing (currently only supports 1)"
    )
    
    args = parser.parse_args()
    
    generate_pseudo_labels(
        input_json_path=args.input_json,
        output_json_path=args.output_json,
        image_base_dir=args.image_base_dir,
        strategy=args.strategy,
        model_name=args.model_name,
        batch_size=args.batch_size,
    )
