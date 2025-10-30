# import
from transformers import AutoProcessor, AutoModel
from PIL import Image
import torch

# load model

processor_name_or_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
model_pretrained_name_or_path = "yuvalkirstain/PickScore_v1"


class Selector():
    
    def __init__(self, device):
        self.device = device
        self.processor = AutoProcessor.from_pretrained(processor_name_or_path, use_fast=True)
        self.model = AutoModel.from_pretrained(model_pretrained_name_or_path).eval().to(device)

    def score(self, images, prompt, softmax=False):

        # preprocess
        image_inputs = self.processor(
            images=images,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(self.device)

        text_inputs = self.processor(
            text=prompt,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(self.device)


        with torch.no_grad():
            # embed
            image_embs = self.model.get_image_features(**image_inputs)
            image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)

            text_embs = self.model.get_text_features(**text_inputs)
            text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)

            # score
            scores =  (text_embs @ image_embs.T)[0]

            if softmax:
                scores = self.model.logit_scale.exp() * scores
                # get probabilities if you have multiple images to choose from
                probs = torch.softmax(scores, dim=-1)
                return probs.cpu().tolist()
            else:
                return scores.cpu().tolist()
    
    def score_batch(self, images_batch, prompts_batch, softmax=False):
        """
        Score a batch of images with their corresponding prompts.
        
        Args:
            images_batch: List of images (PIL Images)
            prompts_batch: List of prompts (strings)
            softmax: Whether to apply softmax to scores
            
        Returns:
            List of scores for each image-prompt pair
            For matching pairs (i-th image with i-th prompt), returns diagonal scores
        """
        batch_size = len(images_batch)
        
        # preprocess
        image_inputs = self.processor(
            images=images_batch,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(self.device)

        text_inputs = self.processor(
            text=prompts_batch,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            # embed
            image_embs = self.model.get_image_features(**image_inputs)
            image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)

            text_embs = self.model.get_text_features(**text_inputs)
            text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)

            # score - compute similarity matrix between all prompts and all images
            # scores[i, j] = similarity between prompt i and image j
            scores_matrix = text_embs @ image_embs.T
            
            # Extract diagonal for matching pairs (i-th prompt with i-th image)
            # This assumes prompts_batch[i] corresponds to images_batch[i]
            scores = torch.diagonal(scores_matrix).cpu().tolist()

            if softmax:
                # Apply softmax if requested
                scores_tensor = torch.tensor(scores)
                scores_tensor = self.model.logit_scale.exp() * scores_tensor
                probs = torch.softmax(scores_tensor, dim=-1)
                return probs.cpu().tolist()
            else:
                return scores

