import os
from typing import Dict, Optional, Union
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor
import hpsv2  


# Types
ImageLike = Union[str, Image.Image]


_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------- HPSv2 ----------
def evaluate_hpsv2(
    image: ImageLike,
    prompt: str,
    hps_version: str = "v2.1",
) -> float:
    """Return HPSv2 score for a single image and prompt."""
    scores = hpsv2.score(image, prompt, hps_version=hps_version)
    if not scores:
        raise RuntimeError("HPSv2 returned no scores.")
    return float(scores[0])


# ---------- PickScore ----------
_pickscore_model = None
_pickscore_processor = None
def _load_pickscore(model_dir: Optional[str] = None, processor_dir: Optional[str] = None):
    global _pickscore_model, _pickscore_processor
    if _pickscore_model is not None and _pickscore_processor is not None:
        return

    default_processor_dir = (
        processor_dir
        or os.path.abspath(
            os.path.join(
                os.path.dirname(__file__), "checkpoints", "CLIP-ViT-H-14-laion2B-s32B-b79K"
            )
        )
    )
    default_model_dir = (
        model_dir
        or os.path.abspath(
            os.path.join(os.path.dirname(__file__), "checkpoints", "pickscore_v1")
        )
    )

    processor_name_or_path = (
        default_processor_dir
        if os.path.isdir(default_processor_dir)
        else "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    )
    model_pretrained_name_or_path = (
        default_model_dir if os.path.isdir(default_model_dir) else "yuvalkirstain/PickScore_v1"
    )

    _pickscore_processor = AutoProcessor.from_pretrained(processor_name_or_path)
    _pickscore_model = AutoModel.from_pretrained(model_pretrained_name_or_path).eval().to(_DEVICE)


def evaluate_pickscore(
    image: ImageLike,
    prompt: str,
    *,
    model_dir: Optional[str] = None,
    processor_dir: Optional[str] = None,
) -> float:
    """Return PickScore for a single image and prompt."""
    _load_pickscore(model_dir=model_dir, processor_dir=processor_dir)

    assert _pickscore_model is not None and _pickscore_processor is not None

    pil_img = image if isinstance(image, Image.Image) else Image.open(image)

    image_inputs = _pickscore_processor(
        images=[pil_img], padding=True, truncation=True, max_length=77, return_tensors="pt"
    ).to(_DEVICE)
    text_inputs = _pickscore_processor(
        text=[prompt], padding=True, truncation=True, max_length=77, return_tensors="pt"
    ).to(_DEVICE)

    with torch.no_grad():
        image_embs = _pickscore_model.get_image_features(**image_inputs)
        image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)

        text_embs = _pickscore_model.get_text_features(**text_inputs)
        text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)

        scores = _pickscore_model.logit_scale.exp() * (text_embs @ image_embs.T)

    return float(scores[0, 0].detach().cpu().item())


_imagereward_model = None
def _load_imagereward(model_name: str = "ImageReward-v1.0"):
    global _imagereward_model
    if _imagereward_model is not None:
        return
    import ImageReward as RM

    _imagereward_model = RM.load(model_name)


def evaluate_imagereward(
    image: ImageLike,
    prompt: str,
    *,
    model_name: str = "ImageReward-v1.0",
) -> float:
    """Return ImageReward score for a single image and prompt."""
    _load_imagereward(model_name=model_name)

    assert _imagereward_model is not None

    # The ImageReward API accepts either image path or PIL image directly
    target = image if isinstance(image, Image.Image) else str(image)
    with torch.no_grad():
        score = _imagereward_model.score(prompt, target)
    # score can be a float tensor or float
    return float(score) if not isinstance(score, list) else float(score[0])


def evaluate_all(
    image: ImageLike,
    prompt: str,
    *,
    hps_version: str = "v2.1",
    pickscore_model_dir: Optional[str] = None,
    pickscore_processor_dir: Optional[str] = None,
    imagereward_model_name: str = "ImageReward-v1.0",
) -> Dict[str, float]:

    return {
        "pickscore": evaluate_pickscore(
            image,
            prompt,
            model_dir=pickscore_model_dir,
            processor_dir=pickscore_processor_dir,
        ),
        "hpsv2": evaluate_hpsv2(image, prompt, hps_version=hps_version),
        "imagereward": evaluate_imagereward(
            image, prompt, model_name=imagereward_model_name
        ),
    }


