import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from sam3.model_builder import build_sam3_image_model


def _load_image(path: Path, device: torch.device) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    np_image = np.array(image, dtype=np.float32) / 255.0
    return torch.from_numpy(np_image).permute(2, 0, 1).unsqueeze(0).to(device)


def _prepare_image(image: torch.Tensor, size: int) -> torch.Tensor:
    image = image.clamp(0, 1)
    image = torch.nn.functional.interpolate(
        image, size=(size, size), mode="bilinear", align_corners=False
    )
    mean = torch.tensor([0.5, 0.5, 0.5], device=image.device).view(1, 3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5], device=image.device).view(1, 3, 1, 1)
    return (image - mean) / std


def _make_inputs(model, image: torch.Tensor, prompts):
    device = image.device
    num_prompts = len(prompts)
    num_images = int(image.shape[0])
    token_ids = model.backbone.language_backbone.tokenizer(
        prompts, context_length=32
    ).to(device)
    img_ids = torch.arange(num_images, device=device, dtype=torch.long)
    img_ids = img_ids.repeat_interleave(num_prompts)
    text_ids = torch.arange(num_prompts, device=device, dtype=torch.long)
    text_ids = text_ids.repeat(num_images)
    return (
        image,
        token_ids,
        img_ids,
        text_ids,
        torch.zeros(1, num_prompts, 4, device=device),
        torch.zeros(num_prompts, 1, device=device, dtype=torch.bool),
        torch.zeros(1, num_prompts, device=device, dtype=torch.long),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        type=Path,
        default=Path("assets/images/cat_dog.jpg"),
        help="Path to input image",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        default="cat,dog",
        help="Comma-separated text prompts",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--num-feature-levels",
        type=int,
        default=1,
        help="Number of feature levels to use",
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        default=Path("artifacts/export/full_sam3_pipeline.pt2"),
        help="Path to exported full pipeline artifact",
    )
    args = parser.parse_args()

    prompts = [p.strip() for p in args.prompts.split(",") if p.strip()]
    if not prompts:
        raise ValueError("Provide at least one prompt")

    model = build_sam3_image_model(
        device=args.device,
        eval_mode=True,
        enable_segmentation=True,
        num_feature_levels=args.num_feature_levels,
    )
    model.eval()

    image = _prepare_image(
        _load_image(args.image, torch.device(args.device)), size=1008
    )
    inputs = _make_inputs(model, image, prompts)

    module = torch.export.load(str(args.artifact)).module()
    with torch.no_grad():
        pred_logits, pred_boxes, pred_masks, pred_presence = module(*inputs)

    pred_scores = pred_logits.squeeze(-1)
    print("Prompt count:", len(prompts))
    print("Pred logits shape:", pred_logits.shape)
    print("Pred boxes shape:", pred_boxes.shape)
    print("Pred masks shape:", pred_masks.shape)
    if pred_presence is not None:
        print("Presence logits shape:", pred_presence.shape)
    for idx, prompt_text in enumerate(prompts):
        max_val = pred_scores[idx].max().item()
        max_idx = pred_scores[idx].argmax().item()
        print(f"Prompt '{prompt_text}' max logit: {max_val:.4f} (idx {max_idx})")


if __name__ == "__main__":
    main()
