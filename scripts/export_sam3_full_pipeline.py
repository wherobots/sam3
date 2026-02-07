import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from sam3.model_builder import build_sam3_image_model
from sam3.model.data_misc import FindStage
from sam3.model.geometry_encoders import Prompt


class FullSam3PipelineWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(
        self,
        images: torch.Tensor,
        token_ids: torch.Tensor,
        img_ids: torch.Tensor,
        text_ids: torch.Tensor,
        box_embeddings: torch.Tensor,
        box_mask: torch.Tensor,
        box_labels: torch.Tensor,
    ):
        model = self.model
        backbone_out = model.backbone.forward_image(images)
        text_encoder = model.backbone.language_backbone
        _, text_tokens = text_encoder.encoder(token_ids)
        text_tokens = text_tokens.transpose(0, 1)
        text_memory = text_encoder.resizer(text_tokens)
        text_attention_mask = token_ids.ne(0)
        text_attention_mask = text_attention_mask.ne(1)
        backbone_out["language_features"] = text_memory
        backbone_out["language_mask"] = text_attention_mask

        find_input = FindStage(
            img_ids=img_ids,
            text_ids=text_ids,
            input_boxes=box_embeddings,
            input_boxes_mask=box_mask,
            input_boxes_label=box_labels,
            input_points=torch.zeros(
                0, int(token_ids.shape[0]), 2, device=images.device
            ),
            input_points_mask=torch.zeros(
                int(token_ids.shape[0]), 0, device=images.device, dtype=torch.bool
            ),
        )
        geometric_prompt = Prompt(
            box_embeddings=box_embeddings,
            box_mask=box_mask,
            box_labels=box_labels,
        )
        out = model.forward_grounding(
            backbone_out=backbone_out,
            find_input=find_input,
            find_target=None,
            geometric_prompt=geometric_prompt,
        )
        return (
            out["pred_logits"],
            out["pred_boxes"],
            out["pred_masks"],
            out.get("presence_logit_dec"),
        )


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
        "--out-dir",
        type=Path,
        default=Path("artifacts/export"),
        help="Directory to write exported artifact",
    )
    args = parser.parse_args()

    prompts = [p.strip() for p in args.prompts.split(",") if p.strip()]
    if not prompts:
        raise ValueError("Provide at least one prompt")

    model = build_sam3_image_model(
        device=args.device, eval_mode=True, enable_segmentation=True
    )
    model.eval()

    image = _prepare_image(
        _load_image(args.image, torch.device(args.device)), size=1008
    )
    inputs = _make_inputs(model, image, prompts)
    wrapper = FullSam3PipelineWrapper(model).to(image.device).eval()
    if image.shape[0] < 2:
        repeat = 2 // image.shape[0]
        export_inputs = (
            image.repeat(repeat, 1, 1, 1),
            inputs[1].repeat(repeat, 1),
            inputs[2].repeat(repeat),
            inputs[3].repeat(repeat),
            inputs[4].repeat(1, repeat, 1),
            inputs[5].repeat(repeat, 1),
            inputs[6].repeat(1, repeat),
        )
    else:
        export_inputs = inputs
    with torch.no_grad():
        exported = torch.export.export(
            wrapper,
            export_inputs,
            dynamic_shapes={
                "images": {
                    0: torch.export.Dim.AUTO,
                    2: 1008,
                    3: 1008,
                },
                "token_ids": {
                    0: torch.export.Dim.AUTO,
                    1: 32,
                },
                "img_ids": {0: torch.export.Dim.AUTO},
                "text_ids": {0: torch.export.Dim.AUTO},
                "box_embeddings": {
                    0: 1,
                    1: torch.export.Dim.AUTO,
                },
                "box_mask": {
                    0: torch.export.Dim.AUTO,
                    1: 1,
                },
                "box_labels": {
                    0: 1,
                    1: torch.export.Dim.AUTO,
                },
            },
            strict=False,
            prefer_deferred_runtime_asserts_over_guards=True,
        )
    out_path = args.out_dir / "full_sam3_pipeline.pt2"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.export.save(exported, str(out_path))
    print("Saved export to", out_path)


if __name__ == "__main__":
    main()
