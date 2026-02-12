import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from sam3.model_builder import build_sam3_image_model
from tests.export.test_decoder_export import (
    _export_decoder_only,
    _export_full_sam3_pipeline,
    _make_decoder_only_inputs,
)
from tests.export.test_encoder_export import EncoderFusionWrapper
from tests.export.test_image_encoder_export import _export_image_encoder
from tests.export.test_text_encoder_export import _export_text_encoder


def _load_image(path: Path, device: torch.device) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    np_image = np.array(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(np_image).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device)


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

    tokenizer = model.backbone.language_backbone.tokenizer
    token_ids = tokenizer(prompts, context_length=32).to(device)

    return (
        image,
        token_ids,
    )


def _save_export(exported, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.export.save(exported, str(path))


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
        help="Directory to write exported artifacts",
    )
    args = parser.parse_args()

    prompts = [p.strip() for p in args.prompts.split(",") if p.strip()]
    if not prompts:
        raise ValueError("Provide at least one prompt")

    model = build_sam3_image_model(device=args.device, eval_mode=True, enable_segmentation=True)
    model.eval()

    image = _load_image(args.image, torch.device(args.device))
    image = _prepare_image(image, size=1008)
    inputs = _make_inputs(model, image, prompts)

    print("Exporting image encoder...")
    image_encoder = _export_image_encoder(model, inputs[0])
    print("Exporting text encoder...")
    text_encoder = _export_text_encoder(model, inputs[1])
    print("Exporting encoder fusion...")
    with torch.no_grad():
        image_module = image_encoder.module()
        text_module = text_encoder.module()
        _, vision_pos_enc, backbone_fpn = image_module(inputs[0])
        text_attention_mask, text_memory = text_module(inputs[1])
        prompt = text_memory
        prompt_mask = text_attention_mask
        img_feats = backbone_fpn[-1]
        img_pos = vision_pos_enc[-1]
        img_mask = torch.zeros(
            img_feats.shape[0],
            img_feats.shape[2],
            img_feats.shape[3],
            device=img_feats.device,
            dtype=torch.bool,
        )
    prompt_batch = prompt.shape[1]
    if img_feats.shape[0] != prompt_batch:
        if img_feats.shape[0] != 1:
            raise ValueError("Image batch does not match prompt batch")
        img_feats = img_feats.repeat(prompt_batch, 1, 1, 1)
        img_pos = img_pos.repeat(prompt_batch, 1, 1, 1)
        img_mask = img_mask.repeat(prompt_batch, 1, 1)

    encoder_wrapper = EncoderFusionWrapper(model.transformer.encoder).to(img_feats.device).eval()
    encoder = torch.export.export(
        encoder_wrapper,
        (img_feats, img_pos, img_mask, prompt, prompt_mask),
        dynamic_shapes={
            "img_feats": {0: torch.export.Dim("batch", min=1, max=4)},
            "img_pos": {0: torch.export.Dim("batch", min=1, max=4)},
            "img_mask": {0: torch.export.Dim("batch", min=1, max=4)},
            "prompt": {0: 32, 1: torch.export.Dim("batch", min=1, max=4)},
            "prompt_mask": {0: torch.export.Dim("batch", min=1, max=4), 1: 32},
        },
        strict=False,
        prefer_deferred_runtime_asserts_over_guards=True,
    )
    print("Exporting full pipeline...")
    pipeline_inputs = _make_inputs(model, image, prompts[:1])
    full_pipeline = _export_full_sam3_pipeline(model, pipeline_inputs)
    print("Exporting decoder only...")
    decoder_inputs = _make_decoder_only_inputs(model, pipeline_inputs)
    (
        backbone_fpn,
        img_ids,
        memory,
        pos_embed,
        prompt,
        prompt_mask,
        level_start_index,
        spatial_shapes,
        valid_ratios,
    ) = decoder_inputs
    if img_ids.shape[0] < 2:
        repeat = 2 // img_ids.shape[0]
        img_ids = img_ids.repeat(repeat)
        memory = memory.repeat(1, repeat, 1)
        pos_embed = pos_embed.repeat(1, repeat, 1)
        prompt = prompt.repeat(1, repeat, 1)
        prompt_mask = prompt_mask.repeat(repeat, 1)
        valid_ratios = valid_ratios.repeat(repeat, 1, 1)
        backbone_fpn = [feat.repeat(repeat, 1, 1, 1) for feat in backbone_fpn]
    decoder_inputs = (
        backbone_fpn,
        img_ids,
        memory,
        pos_embed,
        prompt,
        prompt_mask,
        level_start_index,
        spatial_shapes,
        valid_ratios,
    )
    decoder_only = _export_decoder_only(model, decoder_inputs)

    _save_export(image_encoder, args.out_dir / "image_encoder.pt2")
    _save_export(text_encoder, args.out_dir / "text_encoder.pt2")
    _save_export(encoder, args.out_dir / "encoder_fusion.pt2")
    _save_export(full_pipeline, args.out_dir / "full_sam3_pipeline.pt2")
    _save_export(decoder_only, args.out_dir / "decoder_only.pt2")
    print("Saved exports to", args.out_dir)


if __name__ == "__main__":
    main()
