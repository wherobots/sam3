import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from PIL import ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from sam3.model_builder import build_sam3_image_model
from sam3.model.data_misc import FindStage
from sam3.model.geometry_encoders import Prompt


def _load_pil_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def _pil_to_tensor(image: Image.Image, device: torch.device) -> torch.Tensor:
    np_image = np.array(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(np_image).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device)


def _load_image(path: Path, device: torch.device) -> torch.Tensor:
    return _pil_to_tensor(_load_pil_image(path), device)


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


def _run_full_model(model, inputs):
    images, token_ids = inputs
    num_images = images.shape[0]
    num_prompts = token_ids.shape[0]
    device = images.device
    bs = num_images * num_prompts

    img_ids = torch.arange(num_images, device=device, dtype=torch.long)
    img_ids = img_ids.repeat_interleave(num_prompts)
    text_ids = torch.arange(num_prompts, device=device, dtype=torch.long)
    text_ids = text_ids.repeat(num_images)

    box_embeddings = torch.zeros(1, bs, 4, device=device)
    box_mask = torch.zeros(bs, 1, device=device, dtype=torch.bool)
    box_labels = torch.zeros(1, bs, device=device, dtype=torch.long)
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
        input_points=torch.zeros(0, int(token_ids.shape[0]), 2, device=images.device),
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
        out["pred_masks"],
        out["pred_boxes"],
        out["pred_logits"],
        out["pred_boxes_xyxy"],
    )


def _make_decoder_only_inputs_from_model(
    model,
    backbone_fpn,
    vision_pos_enc,
    text_memory,
    text_attention_mask,
    inputs,
):
    images, token_ids = inputs
    num_images = images.shape[0]
    num_prompts = token_ids.shape[0]
    device = images.device
    bs = num_images * num_prompts

    img_ids = torch.arange(num_images, device=device, dtype=torch.long)
    img_ids = img_ids.repeat_interleave(num_prompts)
    text_ids = torch.arange(num_prompts, device=device, dtype=torch.long)
    text_ids = text_ids.repeat(num_images)

    box_embeddings = torch.zeros(1, bs, 4, device=device)
    box_mask = torch.zeros(bs, 1, device=device, dtype=torch.bool)
    box_labels = torch.zeros(1, bs, device=device, dtype=torch.long)
    backbone_out = {
        "backbone_fpn": backbone_fpn,
        "vision_pos_enc": vision_pos_enc,
        "language_features": text_memory,
        "language_mask": text_attention_mask,
    }
    find_input = FindStage(
        img_ids=img_ids,
        text_ids=text_ids,
        input_boxes=box_embeddings,
        input_boxes_mask=box_mask,
        input_boxes_label=box_labels,
        input_points=torch.zeros(0, int(token_ids.shape[0]), 2, device=images.device),
        input_points_mask=torch.zeros(
            int(token_ids.shape[0]), 0, device=images.device, dtype=torch.bool
        ),
    )
    geometric_prompt = Prompt(
        box_embeddings=box_embeddings,
        box_mask=box_mask,
        box_labels=box_labels,
    )
    prompt, prompt_mask, backbone_out = model._encode_prompt(
        backbone_out, find_input, geometric_prompt
    )
    backbone_out, encoder_out, _ = model._run_encoder(backbone_out, find_input, prompt, prompt_mask)
    return (
        backbone_out["backbone_fpn"],
        img_ids,
        encoder_out["encoder_hidden_states"],
        encoder_out["pos_embed"],
        prompt,
        prompt_mask,
        encoder_out["level_start_index"],
        encoder_out["spatial_shapes"],
        encoder_out["valid_ratios"],
    )


def _load_export(path: Path):
    exported = torch.export.load(str(path))
    return exported.module()


def _to_pil_image(image: torch.Tensor) -> Image.Image:
    image = image.detach().cpu().clamp(0, 1)
    np_image = (image.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(np_image)


def _color_palette(num_colors: int):
    base = [
        (255, 99, 71),
        (65, 105, 225),
        (60, 179, 113),
        (238, 130, 238),
        (255, 215, 0),
        (255, 165, 0),
    ]
    return [base[i % len(base)] for i in range(num_colors)]


def _overlay_masks(image: Image.Image, masks: torch.Tensor, scores: torch.Tensor, out_path: Path):
    num_prompts, num_queries = scores.shape[:2]
    best_idx = scores.squeeze(-1).argmax(dim=1)
    colors = _color_palette(num_prompts)
    base = image.copy().convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    for i in range(num_prompts):
        mask = masks[i, best_idx[i]].detach().cpu()
        mask = mask > 0
        mask_img = Image.fromarray((mask.numpy() * 255).astype(np.uint8), mode="L")
        if mask_img.size != base.size:
            mask_img = mask_img.resize(base.size, resample=Image.Resampling.NEAREST)
        color = colors[i]
        color_img = Image.new("RGBA", base.size, (*color, 120))
        overlay = Image.composite(color_img, overlay, mask_img)
    blended = Image.alpha_composite(base, overlay)
    blended.convert("RGB").save(out_path)


def _draw_boxes(image: Image.Image, boxes_xyxy: torch.Tensor, scores: torch.Tensor, out_path: Path):
    num_prompts, num_queries = scores.shape[:2]
    best_idx = scores.squeeze(-1).argmax(dim=1).clamp(max=boxes_xyxy.shape[1] - 1)
    colors = _color_palette(num_prompts)
    draw = ImageDraw.Draw(image)
    for i in range(num_prompts):
        box_tensor = boxes_xyxy[i, best_idx[i]].detach().cpu().flatten()
        if box_tensor.numel() != 4:
            continue
        box = box_tensor.tolist()
        width, height = image.size
        if max(box) <= 1.0:
            box = [
                box[0] * width,
                box[1] * height,
                box[2] * width,
                box[3] * height,
            ]
        box = [
            max(0.0, min(box[0], width)),
            max(0.0, min(box[1], height)),
            max(0.0, min(box[2], width)),
            max(0.0, min(box[3], height)),
        ]
        color = colors[i]
        draw.rectangle(box, outline=color, width=3)
    image.save(out_path)


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
        "--artifact-dir",
        type=Path,
        default=Path("artifacts/export"),
        help="Directory with exported artifacts",
    )
    args = parser.parse_args()

    prompts = [p.strip() for p in args.prompts.split(",") if p.strip()]
    if not prompts:
        raise ValueError("Provide at least one prompt")
    prompt_count = len(prompts)

    model = build_sam3_image_model(device=args.device, eval_mode=True, enable_segmentation=True)
    model.eval()

    pil_image = _load_pil_image(args.image)
    image = _pil_to_tensor(pil_image, torch.device(args.device))
    image = _prepare_image(image, size=1008)
    inputs = _make_inputs(model, image, prompts)

    with torch.no_grad():
        eager_masks, eager_boxes, eager_logits, eager_boxes_xyxy = _run_full_model(model, inputs)

    image_module = _load_export(args.artifact_dir / "image_encoder.pt2")
    text_module = _load_export(args.artifact_dir / "text_encoder.pt2")
    encoder_module = _load_export(args.artifact_dir / "encoder_fusion.pt2")
    pipeline_module = _load_export(args.artifact_dir / "full_sam3_pipeline.pt2")
    decoder_module = _load_export(args.artifact_dir / "decoder_only.pt2")

    with torch.no_grad():
        _, vision_pos_enc, backbone_fpn = image_module(inputs[0])
        text_attention_mask, text_memory = text_module(inputs[1])
        img_feats = backbone_fpn[-1]
        img_pos = vision_pos_enc[-1]
        prompt_batch = text_attention_mask.shape[0]
        if img_feats.shape[0] != prompt_batch:
            if img_feats.shape[0] != 1:
                raise ValueError("Image batch does not match prompt batch")
            img_feats = img_feats.repeat(prompt_batch, 1, 1, 1)
            img_pos = img_pos.repeat(prompt_batch, 1, 1, 1)
        img_mask = torch.zeros(
            img_feats.shape[0],
            img_feats.shape[2],
            img_feats.shape[3],
            device=img_feats.device,
            dtype=torch.bool,
        )
        enc_out = encoder_module(img_feats, img_pos, img_mask, text_memory, text_attention_mask)
        assert isinstance(enc_out, tuple)
        pipeline_logits, pipeline_boxes, pipeline_masks, pipeline_boxes_xyxy = pipeline_module(
            *inputs
        )
        (
            decoder_backbone_fpn,
            decoder_img_ids,
            decoder_memory,
            decoder_pos_embed,
            decoder_prompt,
            decoder_prompt_mask,
            decoder_level_start_index,
            decoder_spatial_shapes,
            decoder_valid_ratios,
        ) = _make_decoder_only_inputs_from_model(
            model,
            backbone_fpn,
            vision_pos_enc,
            text_memory,
            text_attention_mask,
            inputs,
        )
        batch_target = decoder_img_ids.shape[0]
        if batch_target < 2:
            repeat = 2 // batch_target
            decoder_img_ids = decoder_img_ids.repeat(repeat)
            decoder_memory = decoder_memory.repeat(1, repeat, 1)
            decoder_pos_embed = decoder_pos_embed.repeat(1, repeat, 1)
            decoder_prompt = decoder_prompt.repeat(1, repeat, 1)
            decoder_prompt_mask = decoder_prompt_mask.repeat(repeat, 1)
            decoder_valid_ratios = decoder_valid_ratios.repeat(repeat, 1, 1)
            decoder_backbone_fpn = [feat.repeat(repeat, 1, 1, 1) for feat in decoder_backbone_fpn]
        pred_logits, pred_boxes, pred_masks, pred_boxes_xyxy = decoder_module(
            decoder_backbone_fpn,
            decoder_img_ids,
            decoder_memory,
            decoder_pos_embed,
            decoder_prompt,
            decoder_prompt_mask,
            decoder_level_start_index,
            decoder_spatial_shapes,
            decoder_valid_ratios,
        )
        eager_ref_masks, eager_ref_boxes, eager_ref_logits, eager_ref_boxes_xyxy = _run_full_model(
            model, inputs
        )

    pred_logits = pred_logits[:prompt_count]
    pred_boxes = pred_boxes[:prompt_count]
    pred_masks = pred_masks[:prompt_count]
    pred_boxes_xyxy = pred_boxes_xyxy[:prompt_count]
    pipeline_logits = pipeline_logits[:prompt_count]
    pipeline_masks = pipeline_masks[:prompt_count]
    pipeline_boxes_xyxy = pipeline_boxes_xyxy[:prompt_count]
    eager_ref_logits = eager_ref_logits[:prompt_count]

    print("Prompt count:", prompt_count)
    print("Pred logits shape:", pred_logits.shape)
    print("Pred boxes shape:", pred_boxes.shape)
    print("Pred masks shape:", pred_masks.shape)
    pred_scores = pred_logits.squeeze(-1)
    eager_scores = eager_ref_logits.squeeze(-1)
    pred_max = pred_scores.max(dim=1).values
    eager_max = eager_scores.max(dim=1).values
    pred_best_idx = pred_scores.argmax(dim=1)
    eager_best_idx = eager_scores.argmax(dim=1)
    for idx, prompt_text in enumerate(prompts):
        print(
            f"Prompt '{prompt_text}' max logit: "
            f"export={pred_max[idx].item():.4f} (idx {pred_best_idx[idx].item()}), "
            f"eager={eager_max[idx].item():.4f} (idx {eager_best_idx[idx].item()})"
        )
    torch.testing.assert_close(pred_logits, eager_ref_logits, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(pipeline_logits, eager_ref_logits, rtol=1e-3, atol=1e-3)
    print("Eager vs export logits match")

    out_dir = args.artifact_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    base_image = pil_image.copy()
    _overlay_masks(
        base_image.copy(),
        eager_masks,
        eager_logits,
        out_dir / "eager_masks_overlay.jpg",
    )
    _overlay_masks(
        base_image.copy(),
        pipeline_masks,
        pipeline_logits,
        out_dir / "pipeline_masks_overlay.jpg",
    )
    _overlay_masks(
        base_image.copy(),
        pred_masks,
        pred_logits,
        out_dir / "decoder_only_masks_overlay.jpg",
    )
    for idx, prompt_text in enumerate(prompts):
        _overlay_masks(
            base_image.copy(),
            eager_masks[idx : idx + 1],
            eager_logits[idx : idx + 1],
            out_dir / f"eager_mask_overlay_{idx}.jpg",
        )
        _overlay_masks(
            base_image.copy(),
            pipeline_masks[idx : idx + 1],
            pipeline_logits[idx : idx + 1],
            out_dir / f"pipeline_mask_overlay_{idx}.jpg",
        )
        _overlay_masks(
            base_image.copy(),
            pred_masks[idx : idx + 1],
            pred_logits[idx : idx + 1],
            out_dir / f"decoder_only_mask_overlay_{idx}.jpg",
        )
    _draw_boxes(
        base_image.copy(),
        eager_boxes_xyxy,
        eager_logits,
        out_dir / "eager_boxes_overlay.jpg",
    )
    _draw_boxes(
        base_image.copy(),
        pipeline_boxes_xyxy,
        pipeline_logits,
        out_dir / "pipeline_boxes_overlay.jpg",
    )
    _draw_boxes(
        base_image.copy(),
        pred_boxes_xyxy,
        pred_logits,
        out_dir / "decoder_only_boxes_overlay.jpg",
    )
    print("Saved overlays to", out_dir)


if __name__ == "__main__":
    main()
