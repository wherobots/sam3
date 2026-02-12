import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from sam3.model_builder import build_sam3_image_model
from sam3.model.data_misc import FindStage
from sam3.model.geometry_encoders import Prompt


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
    return out["pred_masks"], out["pred_boxes"], out["pred_logits"]


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


def _timeit(fn, iters: int, device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    end = time.perf_counter()
    return (end - start) / iters


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
    parser.add_argument(
        "--num-feature-levels",
        type=int,
        default=1,
        help="Number of feature levels to use",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
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

    device = torch.device(args.device)
    image = _load_image(args.image, device)
    image = _prepare_image(image, size=1008)
    inputs = _make_inputs(model, image, prompts)

    print("Device (eager):", next(model.parameters()).device)

    def eager_fn():
        _run_full_model(model, inputs)

    with torch.inference_mode():
        for _ in range(args.warmup):
            eager_fn()
        eager_ms = _timeit(eager_fn, args.iters, device) * 1000

    if device.type == "cuda":
        torch.cuda.empty_cache()

    image_module = _load_export(args.artifact_dir / "image_encoder.pt2")
    text_module = _load_export(args.artifact_dir / "text_encoder.pt2")
    encoder_module = _load_export(args.artifact_dir / "encoder_fusion.pt2")
    pipeline_module = _load_export(args.artifact_dir / "full_sam3_pipeline.pt2")
    decoder_module = _load_export(args.artifact_dir / "decoder_only.pt2")
    print("Device (export):", inputs[0].device)

    def image_fn():
        return image_module(inputs[0])

    def text_fn():
        return text_module(inputs[1])

    def encoder_from_outputs(image_out, text_out):
        vision_pos_enc = image_out[1]
        backbone_fpn = image_out[2]
        text_attention_mask, text_memory = text_out
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
        encoder_module(img_feats, img_pos, img_mask, text_memory, text_attention_mask)

    with torch.inference_mode():
        cached_image_out = image_fn()
        cached_text_out = text_fn()

    def encoder_fn():
        encoder_from_outputs(cached_image_out, cached_text_out)

    pipeline_inputs = inputs

    def pipeline_fn():
        pipeline_module(*pipeline_inputs)

    with torch.inference_mode():
        cached_image_out = image_fn()
        cached_text_out = text_fn()
        decoder_only_inputs = _make_decoder_only_inputs_from_model(
            model,
            cached_image_out[2],
            cached_image_out[1],
            cached_text_out[1],
            cached_text_out[0],
            inputs,
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
    ) = decoder_only_inputs
    if decoder_img_ids.shape[0] < 2:
        repeat = 2 // decoder_img_ids.shape[0]
        decoder_img_ids = decoder_img_ids.repeat(repeat)
        decoder_memory = decoder_memory.repeat(1, repeat, 1)
        decoder_pos_embed = decoder_pos_embed.repeat(1, repeat, 1)
        decoder_prompt = decoder_prompt.repeat(1, repeat, 1)
        decoder_prompt_mask = decoder_prompt_mask.repeat(repeat, 1)
        decoder_valid_ratios = decoder_valid_ratios.repeat(repeat, 1, 1)
        decoder_backbone_fpn = [feat.repeat(repeat, 1, 1, 1) for feat in decoder_backbone_fpn]
    decoder_only_inputs = (
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

    def decoder_only_fn():
        decoder_module(*decoder_only_inputs)

    with torch.inference_mode():
        for _ in range(args.warmup):
            pipeline_fn()
        image_ms = _timeit(image_fn, args.iters, device) * 1000
        text_ms = _timeit(text_fn, args.iters, device) * 1000
        encoder_ms = _timeit(encoder_fn, args.iters, device) * 1000
        pipeline_ms = _timeit(pipeline_fn, args.iters, device) * 1000
        decoder_only_ms = _timeit(decoder_only_fn, args.iters, device) * 1000

    print("Eager total (ms):", round(eager_ms, 2))
    print("Export full pipeline total (ms):", round(pipeline_ms, 2))
    print("Export image encoder (ms):", round(image_ms, 2))
    print("Export text encoder (ms):", round(text_ms, 2))
    print("Export encoder fusion (ms):", round(encoder_ms, 2))
    print("Export decoder only (ms):", round(decoder_only_ms, 2))
    del model


if __name__ == "__main__":
    main()
