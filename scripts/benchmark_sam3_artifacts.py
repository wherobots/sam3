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
    num_prompts = len(prompts)

    tokenizer = model.backbone.language_backbone.tokenizer
    token_ids = tokenizer(prompts, context_length=32).to(device)

    img_ids = torch.zeros(num_prompts, device=device, dtype=torch.long)
    text_ids = torch.zeros(num_prompts, device=device, dtype=torch.long)

    box_embeddings = torch.zeros(1, num_prompts, 4, device=device)
    box_mask = torch.zeros(num_prompts, 1, device=device, dtype=torch.bool)
    box_labels = torch.zeros(1, num_prompts, device=device, dtype=torch.long)

    return (
        image,
        token_ids,
        img_ids,
        text_ids,
        box_embeddings,
        box_mask,
        box_labels,
    )


def _run_full_model(model, inputs):
    (
        images,
        token_ids,
        img_ids,
        text_ids,
        box_embeddings,
        box_mask,
        box_labels,
    ) = inputs
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
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    args = parser.parse_args()

    prompts = [p.strip() for p in args.prompts.split(",") if p.strip()]
    if not prompts:
        raise ValueError("Provide at least one prompt")

    model = build_sam3_image_model(
        device=args.device, eval_mode=True, enable_segmentation=True
    )
    model.eval()

    device = torch.device(args.device)
    image = _load_image(args.image, device)
    image = _prepare_image(image, size=1008)
    inputs = _make_inputs(model, image, prompts)

    image_module = _load_export(args.artifact_dir / "image_encoder.pt2")
    text_module = _load_export(args.artifact_dir / "text_encoder.pt2")
    encoder_module = _load_export(args.artifact_dir / "encoder_fusion.pt2")
    decoder_module = _load_export(args.artifact_dir / "decoder.pt2")

    with torch.no_grad():
        # Warmup
        for _ in range(args.warmup):
            _run_full_model(model, inputs)
            vision_pos_enc = image_module(inputs[0])[1]
            backbone_fpn = image_module(inputs[0])[2]
            text_attention_mask, text_memory = text_module(inputs[1])
            img_feats = backbone_fpn[-1]
            img_pos = vision_pos_enc[-1]
            img_mask = torch.zeros(
                img_feats.shape[0],
                img_feats.shape[2],
                img_feats.shape[3],
                device=img_feats.device,
                dtype=torch.bool,
            )
            encoder_module(
                img_feats, img_pos, img_mask, text_memory, text_attention_mask
            )
            (
                images,
                token_ids,
                img_ids,
                text_ids,
                box_embeddings,
                box_mask,
                box_labels,
            ) = inputs
            if token_ids.shape[0] < 2:
                repeat = 2 // token_ids.shape[0]
                token_ids = token_ids.repeat(repeat, 1)
                img_ids = img_ids.repeat(repeat)
                text_ids = text_ids.repeat(repeat)
                box_embeddings = box_embeddings.repeat(1, repeat, 1)
                box_mask = box_mask.repeat(repeat, 1)
                box_labels = box_labels.repeat(1, repeat)
            decoder_module(
                images,
                token_ids,
                img_ids,
                text_ids,
                box_embeddings,
                box_mask,
                box_labels,
            )

    def eager_fn():
        _run_full_model(model, inputs)

    def image_fn():
        image_module(inputs[0])

    def text_fn():
        text_module(inputs[1])

    def encoder_fn():
        vision_pos_enc = image_module(inputs[0])[1]
        backbone_fpn = image_module(inputs[0])[2]
        text_attention_mask, text_memory = text_module(inputs[1])
        img_feats = backbone_fpn[-1]
        img_pos = vision_pos_enc[-1]
        img_mask = torch.zeros(
            img_feats.shape[0],
            img_feats.shape[2],
            img_feats.shape[3],
            device=img_feats.device,
            dtype=torch.bool,
        )
        encoder_module(img_feats, img_pos, img_mask, text_memory, text_attention_mask)

    def decoder_fn():
        (
            images,
            token_ids,
            img_ids,
            text_ids,
            box_embeddings,
            box_mask,
            box_labels,
        ) = inputs
        if token_ids.shape[0] < 2:
            repeat = 2 // token_ids.shape[0]
            token_ids = token_ids.repeat(repeat, 1)
            img_ids = img_ids.repeat(repeat)
            text_ids = text_ids.repeat(repeat)
            box_embeddings = box_embeddings.repeat(1, repeat, 1)
            box_mask = box_mask.repeat(repeat, 1)
            box_labels = box_labels.repeat(1, repeat)
        decoder_module(
            images,
            token_ids,
            img_ids,
            text_ids,
            box_embeddings,
            box_mask,
            box_labels,
        )

    eager_ms = _timeit(eager_fn, args.iters, device) * 1000
    image_ms = _timeit(image_fn, args.iters, device) * 1000
    text_ms = _timeit(text_fn, args.iters, device) * 1000
    encoder_ms = _timeit(encoder_fn, args.iters, device) * 1000
    decoder_ms = _timeit(decoder_fn, args.iters, device) * 1000

    print("Eager total (ms):", round(eager_ms, 2))
    print("Export image encoder (ms):", round(image_ms, 2))
    print("Export text encoder (ms):", round(text_ms, 2))
    print("Export encoder fusion (ms):", round(encoder_ms, 2))
    print("Export decoder (ms):", round(decoder_ms, 2))


if __name__ == "__main__":
    main()
