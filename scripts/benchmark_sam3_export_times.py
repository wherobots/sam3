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
from tests.export.test_decoder_export import (
    _export_decoder_only,
    _export_full_sam3_pipeline,
    _make_decoder_only_inputs,
    _make_inputs,
)
from tests.export.test_encoder_export import EncoderFusionWrapper
from tests.export.test_image_encoder_export import _export_image_encoder
from tests.export.test_text_encoder_export import _export_text_encoder


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


def _time(label: str, fn) -> float:
    start = time.perf_counter()
    fn()
    elapsed = time.perf_counter() - start
    print(f"{label}: {elapsed:.3f}s")
    return elapsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        type=Path,
        default=Path("assets/images/cat_dog.jpg"),
        help="Path to input image",
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
    args = parser.parse_args()

    device = torch.device(args.device)
    model = build_sam3_image_model(
        device=args.device,
        eval_mode=True,
        enable_segmentation=True,
        num_feature_levels=args.num_feature_levels,
    )
    model.eval()

    image = _prepare_image(_load_image(args.image, device), size=1008)
    inputs = _make_inputs(1, 1008, 1008, str(device))

    decoder_inputs = None
    decoder_inputs_error = None
    with torch.no_grad():
        backbone_out = model.backbone.forward_image(image)
        text_encoder = model.backbone.language_backbone
        _, text_tokens = text_encoder.encoder(inputs[1])
        text_tokens = text_tokens.transpose(0, 1)
        text_memory = text_encoder.resizer(text_tokens)
        text_attention_mask = inputs[1].ne(0)
        text_attention_mask = text_attention_mask.ne(1)
        img_feats = backbone_out["backbone_fpn"][-1]
        img_pos = backbone_out["vision_pos_enc"][-1]
        if img_feats.shape[0] < 2:
            repeat = 2 // img_feats.shape[0]
            img_feats = img_feats.repeat(repeat, 1, 1, 1)
            img_pos = img_pos.repeat(repeat, 1, 1, 1)
            text_memory = text_memory.repeat(1, repeat, 1)
            text_attention_mask = text_attention_mask.repeat(repeat, 1)
        img_mask = torch.zeros(
            img_feats.shape[0],
            img_feats.shape[2],
            img_feats.shape[3],
            device=img_feats.device,
            dtype=torch.bool,
        )
        try:
            decoder_inputs = _make_decoder_only_inputs(model, inputs)
        except Exception as exc:
            decoder_inputs_error = exc

    def export_image_encoder():
        _export_image_encoder(model, image)

    def export_text_encoder():
        _export_text_encoder(model, inputs[1])

    def export_encoder_fusion():
        encoder_wrapper = (
            EncoderFusionWrapper(model.transformer.encoder).to(img_feats.device).eval()
        )
        if args.num_feature_levels != 1:
            raise RuntimeError("encoder_fusion export currently expects num_feature_levels=1")
        torch.export.export(
            encoder_wrapper,
            (img_feats, img_pos, img_mask, text_memory, text_attention_mask),
            dynamic_shapes={
                "img_feats": {0: torch.export.Dim.AUTO},
                "img_pos": {0: torch.export.Dim.AUTO},
                "img_mask": {0: torch.export.Dim.AUTO},
                "prompt": {
                    0: torch.export.Dim("seq", min=1, max=64),
                    1: torch.export.Dim.AUTO,
                },
                "prompt_mask": {
                    0: torch.export.Dim.AUTO,
                    1: torch.export.Dim("seq", min=1, max=64),
                },
            },
            strict=False,
            prefer_deferred_runtime_asserts_over_guards=True,
        )

    def export_full_pipeline():
        _export_full_sam3_pipeline(model, inputs)

    def export_decoder_only():
        if decoder_inputs_error is not None:
            raise decoder_inputs_error
        _export_decoder_only(model, decoder_inputs)

    with torch.no_grad():
        _time("Export image encoder", export_image_encoder)
        _time("Export text encoder", export_text_encoder)
        if args.num_feature_levels == 1:
            _time("Export encoder fusion", export_encoder_fusion)
        else:
            print("Export encoder fusion: skipped (num_feature_levels != 1)")
        try:
            _time("Export full pipeline", export_full_pipeline)
        except Exception as exc:
            print(f"Export full pipeline: failed ({type(exc).__name__}: {exc})")
        try:
            _time("Export decoder only", export_decoder_only)
        except Exception as exc:
            print(f"Export decoder only: failed ({type(exc).__name__}: {exc})")


if __name__ == "__main__":
    main()
