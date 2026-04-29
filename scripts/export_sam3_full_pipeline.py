"""
Export the full SAM3 image pipeline as a single ``torch.export`` program
suitable for ``torch.export.save`` / ``torch.export.load``.

Inputs to the exported program:
    images:    (B, 3, 1008, 1008) float32  -- B is dynamic, H/W are fixed
    token_ids: (P, 32)            int64    -- P is dynamic (>= 1), L=32 fixed

Outputs (4-tuple):
    pred_logits, pred_boxes, pred_masks, presence_logit_dec

This wrapper unifies the image encoder, text encoder, encoder fusion, and
decoder into a single graph so consumers can ship one ``.pt2`` artifact.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, cast

import torch
from torch.export.dynamic_shapes import Dim

from sam3.model.data_misc import FindStage
from sam3.model.geometry_encoders import Prompt
from sam3.model_builder import build_sam3_image_model

INPUT_SIZE = 1008
CONTEXT_LENGTH = 32


class FullSam3PipelineWrapper(torch.nn.Module):
    """Single-graph wrapper over the SAM3 grounding pipeline."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        images: torch.Tensor,
        token_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        model = cast(Any, self.model)
        device = images.device
        bs = images.shape[0] * token_ids.shape[0]

        img_ids = torch.arange(images.shape[0], device=device, dtype=torch.long)
        img_ids = img_ids.repeat_interleave(token_ids.shape[0])
        text_ids = torch.arange(token_ids.shape[0], device=device, dtype=torch.long)
        text_ids = text_ids.repeat(images.shape[0])

        # Text-only grounding: empty geometric prompts.
        box_embeddings = torch.zeros(1, bs, 4, device=device)
        box_mask = torch.ones(bs, 1, device=device, dtype=torch.bool)
        box_labels = torch.zeros(1, bs, device=device, dtype=torch.long)

        # Run the actual model under bf16 autocast on CUDA. The ViT MLP uses
        # sam3.perflib.fused.addmm_act which forces bf16 internally; without
        # autocast around the forward, the bf16 output collides with fp32
        # weights downstream. Sam3TrackingPredictor enters this same autocast
        # in __init__, so eager production runs already happen in bf16.
        autocast_ctx = (
            torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
            if device.type == "cuda"
            else torch.amp.autocast(device_type="cpu", enabled=False)
        )
        with autocast_ctx:
            backbone_out = model.backbone.forward_image(images)
            text_encoder = model.backbone.language_backbone
            _, text_tokens = text_encoder.encoder(token_ids)
            text_tokens = text_tokens.transpose(0, 1)
            text_memory = text_encoder.resizer(text_tokens)
            text_attention_mask = token_ids.ne(0).ne(1)
            backbone_out["language_features"] = text_memory
            backbone_out["language_mask"] = text_attention_mask

            find_input = FindStage(
                img_ids=img_ids,
                text_ids=text_ids,
                input_boxes=box_embeddings,
                input_boxes_mask=box_mask,
                input_boxes_label=box_labels,
                input_points=torch.zeros(0, bs, 2, device=device),
                input_points_mask=torch.zeros(bs, 0, device=device, dtype=torch.bool),
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
        # Cast outputs back to fp32 so downstream consumers don't have to.
        return (
            out["pred_logits"].float(),
            out["pred_boxes"].float(),
            out["pred_masks"].float(),
            out["presence_logit_dec"].float() if out.get("presence_logit_dec") is not None else None,
        )


def export_full_pipeline(
    model: torch.nn.Module,
    *,
    device: torch.device,
    num_export_prompts: int = 3,
) -> torch.export.ExportedProgram:
    """Trace ``FullSam3PipelineWrapper`` with dynamic batch and prompt dims.

    ``num_export_prompts`` must be >= 3 so the prompt dim is treated as
    dynamic during tracing (a length-2 example would let the tracer specialise
    the dim away).
    """
    if num_export_prompts < 3:
        raise ValueError("Use >= 3 prompts so the prompt dim stays dynamic")

    wrapper = FullSam3PipelineWrapper(model).to(device).eval()

    # Trace with batch=2 so Dim.AUTO doesn't specialize the batch dim away.
    images = torch.randn(2, 3, INPUT_SIZE, INPUT_SIZE, device=device)
    token_ids = torch.zeros(num_export_prompts, CONTEXT_LENGTH, dtype=torch.long, device=device)
    token_ids[:, 0] = 49406  # <|startoftext|> so attention mask is non-empty

    # Named Dim with min=1 so consumers can call with batch=1; Dim.AUTO would
    # take its min from the example shape (2) and refuse batch=1 at runtime.
    batch = Dim("batch", min=1)
    num_prompts = Dim("num_prompts", min=1)
    with torch.no_grad():
        return torch.export.export(
            wrapper,
            (images, token_ids),
            dynamic_shapes={
                "images": {0: batch, 2: INPUT_SIZE, 3: INPUT_SIZE},
                "token_ids": {0: num_prompts, 1: CONTEXT_LENGTH},
            },
            strict=False,
            prefer_deferred_runtime_asserts_over_guards=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/export/full_sam3_pipeline.pt2"),
        help="Destination .pt2 path",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--num-export-prompts",
        type=int,
        default=3,
        help="Prompts used during tracing — must be >= 3 to keep the dim dynamic",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    model = build_sam3_image_model(
        device=str(device),
        eval_mode=True,
        enable_segmentation=True,
        num_feature_levels=1,
    )
    model.eval()

    exported = export_full_pipeline(
        model, device=device, num_export_prompts=args.num_export_prompts
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.export.save(exported, str(args.out))
    print(f"Saved export to {args.out}")


if __name__ == "__main__":
    main()
