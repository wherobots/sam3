"""End-to-end export test for the SAM3 full pipeline.

Marked ``slow`` because it builds the real model (heavy memory + first call
allocates a CUDA model). Run explicitly with::

    pytest tests/export/test_full_pipeline_export.py -m slow
"""

from __future__ import annotations

import os

import pytest
import torch

from sam3.model_builder import build_sam3_image_model
from scripts.export_sam3_full_pipeline import (
    CONTEXT_LENGTH,
    INPUT_SIZE,
    export_full_pipeline,
)


def _device() -> torch.device:
    if os.getenv("SAM3_EXPORT_FORCE_CPU") == "1":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.mark.slow
def test_full_pipeline_export_traces_and_runs() -> None:
    device = _device()
    model = build_sam3_image_model(
        device=str(device),
        eval_mode=True,
        enable_segmentation=True,
        num_feature_levels=1,
    )
    model.eval()

    exported = export_full_pipeline(model, device=device, num_export_prompts=3)

    # The graph must accept variable batch size and variable num_prompts.
    eager_wrapper = exported.module()

    images = torch.randn(2, 3, INPUT_SIZE, INPUT_SIZE, device=device)
    token_ids = torch.zeros(4, CONTEXT_LENGTH, dtype=torch.long, device=device)
    token_ids[:, 0] = 49406

    with torch.no_grad():
        pred_logits, pred_boxes, pred_masks, presence = eager_wrapper(images, token_ids)

    bs_total = images.shape[0] * token_ids.shape[0]
    assert pred_logits.shape[0] == bs_total
    assert pred_boxes.shape[0] == bs_total
    assert pred_masks.shape[0] == bs_total
    if presence is not None:
        assert presence.shape[0] == bs_total
