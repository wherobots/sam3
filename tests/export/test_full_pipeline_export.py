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
    FullSam3PipelineWrapper,
    export_full_pipeline,
)


def _device() -> torch.device:
    if os.getenv("SAM3_EXPORT_FORCE_CPU") == "1":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="module")
def sam3_model() -> torch.nn.Module:
    device = _device()
    model = build_sam3_image_model(
        device=str(device),
        eval_mode=True,
        enable_segmentation=True,
        num_feature_levels=1,
    )
    model.eval()
    return model


@pytest.mark.slow
def test_full_pipeline_export_matches_eager(sam3_model: torch.nn.Module) -> None:
    """Exported and eager outputs must agree on the same input."""
    device = _device()
    torch.manual_seed(0)
    images = torch.randn(2, 3, INPUT_SIZE, INPUT_SIZE, device=device)
    token_ids = torch.zeros(3, CONTEXT_LENGTH, dtype=torch.long, device=device)
    token_ids[:, 0] = 49406

    wrapper = FullSam3PipelineWrapper(sam3_model).to(device).eval()
    with torch.no_grad():
        eager_out = wrapper(images, token_ids)

    ep = export_full_pipeline(sam3_model, device=device, num_export_prompts=3)
    with torch.no_grad():
        exported_out = ep.module()(images, token_ids)

    assert len(eager_out) == len(exported_out) == 4
    for idx, (e, x) in enumerate(zip(eager_out, exported_out)):
        if e is None and x is None:
            continue
        assert e is not None and x is not None, f"output {idx} disagrees on None-ness"
        torch.testing.assert_close(e, x, rtol=0, atol=0, msg=f"output {idx} differs")


@pytest.mark.slow
def test_full_pipeline_export_save_load_roundtrip(
    sam3_model: torch.nn.Module, tmp_path
) -> None:
    """Save the export to a .pt2, reload, and confirm it still runs."""
    import torchvision.ops  # noqa: F401  -- registers roi_align before load

    device = _device()
    ep = export_full_pipeline(sam3_model, device=device, num_export_prompts=3)
    out_path = tmp_path / "full_sam3_pipeline.pt2"
    torch.export.save(ep, str(out_path))

    loaded = torch.export.load(str(out_path))
    images = torch.randn(2, 3, INPUT_SIZE, INPUT_SIZE, device=device)
    token_ids = torch.zeros(3, CONTEXT_LENGTH, dtype=torch.long, device=device)
    token_ids[:, 0] = 49406
    with torch.no_grad():
        out = loaded.module()(images, token_ids)
    assert out[0].shape[0] == images.shape[0] * token_ids.shape[0]


@pytest.mark.slow
@pytest.mark.parametrize(("batch", "num_prompts"), [(1, 1), (1, 2), (3, 2), (2, 4)])
def test_full_pipeline_export_supports_dynamic_shapes(
    sam3_model: torch.nn.Module, batch: int, num_prompts: int
) -> None:
    """The exported program must accept the dynamic batch / num_prompts grid."""
    device = _device()
    ep = export_full_pipeline(sam3_model, device=device, num_export_prompts=3)
    module = ep.module()

    images = torch.randn(batch, 3, INPUT_SIZE, INPUT_SIZE, device=device)
    token_ids = torch.zeros(num_prompts, CONTEXT_LENGTH, dtype=torch.long, device=device)
    token_ids[:, 0] = 49406

    with torch.no_grad():
        pred_logits, pred_boxes, pred_masks, presence = module(images, token_ids)

    bs_total = batch * num_prompts
    assert pred_logits.shape[0] == bs_total
    assert pred_boxes.shape[0] == bs_total
    assert pred_masks.shape[0] == bs_total
    assert presence is not None and presence.shape[0] == bs_total
