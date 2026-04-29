from __future__ import annotations

from typing import Any, cast

import pytest
import torch

from sam3.model.data_misc import FindStage
from sam3.model.geometry_encoders import Prompt
from tests.export.utils import capture_stderr_on_fail, get_device, save_output_shapes


class FullSam3PipelineWrapper(torch.nn.Module):
    def __init__(self, model: Any):
        super().__init__()
        self.model = model

    def forward(
        self,
        images: torch.Tensor,
        token_ids: torch.Tensor,
    ):
        model = cast(Any, self.model)
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

        return (
            out["pred_logits"],
            out["pred_boxes"],
            out["pred_masks"],
            out.get("presence_logit_dec"),
        )


def _make_inputs(batch: int, height: int, width: int, device: str):
    images = torch.randn(batch, 3, height, width, device=device)
    token_ids = torch.ones(batch, 32, device=device, dtype=torch.long)
    token_ids[:, -1] = 0
    return images, token_ids


def _make_decoder_only_inputs(model: Any, inputs):
    images, token_ids = inputs
    model_any = cast(Any, model)
    model_any = model_any.eval()
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

    backbone_out = model_any.backbone.forward_image(images)
    text_encoder = model_any.backbone.language_backbone
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
        input_points=torch.zeros(0, bs, 2, device=device),
        input_points_mask=torch.zeros(bs, 0, device=device, dtype=torch.bool),
    )
    geometric_prompt = Prompt(
        box_embeddings=box_embeddings,
        box_mask=box_mask,
        box_labels=box_labels,
    )

    prompt, prompt_mask, backbone_out = model_any._encode_prompt(
        backbone_out, find_input, geometric_prompt
    )
    backbone_out, encoder_out, _ = model_any._run_encoder(
        backbone_out, find_input, prompt, prompt_mask
    )
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


def _export_full_sam3_pipeline(model: Any, inputs):
    images, token_ids = inputs
    device = images.device
    wrapper = FullSam3PipelineWrapper(model).to(device).eval()  # type: ignore[arg-type]
    if images.shape[0] < 2:
        repeat = 2 // images.shape[0]
        export_inputs = (
            images.repeat(repeat, 1, 1, 1),
            token_ids.repeat(repeat, 1),
        )
    else:
        export_inputs = inputs
    with torch.no_grad():
        return torch.export.export(
            wrapper,
            export_inputs,
            dynamic_shapes={
                "images": {
                    0: torch.export.Dim.AUTO,
                    2: 1008,
                    3: 1008,
                },
                "token_ids": {
                    0: torch.export.Dim("num_prompts", min=1),
                    1: 32,
                },
            },
            strict=False,
            prefer_deferred_runtime_asserts_over_guards=True,
        )


class DecoderOnlyWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(
        self,
        backbone_fpn,
        img_ids: torch.Tensor,
        memory: torch.Tensor,
        pos_embed: torch.Tensor,
        prompt: torch.Tensor,
        prompt_mask: torch.Tensor,
        level_start_index: torch.Tensor,
        spatial_shapes: torch.Tensor,
        valid_ratios: torch.Tensor,
    ):
        model = cast(Any, self.model)
        vis_feat_sizes = [(feat.shape[-2], feat.shape[-1]) for feat in backbone_fpn]
        encoder_out = {
            "pos_embed": pos_embed,
            "padding_mask": None,
            "level_start_index": level_start_index,
            "spatial_shapes": spatial_shapes,
            "valid_ratios": valid_ratios,
            "vis_feat_sizes": vis_feat_sizes,
        }
        out = {"encoder_hidden_states": memory}
        out, hs = model._run_decoder(
            memory=memory,
            pos_embed=pos_embed,
            src_mask=None,
            out=out,
            prompt=prompt,
            prompt_mask=prompt_mask,
            encoder_out=encoder_out,
        )
        backbone_out = {"backbone_fpn": backbone_fpn}
        model._run_segmentation_heads(
            out=out,
            backbone_out=backbone_out,
            img_ids=img_ids,
            vis_feat_sizes=vis_feat_sizes,
            encoder_hidden_states=out["encoder_hidden_states"],
            prompt=prompt,
            prompt_mask=prompt_mask,
            hs=hs,
        )
        return (
            out["pred_logits"],
            out["pred_boxes"],
            out["pred_masks"],
            out["pred_boxes_xyxy"],
        )


def _export_decoder_only(model: Any, inputs):
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
    ) = inputs
    device = memory.device
    wrapper = DecoderOnlyWrapper(model).to(device).eval()  # type: ignore[arg-type]
    dynamic_shapes = [
        [{0: torch.export.Dim.AUTO} for _ in backbone_fpn],
        {0: torch.export.Dim.AUTO},
        {1: torch.export.Dim.AUTO},
        {1: torch.export.Dim.AUTO},
        {1: torch.export.Dim.AUTO},
        {0: torch.export.Dim.AUTO},
        {},
        {},
        {0: torch.export.Dim.AUTO},
    ]
    with torch.no_grad():
        exported = torch.export.export(
            wrapper,
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
            ),
            dynamic_shapes=dynamic_shapes,
            strict=False,
            prefer_deferred_runtime_asserts_over_guards=True,
        )
    return exported


@pytest.mark.slow
def test_decoder_export_static(sam3_model):
    device = get_device()
    inputs = _make_inputs(1, 1008, 1008, device)
    with capture_stderr_on_fail("export_static"):
        exported = _export_full_sam3_pipeline(sam3_model, inputs)
    assert exported is not None


@pytest.mark.slow
def test_decoder_export_loads(sam3_model):
    device = get_device()
    inputs = _make_inputs(1, 1008, 1008, device)
    with capture_stderr_on_fail("export_loads"):
        exported = _export_full_sam3_pipeline(sam3_model, inputs)
    module = exported.module()
    with torch.no_grad():
        out = module(*inputs)
    assert isinstance(out, tuple)
    assert len(out) == 4


@pytest.mark.slow
def test_decoder_export_matches_eager(sam3_model):
    device = get_device()
    inputs = _make_inputs(1, 1008, 1008, device)
    wrapper = FullSam3PipelineWrapper(sam3_model).to(device).eval()
    with torch.no_grad():
        eager_out = wrapper(*inputs)
    with capture_stderr_on_fail("export_match"):
        exported = _export_full_sam3_pipeline(sam3_model, inputs)
    module = exported.module()
    with torch.no_grad():
        export_out = module(*inputs)
    save_output_shapes("full_pipeline_eager", inputs, eager_out)
    save_output_shapes("full_pipeline_export", inputs, export_out)
    for eager, compiled in zip(eager_out, export_out):
        if eager is None:
            assert compiled is None
        else:
            torch.testing.assert_close(eager, compiled, rtol=1e-3, atol=1e-3)


@pytest.mark.slow
@pytest.mark.parametrize("batch", [1, 2])
def test_full_sam3_pipeline_export_inference_shapes(sam3_model, batch: int):
    device = get_device()
    inputs = _make_inputs(1, 1008, 1008, device)
    with capture_stderr_on_fail("export_inference_shapes"):
        exported = _export_full_sam3_pipeline(sam3_model, inputs)
    module = exported.module()
    new_inputs = _make_inputs(batch, 1008, 1008, device)
    with torch.no_grad():
        out = module(*new_inputs)
    save_output_shapes(f"full_pipeline_export_batch_{batch}", new_inputs, out)
    assert isinstance(out, tuple)


@pytest.mark.slow
def test_decoder_only_export_loads(sam3_model):
    device = get_device()
    inputs = _make_inputs(1, 1008, 1008, device)
    decoder_inputs = _make_decoder_only_inputs(sam3_model, inputs)
    with capture_stderr_on_fail("export_decoder_only_loads"):
        exported = _export_decoder_only(sam3_model, decoder_inputs)
    module = exported.module()
    with torch.no_grad():
        out = module(*decoder_inputs)
    assert isinstance(out, tuple)
    assert len(out) == 4


@pytest.mark.slow
def test_decoder_only_export_matches_eager(sam3_model):
    device = get_device()
    inputs = _make_inputs(1, 1008, 1008, device)
    decoder_inputs = _make_decoder_only_inputs(sam3_model, inputs)
    wrapper = DecoderOnlyWrapper(sam3_model).to(device).eval()
    with torch.no_grad():
        eager_out = wrapper(*decoder_inputs)
    with capture_stderr_on_fail("export_decoder_only_match"):
        exported = _export_decoder_only(sam3_model, decoder_inputs)
    module = exported.module()
    with torch.no_grad():
        export_out = module(*decoder_inputs)
    save_output_shapes("decoder_only_eager", None, eager_out)
    save_output_shapes("decoder_only_export", None, export_out)
    for eager, compiled in zip(eager_out, export_out):
        torch.testing.assert_close(eager, compiled, rtol=1e-3, atol=1e-3)
