import argparse
from typing import Iterable

import torch

from sam3.model.data_misc import FindStage
from sam3.model.geometry_encoders import Prompt
from sam3.model_builder import build_sam3_image_model


SAM3_INPUT_SIZE = 1008
SAM3_CONTEXT_LENGTH = 32


class TwoInputWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        images: torch.Tensor,
        token_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
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

        backbone_out = self.model.backbone.forward_image(images)
        text_encoder = self.model.backbone.language_backbone
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
        out = self.model.forward_grounding(
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


def _parse_int_list(value: str) -> list[int]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    return [int(item) for item in items]


def _make_token_ids(
    model: torch.nn.Module, prompts: Iterable[str], device: torch.device
) -> torch.Tensor:
    tokenizer = model.backbone.language_backbone.tokenizer
    return tokenizer(list(prompts), context_length=SAM3_CONTEXT_LENGTH).to(device)


def _run_export(
    model: torch.nn.Module,
    device: torch.device,
    export_prompts: list[str],
    image_batch: int,
    dynamic_images: bool,
) -> torch.export.ExportedProgram:
    image = torch.randn(image_batch, 3, SAM3_INPUT_SIZE, SAM3_INPUT_SIZE, device=device)
    token_ids = _make_token_ids(model, export_prompts, device)
    wrapper = TwoInputWrapper(model).to(device).eval()
    images_dim = torch.export.Dim.AUTO if dynamic_images else image_batch
    num_prompts_dim = torch.export.Dim("num_prompts", min=1)
    with torch.no_grad():
        return torch.export.export(
            wrapper,
            (image, token_ids),
            dynamic_shapes={
                "images": {0: images_dim, 2: SAM3_INPUT_SIZE, 3: SAM3_INPUT_SIZE},
                "token_ids": {0: num_prompts_dim, 1: SAM3_CONTEXT_LENGTH},
            },
            strict=False,
            prefer_deferred_runtime_asserts_over_guards=True,
        )


def _run_inference(
    model: torch.nn.Module,
    module: torch.nn.Module,
    device: torch.device,
    image_batch: int,
    prompt_count: int,
) -> None:
    prompts = [f"prompt_{idx}" for idx in range(prompt_count)]
    image = torch.randn(image_batch, 3, SAM3_INPUT_SIZE, SAM3_INPUT_SIZE, device=device)
    token_ids = _make_token_ids(model, prompts, device)
    with torch.no_grad():
        out = module(image, token_ids)
    print(f"run prompts={prompt_count} outputs={len(out)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional path to sam3 checkpoint.",
    )
    parser.add_argument(
        "--image-batch",
        type=int,
        default=1,
        help="Number of images for export and inference.",
    )
    parser.add_argument(
        "--export-prompts",
        type=str,
        default="cat,dog,building",
        help="Comma-separated prompts for export sample.",
    )
    parser.add_argument(
        "--test-prompts",
        type=str,
        default="1,2,5",
        help="Comma-separated prompt counts to run after export.",
    )
    parser.add_argument(
        "--dynamic-images",
        action="store_true",
        help="Export with a dynamic image batch dimension.",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    print("torch", torch.__version__, "cuda", torch.version.cuda, "device", device)
    model = build_sam3_image_model(
        device=str(device),
        eval_mode=True,
        enable_segmentation=True,
        checkpoint_path=args.checkpoint,
        load_from_HF=args.checkpoint is None,
    )
    model.eval()

    export_prompts = [p.strip() for p in args.export_prompts.split(",") if p.strip()]
    test_prompt_counts = _parse_int_list(args.test_prompts)

    exported = _run_export(
        model=model,
        device=device,
        export_prompts=export_prompts,
        image_batch=args.image_batch,
        dynamic_images=args.dynamic_images,
    )
    module = exported.module()
    for prompt_count in test_prompt_counts:
        try:
            _run_inference(
                model=model,
                module=module,
                device=device,
                image_batch=args.image_batch,
                prompt_count=prompt_count,
            )
        except Exception as exc:
            print(f"run prompts={prompt_count} failed: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
