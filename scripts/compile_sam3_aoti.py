import argparse
from pathlib import Path

import torch
import torchvision.ops  # noqa: F401


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("artifacts/export/full_sam3_pipeline.pt2"),
        help="Path to exported full pipeline pt2",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/aoti/full_sam3_pipeline_aoti.pt2"),
        help="Output path for the AOTInductor package",
    )
    parser.add_argument(
        "--device-index",
        type=int,
        default=None,
        help="CUDA device index to compile on",
    )
    parser.add_argument(
        "--max-autotune",
        action="store_true",
        help="Enable max_autotune for AOTInductor",
    )
    args = parser.parse_args()

    if args.device_index is not None and torch.cuda.is_available():
        torch.cuda.set_device(args.device_index)
    if not torch.cuda.is_available():
        print("Warning: CUDA not available; AOTInductor will target CPU.")

    exported = torch.export.load(str(args.input))
    inductor_configs = {"max_autotune": True} if args.max_autotune else None

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_path = torch._inductor.aoti_compile_and_package(
        exported,
        package_path=str(args.output),
        inductor_configs=inductor_configs,
    )
    print("Saved AOTInductor package to", output_path)


if __name__ == "__main__":
    main()
