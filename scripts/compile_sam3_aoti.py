"""
AOT-Inductor compile the exported SAM3 full pipeline ``.pt2``.

Loads an ExportedProgram saved by ``scripts/export_sam3_full_pipeline.py``
and packages it into an AOTI ``.pt2`` runner that can be loaded with
``torch._inductor.aoti_load_package``.

Usage::

    python scripts/compile_sam3_aoti.py \
        --in artifacts/export/full_sam3_pipeline.pt2 \
        --out artifacts/aoti/full_sam3_pipeline_aoti.pt2

Environment requirements (CUDA build):
- ``CUDA_HOME`` must point at a CUDA toolkit with ``nvcc``. If the system
  doesn't have one installed, the simplest path is::

      uv pip install --index-url https://pypi.nvidia.com \\
          nvidia-cuda-nvcc nvidia-cuda-cccl
      export CUDA_HOME=$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cu13
      export PATH=$CUDA_HOME/bin:$PATH

  The torch wheels (``torch==2.10.0+cu128``) do not bundle ``nvcc`` —
  only the runtime libraries.
- ``import torchvision.ops`` runs before ``torch.export.load`` so that
  ``torch.ops.torchvision.roi_align.default`` is registered before the
  serialized graph references it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch._inductor
import torch._inductor.codecache  # noqa: F401  -- workaround for torch 2.10 aoti_load_package
import torchvision.ops  # noqa: F401  -- registers roi_align before load


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--in",
        dest="in_path",
        type=Path,
        default=Path("artifacts/export/full_sam3_pipeline.pt2"),
        help="Path to the exported .pt2 (output of export_sam3_full_pipeline.py)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/aoti/full_sam3_pipeline_aoti.pt2"),
        help="Destination AOTI-packaged .pt2",
    )
    args = parser.parse_args()

    if not args.in_path.exists():
        raise FileNotFoundError(f"{args.in_path} not found — run export first")

    print(f"Loading exported program from {args.in_path}")
    exported = torch.export.load(str(args.in_path))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    print(f"Compiling with torch._inductor.aoti_compile_and_package -> {args.out}")
    torch._inductor.aoti_compile_and_package(
        exported,
        package_path=str(args.out),
    )
    print(f"Saved AOTI package to {args.out}")


if __name__ == "__main__":
    main()
