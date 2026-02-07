import argparse
import sys
import time
from pathlib import Path

import torch
import torchvision.ops  # noqa: F401

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact",
        type=Path,
        default=Path("artifacts/export/full_sam3_pipeline.pt2"),
        help="Path to exported full pipeline artifact",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    start = time.perf_counter()
    module = torch.export.load(str(args.artifact)).module()
    module.to(args.device)
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    print(f"Load full pipeline to {args.device}: {elapsed:.3f}s")


if __name__ == "__main__":
    main()
