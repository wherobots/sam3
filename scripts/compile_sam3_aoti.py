import argparse
import torch
import torchvision.ops  # noqa: F401


torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

exported = torch.export.load("artifacts/export/full_sam3_pipeline.pt2")
torch._inductor.aoti_compile_and_package(
    exported,
    package_path="artifacts/aoti/full_sam3_pipeline_aoti.pt2",
)
