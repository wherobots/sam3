# Export Progress Summary

## Current state

- **Export scripts** now work end-to-end for the single-level path and produce PT2
  artifacts for image encoder, text encoder, encoder fusion, and decoder.
- **Test script** runs the exported artifacts, checks logits against eager, and
  writes mask/box overlays.
- **Decoder export** is pinned to a **minimum batch of 2** to avoid export guard
  failures. The test script pads inputs to batch >=2 when needed, then slices
  outputs back to the original prompt count.

## Changes made

- `scripts/export_sam3_artifacts.py`
  - Use single-level features (`backbone_fpn[-1]`) for encoder fusion export.
  - Align image feature batch with prompt batch (repeat if needed).
  - Use `torch.export.save(...)` for PT2 artifacts.
  - Export decoder with a minimal prompt list (single prompt) and updated decoder
    export logic to keep batch dynamic.

- `scripts/test_sam3_artifacts.py`
  - Use single-level features for encoder fusion.
  - Repeat image features to match prompt batch for encoder fusion.
  - Pad decoder inputs to batch >=2 if needed; compare against eager on the same
    padded inputs and slice outputs back to the original prompt count.
  - Resize masks to the input image for overlays and guard box drawing against
    size mismatches.

- `tests/export/test_decoder_export.py`
  - Updated decoder export helper to attempt batch=1 (dynamic) and fall back to
    batch=2. In practice, batch=2 is the stable path.

- New script: `scripts/benchmark_sam3_artifacts.py`
  - Benchmarks exported image encoder, text encoder, encoder fusion, decoder, and
    compares to full eager inference time.

## Open items

- Decide whether to keep decoder export pinned to batch >=2 or revisit batch=1
  export constraints.
- Integrate the new benchmark script into docs or CI if desired.
