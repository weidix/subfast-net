# Inference Optimization Notes

> Historical record: this document preserves measurements from an external Rust/MPSGraph runtime. This repository provides the Python export command and artifacts only; it does not contain that runtime or a `cargo` command.

This document records the unified export settings used for that experiment.

## Summary

The optimized runtime artifact is exported from the same trained checkpoint. It does not require retraining when using the existing `512x512` model.

Optimized export:

```bash
uv run subfast-net export unified \
  --batch-size 16 \
  --head-output \
  outputs/full_512/best.pt \
  outputs/full_512/unified_batch16_head
```

For the previously tested `full_256` checkpoint, the optimized artifact was:

```text
outputs/full_256/unified_batch16_head/model.json
outputs/full_256/unified_batch16_head/weights.bin
```

## What Changed

| Change | Effect | Retrain needed |
|---|---|---:|
| Conv + BatchNorm folding | Merges inference BatchNorm into Conv weights and bias, reducing graph nodes | No |
| `--batch-size 16` | Exports a batch-16 input shape so MPSGraph run overhead is amortized | No |
| `--head-output` | Uses detector head logits directly instead of final full-resolution upsample | No |

The optimized graph shape is:

```text
input:  [16, 3, 512, 512]
output: [16, 2, 128, 128]
```

The ordinary graph shape is:

```text
input:  [1, 3, 512, 512]
output: [1, 2, 512, 512]
```

## Why `--batch-size 16`

Single-image MPSGraph execution has noticeable fixed overhead. Batch-16 reduces average model time per image by running multiple images per graph execution.

This is a throughput optimization. It is best for batch/offline processing. It is not the same as single-image request latency.

Runtime behavior:

- Preprocess images one by one.
- Stack up to 16 preprocessed tensors.
- Run one MPSGraph call.
- Split the batch output back into per-image records.
- Pad the final short batch internally when fewer than 16 images remain.

## Why `--head-output`

The model head naturally produces logits at `128x128` for a `512x512` input. The original model forward upsamples that output back to `512x512`.

For detection postprocess, using the `128x128` head output is enough as long as the box coordinates are mapped back to padded/input coordinates. This reduces postprocess work because it processes 16x fewer pixels.

## Measured Speed

Measured on the 300-image validation set through the Rust MPSGraph runtime.

| Artifact | Inference | Postprocess | Subtitle outputs |
|---|---:|---:|---:|
| Original single-image, `512x512` output | `4.599 ms/img` | `1.054 ms/img` | `150/300` |
| Optimized batch-16, head output | `2.581 ms/img` | `0.070 ms/img` | `150/300` |

Interpretation:

- `inference` is model forward time only.
- `postprocess` is sigmoid, thresholds, connected components, and box generation.
- These numbers do not include image decode, resize, normalization, JSONL writing, or full loop overhead.
- `ms/img` for batch-16 is average per image after dividing batch execution time across the images in the batch.

## Measured Effect

Validation metrics after mapping predictions to the same padded coordinate space:

| Metric | Original | Optimized |
|---|---:|---:|
| TP | 146 | 146 |
| FP | 13 | 10 |
| FN | 20 | 20 |
| Precision | 0.9182 | 0.9359 |
| Recall | 0.8795 | 0.8795 |
| F1 | 0.8985 | 0.9068 |
| Mean best IoU | 0.7215 | 0.7070 |

Conclusion: the optimized artifact did not reduce validation F1 on this set. Box boundaries are slightly coarser because postprocess works from `128x128` logits, so mean best IoU is slightly lower.

## When Retraining Is Needed

No retraining is needed for:

- BatchNorm folding.
- Changing exported batch size.
- Using `--head-output` with the same input size.

Retraining or at least fresh held-out validation is needed if changing the model input size, for example exporting/running the same weights at `256x256`.

In the current experiment, `256x256` batch-16 reached about `0.894 ms/img` inference, but subtitle outputs dropped from `150` to `114` at default thresholds. That is not an equivalent optimization of the existing model.

## Commands

Export optimized artifact from `full_512`:

```bash
uv run subfast-net export unified \
  --batch-size 16 \
  --head-output \
  outputs/full_512/best.pt \
  outputs/full_512/unified_batch16_head
```

Export ordinary artifact from `full_512`:

```bash
uv run subfast-net export unified \
  outputs/full_512/best.pt \
  outputs/full_512/unified
```

The reported runtime validation was performed outside this repository with the exported `model.json` and `weights.bin`. It is retained as historical measurement context, not as a runnable project command.
