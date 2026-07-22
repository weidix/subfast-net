# Frame Presence

`subfast-frame-presence` trains one PyTorch model to decide whether a complete video frame contains subtitles. It does not perform OCR or text decoding.

## Input Contract

- Each model item is one complete RGB frame.
- The only image transform is direct stretch resizing to `--image-size` (default `512x288`).
- There is no ROI crop, ROI mask, padding, color conversion, normalization, augmentation, cache-preparation command, or inference-time filtering.
- The model accepts only `images` with shape `N x 3 x H x W`. Batching never combines frames into one model input.

The dense subtitle mask is training supervision derived from the existing full-frame labels. It is not an input and is not used during inference. The scalar presence score is produced by the model's learned full-frame evidence head, with a fixed sigmoid threshold of `0.5`.

## Compact Model

Architecture v3 uses standard PyTorch only: four downsampling `Conv2d-BatchNorm-ReLU` blocks, one dilated context block, two bilinear top-down merges, a stride-4 dense region head, and one global evidence head. The inference graph has 73,130 parameters and 22 leaf modules (9 convolutions, 6 batch norms, 6 ReLUs, and 1 linear layer). It contains no platform-specific operators or calibrated runtime variant.

## Train

The canonical run uses all six configured generated roots and the held-out `data/validation_samples` root. It is limited to ten epochs by the CLI and settings model.

```bash
uv run subfast-frame-presence \
  --output-dir outputs/frame_presence_v3 \
  --epochs 10 \
  --device auto
```

The aggregate CLI uses the same spelling:

```bash
uv run subfast-net train frame-presence --output-dir outputs/frame_presence_v3 --epochs 10
```

`--train-root` may be repeated to replace the configured train roots. `--max-val-samples` is only for diagnostics; a limited validation set is explicitly marked incomplete and cannot satisfy the acceptance result.

## Outputs

Each run preserves the complete process rather than only final weights:

```text
outputs/frame_presence_v3/
├── run_config.json
├── source_snapshot.json
├── data_manifest_train.jsonl
├── data_manifest_validation.jsonl
├── metrics.jsonl
├── epoch_outputs/epoch_0001/
│   ├── checkpoint.pt
│   ├── metrics.json
│   └── validation_scores.jsonl
├── best.pt
├── best_inference.pt
├── last.pt
└── summary.json
```

`checkpoint.pt`, `best.pt`, and `last.pt` contain the model, optimizer, configuration, metrics, and random-generator state. `best_inference.pt` contains only the model and the fixed full-frame input contract. `summary.json` reports Recall, F1, false-positive and false-negative counts, and the score gap calculated as the minimum positive score minus the maximum negative score. A run is accepted only when all held-out requirements are met: Recall `1`, F1 `1`, no errors, and Gap at least `0.8`.

The canonical process stops at the first accepted epoch, or after epoch 10 when the requirements remain unmet; it does not append a second tuning stage.

## Verified Run

The canonical v3 run completed on 2026-07-22 and stopped at epoch 2:

| Model | Parameters | Leaf modules | Full-frame MPS FP32 latency | Recall | F1 | FP / FN | Gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v1 | 103,084 | 78 | 6.61 ms | 1 | 1 | 0 / 0 | 0.913935 |
| v3 | 73,130 | 22 | 0.84 ms | 1 | 1 | 0 / 0 | 0.828238 |

Latency is eager PyTorch, batch size 1, one complete `512x288` frame, 20 warmups, and 100 timed iterations on the same MPS device. The portable v3 checkpoint was also verified with an ordinary CPU forward pass.
