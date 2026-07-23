# Frame Presence

`subfast-frame-presence` trains one PyTorch model to decide whether a full frame, subtitle ROI, or arbitrary cropped image region contains subtitles. It does not perform OCR or text decoding.

## Input Contract

- Every model item is one RGB image and is stretch-resized to `--image-size` (default `512x288`).
- Training mixes the original full frames, the configured ROI datasets, and reproducible random crops generated from the full frames.
- Positive random crops fully retain at least one labeled subtitle box. Empty frames are cropped freely, so both subtitle occupancy and background coverage vary across epochs.
- Random crop coordinates are derived from the run seed, epoch, item index, and crop-view index. Resuming a run therefore reproduces the same samples.
- Inference accepts full frames, ROIs, and other image regions through the same input tensor. There is no ROI mask, padding, normalization, or inference-time filtering.
- The model accepts only `images` with shape `N x 3 x H x W`. Batching never combines frames into one model input.

The dense subtitle mask is training supervision derived from each source label and transformed into the crop coordinate space when needed. It is not an input and is not used during inference. The scalar presence score is produced by the model's learned evidence head, with a fixed sigmoid threshold of `0.5`.

## Compact Model

Architecture v4 uses standard PyTorch only. Four compact downsampling blocks produce a stride-8 dense region map and stride-16 context. One low-resolution dilated block supplies full-frame context, which is projected back into the region map with a single top-down merge. Compared with v3, it removes the second top-down merge and the high-resolution refinement block, reduces the first-stage width, and uses a `3x3` evidence window whose full-frame support matches the old stride-4 `5x5` window. There are no custom kernels, calibration, or platform-specific operators. The training graph has 48,082 parameters. The inference checkpoint deterministically folds its five BatchNorm operators into the preceding convolutions and has 47,914 parameters; this changes neither the learned function nor the input contract.

## Train

The default `train_roots` contains all six generated-frame roots and all six ROI roots. The default `val_roots` contains `data/validation_samples` and `data/roi_validation_samples`. ROI folders are identified by `summary.json` containing `roi_size`; no separate ROI root setting is required. Each source full frame contributes one full-frame item and one random-crop item; each ROI contributes one ROI item. Validation reports aggregate, full-frame, and ROI metrics separately, and early stopping requires both validation domains to pass.

```bash
uv run subfast-frame-presence \
  --output-dir outputs/frame_presence_v4_mixed \
  --epochs 10 \
  --device auto
```

The aggregate CLI uses the same spelling:

```bash
uv run subfast-net train frame-presence \
  --output-dir outputs/frame_presence_v4_mixed \
  --epochs 10
```

`--train-root` and `--val-root` may both be repeated with any mix of full-frame and ROI folders. Supplying either option replaces its configured list. `--random-crop-views` controls the number of crop items per full-frame image, while `--random-crop-min-scale` and `--random-crop-max-scale` control crop width and height relative to that frame. ROI images are not randomly cropped. `--max-val-samples` remains diagnostics-only; a limited validation set cannot satisfy acceptance.

For a single weight-initialized experiment, `--init-checkpoint` loads only model weights and resets the optimizer. It is distinct from `--resume` and permits a new dataset. Use `--no-early-stop` when the requested epoch count must run in full. The output records the base checkpoint path and SHA-256 plus validation metrics before the first update.

`--resume` is reserved for continuing an interrupted run with the same dataset and model settings. It restores the model, optimizer, completed epoch, and random-generator state from `last.pt`; an interrupted epoch is restarted from its beginning.

## Outputs

Each run preserves the complete process rather than only final weights:

```text
outputs/frame_presence_v4_mixed/
├── run_config.json
├── source_snapshot.json
├── source.patch
├── data_manifest_train.jsonl
├── data_manifest_validation.jsonl
├── metrics.jsonl
├── benchmark.json
├── epoch_outputs/epoch_0001/
│   ├── checkpoint.pt
│   ├── metrics.json
│   └── validation_scores.jsonl
├── best.pt
├── best_inference.pt
├── last.pt
└── summary.json
```

`checkpoint.pt`, `best.pt`, and `last.pt` contain the model, optimizer, configuration, metrics, and random-generator state. `best_inference.pt` contains the same learned model after deterministic Conv-BatchNorm folding and the fixed full-frame input contract. When the run starts from uncommitted code, `source.patch` records the exact diff from the Git revision in `source_snapshot.json`. `benchmark.json` is measured in a fresh process so that the completed training process cannot distort MPS latency. `summary.json` reports Recall, F1, false-positive and false-negative counts, score Gap, and the inference benchmark. A run is accepted only when all held-out requirements and the MPS efficiency target are met.

The canonical process stops at the first accepted epoch, or after epoch 10 when the requirements remain unmet; it does not append a second tuning stage.

The saved inference checkpoint can be benchmarked independently with the same CLI style as the other training families:

```bash
uv run subfast-net benchmark frame-presence outputs/frame_presence_v4/best_inference.pt --device mps
```

## Verified Run

The v3 baseline and original full-frame-only v4 run completed on 2026-07-22. These are historical baseline results; the mixed training workflow must be evaluated against both held-out domains before claiming equivalent quality:

| Model | Parameters | Leaf modules | Full-frame MPS FP32 latency | Recall | F1 | FP / FN | Gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v1 | 103,084 | 78 | 6.61 ms | 1 | 1 | 0 / 0 | 0.913935 |
| v3 | 73,130 | 22 | 0.84 ms | 1 | 1 | 0 / 0 | 0.828238 |
| v4 | 47,914 fused | 18 | 0.353327 ms | 1 | 1 | 0 / 0 | 0.972646 |

Latency is eager PyTorch FP32, batch size 1, one complete `512x288` frame already on the device, 20 warmups, and timed windows of 100 forwards. Image reading, resize, transfer, sigmoid, thresholding, and CPU output access are excluded. The v4 efficiency target is at least `2x` the v3 baseline, or no more than `0.42 ms` on the same MPS device.

## ROI Fine-Tune Diagnostic

The requested diagnostic run initialized from `outputs/frame_presence_v4/best.pt`, reset the optimizer, trained on `data/roi_samples1` through `data/roi_samples6` for all five epochs, and validated on `data/roi_validation_samples`. This is an ROI-domain experiment, not a replacement for the canonical full-frame run. Its complete artifacts are in `outputs/frame_presence_v4_roi_finetune`.

| Checkpoint and validation set | Accuracy | Recall | F1 | FP / FN | Gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| v4 base, ROI validation before fine-tuning | 0.701170 | 0.316872 | 0.481250 | 0 / 332 | -0.000032 |
| ROI fine-tune best (epoch 4), ROI validation | 1 | 1 | 1 | 0 / 0 | 0.999935 |
| v4 base, original full-frame validation | 1 | 1 | 1 | 0 / 0 | 0.972646 |
| ROI fine-tune best (epoch 4), original full-frame validation | 0.588659 | 0.063786 | 0.119461 | 2 / 455 | -0.521039 |
| ROI fine-tune last (epoch 5), original full-frame validation | 0.578758 | 0.041152 | 0.078740 | 2 / 466 | -0.819454 |

The ROI-only fine-tune causes severe forgetting on complete frames and therefore fails the full-frame acceptance contract. The cross-domain comparison and per-sample scores are stored under `outputs/frame_presence_v4_roi_finetune/evaluations/original_validation`.

Its fused v4 inference graph measured `2.154676 ms` median and `2.351103 ms` p90 on CPU with four PyTorch intra-op threads. This is batch-one `512x288` FP32 forward time only; `outputs/frame_presence_v4_roi_finetune/benchmark_cpu.json` records the complete benchmark contract.
