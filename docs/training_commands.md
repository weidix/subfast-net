# Training Commands

This document lists the current training and export commands for the PyTorch subtitle detector.

## Full Training

Train a `512x512` model with the current full local dataset:

```bash
uv run subfast-net train \
  --train-root data/generated_samples1 \
  --train-root data/generated_samples2 \
  --train-root data/generated_samples3 \
  --train-root data/generated_samples4 \
  --train-root data/mixed_subtitle_samples \
  --val-root data/validation_samples \
  --output-dir outputs/full_512 \
  --image-size 512 \
  --batch-size 8 \
  --epochs 10 \
  --lr 0.0003 \
  --max-train-samples 19000 \
  --negative-ratio 0.35 \
  --log-interval 10 \
  --device auto
```

The same command for a separate new run:

```bash
uv run subfast-net train \
  --train-root data/generated_samples1 \
  --train-root data/generated_samples2 \
  --train-root data/generated_samples3 \
  --train-root data/generated_samples4 \
  --train-root data/mixed_subtitle_samples \
  --val-root data/validation_samples \
  --output-dir outputs/full_512_new \
  --image-size 512 \
  --batch-size 8 \
  --epochs 10 \
  --lr 0.0003 \
  --max-train-samples 19000 \
  --negative-ratio 0.35 \
  --log-interval 10 \
  --device auto
```

## Quick Smoke Training

Use this only to check that the training pipeline runs:

```bash
uv run subfast-net train \
  --train-root data/generated_samples1 \
  --val-root data/validation_samples \
  --output-dir outputs/smoke_train \
  --image-size 256 \
  --batch-size 4 \
  --epochs 1 \
  --max-train-samples 64 \
  --max-val-samples 32 \
  --log-interval 5 \
  --device auto
```

Smoke runs are not training-quality evidence. Use held-out validation from a full run before comparing model quality.

## Resume Training

Resume from an output directory:

```bash
uv run subfast-net train \
  --train-root data/generated_samples1 \
  --train-root data/generated_samples2 \
  --train-root data/generated_samples3 \
  --train-root data/generated_samples4 \
  --train-root data/mixed_subtitle_samples \
  --val-root data/validation_samples \
  --output-dir outputs/full_512 \
  --resume outputs/full_512 \
  --image-size 512 \
  --batch-size 8 \
  --epochs 15 \
  --lr 0.0003 \
  --max-train-samples 19000 \
  --negative-ratio 0.35 \
  --log-interval 10 \
  --device auto
```

`--resume` accepts:

- A checkpoint file such as `outputs/full_512/best.pt`.
- An epoch checkpoint directory such as `outputs/full_512/epoch_outputs/epoch_0010`.
- An output directory such as `outputs/full_512`.

When resuming from an output directory, the code picks the latest `epoch_outputs/epoch_*/model.pt` if present, otherwise `best.pt`.

## Export Unified Runtime Artifacts

Optimized runtime artifact:

```bash
uv run subfast-net export-unified \
  --batch-size 16 \
  --head-output \
  outputs/full_512/best.pt \
  outputs/full_512/unified_batch16_head
```

Ordinary single-image artifact:

```bash
uv run subfast-net export-unified \
  outputs/full_512/best.pt \
  outputs/full_512/unified
```

For details about `--batch-size 16` and `--head-output`, see:

```text
docs/inference_optimization.md
```

## Output Files

Each training run writes these files under `--output-dir`:

| Path | Meaning |
|---|---|
| `best.pt` | Best checkpoint by validation F1 |
| `metrics.jsonl` | Train-step and validation metrics |
| `summary.json` | Last validation summary |
| `epoch_outputs/epoch_000N/model.pt` | Per-epoch checkpoint |
| `epoch_outputs/epoch_000N/outputs.json` | Sample validation outputs for that epoch |

`metrics.jsonl` includes both `train_step` rows and `validation` rows. The terminal output is intentionally compact; use the files for detailed inspection.

## Common Parameters

| Parameter | Meaning |
|---|---|
| `--train-root` | Training dataset root. Repeat it for multiple roots. |
| `--val-root` | Validation dataset root. |
| `--output-dir` | Run output directory. |
| `--image-size` | Letterbox target size used by training and validation. |
| `--batch-size` | PyTorch training batch size. |
| `--epochs` | Final epoch number to run to. For resume, set this higher than the completed epoch. |
| `--lr` | AdamW learning rate. |
| `--max-train-samples` | Cap the training sample count. |
| `--max-val-samples` | Cap the validation sample count. |
| `--positive-ratio` | Target ratio of labeled samples in the limited training set. |
| `--negative-ratio` | Target ratio of empty-label samples in the limited training set. |
| `--val-positive-ratio` | Target ratio of labeled samples in the limited validation set. |
| `--val-negative-ratio` | Target ratio of empty-label samples in the limited validation set. |
| `--log-interval` | Step interval for writing train-step rows to `metrics.jsonl`. |
| `--no-epoch-outputs` | Disable per-epoch validation sample output JSON. |
| `--device` | `auto`, `cpu`, `mps`, or `cuda` depending on local availability. |
