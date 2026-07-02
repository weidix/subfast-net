# ROI Presence + Embedding Training Commands

This document lists the current commands for the ROI subtitle Presence + Embedding model.

This is separate from bbox detector training. Use `train-roi` and `validate-roi` for ROI Presence + Embedding checkpoints.

## Current Data Roots

| Root | Role | Samples | Positive | Empty | ROI size |
|---|---|---:|---:|---:|---|
| `data/roi_samples1` | train | 2725 | 1984 | 741 | `1036x96` |
| `data/roi_samples2` | train | 2001 | 1312 | 689 | `833x492` |
| `data/roi_samples4` | train | 7021 | 1796 | 5225 | `920x364` |
| `data/roi_samples5` | train | 7111 | 1706 | 5405 | `1006x140` |
| `data/roi_samples6` | train | 9247 | 3039 | 6208 | `1032x180` |
| `data/roi_validation_samples` | eval | 1111 | 486 | 625 | `1292x131` |
| `data/roi_samples3` | eval | 943 | 414 | 529 | `1290x204` |

These roots have different native ROI sizes, so multi-root training must use explicit deterministic resize.

## Full ROI Training

```bash
uv run subfast-net train-roi \
  --train-root data/roi_samples1 \
  --train-root data/roi_samples2 \
  --train-root data/roi_samples4 \
  --train-root data/roi_samples5 \
  --train-root data/roi_samples6 \
  --val-root data/roi_validation_samples \
  --output-dir outputs/roi_presence_embedding_full \
  --resize-roi 256x64 \
  --batch-size 32 \
  --epochs 10 \
  --lr 0.0003 \
  --negative-ratio 0.35 \
  --log-interval 50 \
  --device auto
```

## Effect Check Run

This is the shorter run used to verify the current training path:

```bash
uv run subfast-net train-roi \
  --train-root data/roi_samples1 \
  --train-root data/roi_samples2 \
  --train-root data/roi_samples4 \
  --train-root data/roi_samples5 \
  --train-root data/roi_samples6 \
  --val-root data/roi_validation_samples \
  --output-dir outputs/roi_presence_embedding_effect_check \
  --resize-roi 256x64 \
  --batch-size 32 \
  --epochs 5 \
  --lr 0.0003 \
  --max-train-samples 4000 \
  --max-val-samples 600 \
  --negative-ratio 0.35 \
  --val-negative-ratio 0.35 \
  --width 16 \
  --log-interval 50 \
  --device auto
```

Embedding metrics now use trusted pairs only:

- Local positive pairs: same root, same video, both subtitle-present, frame distance within `--embedding-pair-frame-window`, same `segment_id`.
- Local negative pairs: same root, same video, both subtitle-present, frame distance within `--embedding-pair-frame-window`, different `segment_id`.
- OCR negative pairs: same root, both subtitle-present, different `segment_id`, usable OCR text on both sides, and normalized OCR similarity at or below `--embedding-ocr-negative-max-similarity`.

No-subtitle samples still train the Presence head, but they are not used for embedding pairs.

## Validate Checkpoint

Validate on the main ROI validation set:

```bash
uv run subfast-net validate-roi \
  outputs/roi_presence_embedding_effect_check/best.pt \
  --root data/roi_validation_samples \
  --resize-roi 256x64 \
  --batch-size 64 \
  --device auto
```

Validate on the second ROI eval set:

```bash
uv run subfast-net validate-roi \
  outputs/roi_presence_embedding_effect_check/best.pt \
  --root data/roi_samples3 \
  --resize-roi 256x64 \
  --batch-size 64 \
  --device auto
```

## Resume Training

```bash
uv run subfast-net train-roi \
  --train-root data/roi_samples1 \
  --train-root data/roi_samples2 \
  --train-root data/roi_samples4 \
  --train-root data/roi_samples5 \
  --train-root data/roi_samples6 \
  --val-root data/roi_validation_samples \
  --output-dir outputs/roi_presence_embedding_full \
  --resume outputs/roi_presence_embedding_full \
  --resize-roi 256x64 \
  --batch-size 32 \
  --epochs 20 \
  --lr 0.0003 \
  --negative-ratio 0.35 \
  --log-interval 50 \
  --device auto
```

`--resume` accepts:

- A checkpoint file such as `outputs/roi_presence_embedding_full/best.pt`.
- An epoch checkpoint directory such as `outputs/roi_presence_embedding_full/epoch_outputs/epoch_0010`.
- An output directory such as `outputs/roi_presence_embedding_full`.

## Output Files

Each ROI training run writes:

| Path | Meaning |
|---|---|
| `best.pt` | Best ROI checkpoint by validation Presence F1 |
| `metrics.jsonl` | ROI train-step and validation metrics |
| `summary.json` | Last ROI validation summary |
| `epoch_outputs/epoch_000N/model.pt` | Per-epoch ROI checkpoint |

ROI checkpoints include `model_type = "roi_presence_embedding"` and are not compatible with bbox detector checkpoints.

## Common Parameters

| Parameter | Meaning |
|---|---|
| `--train-root` | ROI training dataset root. Repeat it for multiple roots. |
| `--val-root` | ROI validation dataset root. |
| `--output-dir` | Run output directory. |
| `--resize-roi` | Explicit deterministic resize as `WIDTHxHEIGHT`; required when ROI roots have different native sizes. |
| `--batch-size` | PyTorch training batch size. |
| `--epochs` | Final epoch number to run to. For resume, set this higher than the completed epoch. |
| `--lr` | AdamW learning rate. |
| `--max-train-samples` | Cap the training sample count. |
| `--max-val-samples` | Cap the validation sample count with segment-aware ROI sampling. |
| `--positive-ratio` | Target subtitle-present ratio in a capped training set. |
| `--negative-ratio` | Target no-subtitle ratio in a capped training set. |
| `--val-positive-ratio` | Target subtitle-present ratio in a capped validation set. Positive validation samples are selected by subtitle segment so same-subtitle pairs remain available when possible. |
| `--val-negative-ratio` | Target no-subtitle ratio in a capped validation set. |
| `--embedding-loss-weight` | Weight applied to embedding loss in `total_loss`. |
| `--embedding-pair-frame-window` | Maximum frame-index distance for local embedding pairs. Samples missing video or frame metadata cannot form local pairs. |
| `--embedding-ocr-negative-enabled` / `--no-embedding-ocr-negative-enabled` | Enable or disable conservative OCR strong-difference negative pairs. |
| `--embedding-ocr-negative-max-similarity` | Maximum normalized OCR text similarity allowed for OCR negative pairs. Lower is more conservative. |
| `--embedding-temperature` | Temperature used by pairwise embedding loss. |
| `--embedding-similarity-threshold` | Similarity threshold used by embedding pair accuracy. |
| `--width` | Small backbone width. |
| `--embedding-dim` | Embedding dimension. Default is `128`. |
| `--log-interval` | Step interval for writing train-step rows to `metrics.jsonl`. |
| `--device` | `auto`, `cpu`, `mps`, or `cuda` depending on local availability. |
