# ROI Presence + Embedding Training Commands

本文记录独立的 ROI Presence + Embedding 训练结构。presence-only 训练方式见 [ROI Presence 模型训练](roi-presence.md)。

This document lists the commands for the ROI subtitle Presence + Embedding model.

This is separate from bbox detector training. Use the canonical `train embedding` command to train the combined Presence + Embedding model.

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
uv run subfast-net train embedding \
  --train-root data/roi_samples1 \
  --train-root data/roi_samples2 \
  --train-root data/roi_samples4 \
  --train-root data/roi_samples5 \
  --train-root data/roi_samples6 \
  --val-root data/roi_validation_samples \
  --output-dir outputs/roi_presence_embedding_full \
  --resize-roi 256x64 \
  --presence-batch-size 32 \
  --embedding-batch-size 32 \
  --presence-epochs 3 \
  --embedding-epochs 3 \
  --joint-epochs 4 \
  --lr 0.0003 \
  --joint-lr 0.00003 \
  --presence-negative-ratio 0.35 \
  --embedding-negative-ratio 0.5 \
  --joint-presence-negative-ratio 0.35 \
  --joint-embedding-batch-negative-ratio 0.5 \
  --log-interval 50 \
  --device auto
```

## Presence-only Training

```bash
uv run subfast-net train embedding \
  --train-root data/roi_samples1 \
  --train-root data/roi_samples2 \
  --train-root data/roi_samples4 \
  --train-root data/roi_samples5 \
  --train-root data/roi_samples6 \
  --val-root data/roi_validation_samples \
  --output-dir outputs/roi_presence_only \
  --resize-roi 256x64 \
  --presence-batch-size 32 \
  --embedding-batch-size 32 \
  --presence-epochs 3 \
  --embedding-epochs 0 \
  --joint-epochs 0 \
  --lr 0.0003 \
  --presence-negative-ratio 0.35 \
  --log-interval 50 \
  --device auto
```

## Effect Check Run

This is the shorter run used to verify the current training path:

```bash
uv run subfast-net train embedding \
  --train-root data/roi_samples1 \
  --train-root data/roi_samples2 \
  --train-root data/roi_samples4 \
  --train-root data/roi_samples5 \
  --train-root data/roi_samples6 \
  --val-root data/roi_validation_samples \
  --output-dir outputs/roi_presence_embedding_effect_check \
  --resize-roi 256x64 \
  --presence-batch-size 32 \
  --embedding-batch-size 32 \
  --presence-epochs 2 \
  --embedding-epochs 2 \
  --joint-epochs 1 \
  --lr 0.0003 \
  --joint-lr 0.00003 \
  --max-train-samples 4000 \
  --max-val-samples 600 \
  --presence-negative-ratio 0.35 \
  --embedding-negative-ratio 0.5 \
  --joint-presence-negative-ratio 0.35 \
  --joint-embedding-batch-negative-ratio 0.5 \
  --val-negative-ratio 0.35 \
  --width 16 \
  --log-interval 50 \
  --device auto
```

Embedding metrics now use trusted pairs only:

- Local positive pairs: same root, both subtitle-present, same `segment_id`.
- Local negative pairs: same root, same video, both subtitle-present, adjacent different `segment_id` values in the dataset frame order.
- OCR negative pairs: same root, both subtitle-present, different `segment_id`, usable OCR text on both sides, normalized OCR similarity at or below `--embedding-ocr-negative-max-similarity`, capped by `--embedding-ocr-negative-ratio`.

No-subtitle samples still train the Presence head, but they are not used for embedding pairs.

Stage 3 checkpoint score is `0.5 * mean(global/normal/short Presence F1) + 0.5 * mean(global/normal/style-hard-negative Embedding accuracy)`.

## Validate Checkpoint

Validate on the main ROI validation set:

```bash
uv run subfast-net validate embedding \
  outputs/roi_presence_embedding_effect_check/best.pt \
  --root data/roi_validation_samples \
  --resize-roi 256x64 \
  --batch-size 64 \
  --device auto
```

Validate on the second ROI eval set:

```bash
uv run subfast-net validate embedding \
  outputs/roi_presence_embedding_effect_check/best.pt \
  --root data/roi_samples3 \
  --resize-roi 256x64 \
  --batch-size 64 \
  --device auto
```

## Resume Training

```bash
uv run subfast-net train embedding \
  --train-root data/roi_samples1 \
  --train-root data/roi_samples2 \
  --train-root data/roi_samples4 \
  --train-root data/roi_samples5 \
  --train-root data/roi_samples6 \
  --val-root data/roi_validation_samples \
  --output-dir outputs/roi_presence_embedding_full \
  --resume outputs/roi_presence_embedding_full \
  --resize-roi 256x64 \
  --presence-batch-size 32 \
  --embedding-batch-size 32 \
  --presence-epochs 3 \
  --embedding-epochs 3 \
  --joint-epochs 14 \
  --lr 0.0003 \
  --joint-lr 0.00003 \
  --presence-negative-ratio 0.35 \
  --embedding-negative-ratio 0.5 \
  --joint-presence-negative-ratio 0.35 \
  --joint-embedding-batch-negative-ratio 0.5 \
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
| `best_presence.pt` | Best first-stage checkpoint by validation Presence score, when Presence stage runs |
| `best_embedding.pt` | Best second-stage checkpoint by validation Embedding score, when Embedding stage runs |
| `best_joint.pt` | Best third-stage checkpoint by the combined Presence + Embedding validation score, when joint stage runs |
| `best.pt` | Best checkpoint from the latest completed stage; after normal three-stage training this is the same model as `best_joint.pt` |
| `metrics.jsonl` | ROI train-step and validation metrics |
| `summary.json` | Best ROI checkpoint summary, with key validation metrics and output file paths |
| `epoch_outputs/epoch_000N/model.pt` | Per-epoch ROI checkpoint |
| `epoch_outputs/epoch_000N/metrics.json` | Per-epoch ROI validation and training metrics |

ROI checkpoints include `model_type = "roi_presence_embedding"` and are not compatible with bbox detector checkpoints.

## Common Parameters

| Parameter | Meaning |
|---|---|
| `--train-root` | ROI training dataset root. Repeat it for multiple roots. |
| `--val-root` | ROI validation dataset root. |
| `--output-dir` | Run output directory. |
| `--resize-roi` | Explicit deterministic resize as `WIDTHxHEIGHT`; required when ROI roots have different native sizes. |
| `--presence-batch-size` | Presence-stage training batch size. |
| `--embedding-batch-size` | Embedding-stage training batch size. |
| `--joint-presence-batch-size` | Joint-stage Presence batch size. Defaults to `--presence-batch-size` when omitted. |
| `--joint-embedding-batch-size` | Joint-stage Embedding batch size. Defaults to `--embedding-batch-size` when omitted. |
| `--presence-epochs` | Stage 1 epochs: train Backbone + Presence Head with Presence loss; freeze Embedding Head. Use `0` to skip. |
| `--embedding-epochs` | Stage 2 epochs: train Embedding Head with Embedding loss; freeze Backbone + Presence Head. Use `0` to skip. |
| `--joint-epochs` | Stage 3 epochs: jointly fine-tune all modules and select by combined validation performance. Use `0` to skip. |
| `--lr` | AdamW learning rate for stages 1 and 2. |
| `--joint-lr` | Smaller AdamW learning rate for stage 3. |
| `--max-train-samples` | Cap the training sample count. |
| `--max-val-samples` | Cap the validation sample count with segment-aware ROI sampling. |
| `--presence-positive-ratio` | Sets the complementary subtitle-present fraction for Presence-stage batches. |
| `--presence-negative-ratio` | Target no-subtitle fraction in Presence-stage batches. The sampler cycles samples so none are discarded. |
| `--val-positive-ratio` | Target subtitle-present ratio in a capped validation set. Positive validation samples are selected by subtitle segment so same-subtitle pairs remain available when possible. |
| `--val-negative-ratio` | Target no-subtitle ratio in a capped validation set. |
| `--embedding-loss-weight` | Weight applied to embedding loss in `total_loss`. |
| `--embedding-negative-ratio` | Target fraction of selected Embedding-stage pairs that come from different segments. All positive pairs are retained and the hardest negative pairs are selected first. |
| `--joint-presence-negative-ratio` | Target no-subtitle fraction in Joint-stage Presence batches. Defaults to `--presence-negative-ratio` when omitted. |
| `--joint-embedding-batch-negative-ratio` | Target fraction of selected Joint-stage Embedding-batch pairs that come from different segments. Defaults to `--embedding-negative-ratio` when omitted. |
| `--embedding-ocr-negative-enabled` / `--no-embedding-ocr-negative-enabled` | Enable or disable conservative OCR strong-difference negative pairs. |
| `--embedding-ocr-negative-max-similarity` | Maximum normalized OCR text similarity allowed for OCR negative pairs. Lower is more conservative. |
| `--embedding-ocr-negative-ratio` | Maximum OCR strong-difference negative share relative to trusted local negatives. Default is `0.3`. |
| `--embedding-temperature` | Temperature used by pairwise embedding loss. |
| `--embedding-similarity-threshold` | Similarity threshold used by embedding pair accuracy. |
| `--width` | Small backbone width. |
| `--embedding-dim` | Embedding dimension. Default is `128`. |
| `--log-interval` | Step interval for writing train-step rows to `metrics.jsonl`. |
| `--device` | `auto`, `cpu`, `mps`, or `cuda` depending on local availability. |
