# Frame Presence V5

`subfast-frame-presence` trains Frame Presence V5 (`architecture_version = 5`) to decide whether an RGB full frame, subtitle ROI, or random crop contains subtitles. It does not perform OCR or text decoding. V5 is trained from random initialization and cannot load V1-V4 weights.

## Preprocessing Contract

Every input domain uses the same rule. Multiply the source width and height by `resize_scale` (default `0.25`), round each result to the nearest multiple of 16 using half-up ties, and stretch-resize with bilinear interpolation. Thus `1920x1080` becomes `480x272`. The RGB tensor is float32 in `[0,1]`; padding is forbidden.

Training protects refined small-subtitle boxes. Boxes are first mapped to the standard scaled output. When any box has a short edge below `min_subtitle_short_edge` (default `8.0 px`), the sample uses the minimum larger scale that satisfies every box, capped at `1.0`. Valid boxes and hard samples are retained; only images explicitly marked `drop_image` by manual data review are excluded as invalid data. Negative samples draw from the positive samples' actual resize-scale distribution. Validation always uses the configured standard scale and never uses annotation-driven protection.

Each small-subtitle event is printed with the sample ID, image path, source size, standard output size, mapped minimum short edge, protection scale, and protected output size. The same records are saved in `small_subtitle_warnings.jsonl`; aggregate counts are saved in `small_subtitle_summary.json` and checkpoint metadata.

## Model And Batching

V5 keeps the compact V4 topology and stride-8 region output. Scheme A is the default: all BatchNorm layers are removed and no replacement normalization is used. Convolutions use Kaiming initialization, inputs are scaled to `[0,1]`, and gradients are clipped.

A logical macro batch mixes full-frame, ROI, and random-crop domains plus positive and negative samples. Positive crops retain every refined subtitle box. The macro batch is split by exact `HxW` into execution micro batches without padding. Presence class balancing and region losses are reduced across the complete macro batch, followed by one backward pass and one optimizer step. Scheme B is available through `--normalization group_norm`; GroupNorm chooses a channel-compatible group count and supports micro batches of one. A Scheme B run must use a new output directory and fresh random initialization from Epoch 1.

## Train

```bash
uv run subfast-frame-presence \
  --output-dir outputs/frame_presence_v5 \
  --resize-scale 0.25 \
  --min-subtitle-short-edge 8.0 \
  --epochs 10 \
  --device auto
```

`--train-root` and `--val-root` may be repeated. `--random-crop-views`, `--random-crop-min-scale`, and `--random-crop-max-scale` configure crop augmentation. `--max-val-samples` is diagnostic only and cannot pass acceptance. `--resume` accepts only an interrupted V5 run with identical preprocessing, model, dataset, and sampling settings; there is no old-checkpoint initialization option.

Each Epoch runs the complete independent held-out validation set at a fixed decision threshold of `0.5`. Training succeeds only when one Epoch from 1 through 10 simultaneously has Recall `1`, F1 `1`, FP `0`, FN `0`, and Gap at least `0.8`, where Gap is the minimum positive sigmoid score minus the maximum negative sigmoid score.

## Outputs

```text
outputs/frame_presence_v5/
├── run_config.json
├── source_snapshot.json
├── source.patch
├── data_manifest_train.jsonl
├── data_manifest_validation.jsonl
├── small_subtitle_warnings.jsonl
├── small_subtitle_summary.json
├── metrics.jsonl
├── benchmark.json
├── epoch_outputs/epoch_0001/
│   ├── checkpoint.pt
│   ├── metrics.json
│   └── validation_scores.jsonl
├── best.pt
├── best_inference.pt
├── last.pt
├── export/
│   ├── config.json
│   └── model.safetensors
└── summary.json
```

All checkpoints, `run_config.json`, `summary.json`, and exported safetensors `config.json` identify Frame Presence V5, architecture version 5, the resize rule, interpolation, alignment, value range, normalization scheme, and no-padding macro/micro-batch contract.

## Verified V5 Run

The canonical fresh Scheme A run in `outputs/frame_presence_v5` passed on Epoch 1 using all 2,217 independent held-out samples at threshold `0.5`.

| Precision | Recall | F1 | FP | FN | Minimum positive | Maximum negative | Gap |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.0 | 1.0 | 1.0 | 0 | 0 | 0.998440 | 0.041320 | 0.957120 |

The run produced one small-subtitle warning: `video0001_f00002250#crop0` mapped to a `7.8904 px` short edge at the standard `128x48` output and was protected at scale `0.25570776`, producing `128x64`. The protected box satisfies the configured `8.0 px` minimum. Scheme B was not used because Scheme A was stable and passed the complete held-out acceptance contract.

```bash
uv run subfast-net benchmark frame-presence outputs/frame_presence_v5/best_inference.pt --device mps
uv run subfast-export safetensors outputs/frame_presence_v5/best_inference.pt outputs/frame_presence_v5/export
```
