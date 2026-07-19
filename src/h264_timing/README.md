# Direct subtitle segments from H.264 timing features

This project trains a small temporal detector for complete burned-in subtitle
segments. Every compressed-frame anchor predicts a scored `(start, end)` pair.
Auxiliary start/end event heads may refine and confirm the two boundaries of an
already-regressed segment, but they never create segments by grouping frame
presence predictions.

The feature cache combines H.264 packet/slice statistics with compact visual
statistics from a decoded bottom-20% ROI. Pixel arrays are not stored. This
remains a same-source diagnostic route, not evidence of cross-video or
cross-encoder generalization.

## What changed from P1 v1

P1 v1 could complete a training command but could not produce useful intervals:

- the last 20% of a VCL payload was incorrectly treated as a spatial proxy for
  the bottom 20% of the picture;
- fixed boundary weights and fixed postprocessing thresholds collapsed the
  boundary and interval metrics to zero;
- train and validation cue-duration/active-rate distributions were badly
  mismatched;
- a tied validation score left `best.pt` at the first epoch.

P1 v2 replaces that route with a verified horizontal-slice contract, paired
signal/control samples, bounded randomized cues, dynamic target weights,
train-split threshold calibration, and deterministic checkpoint selection.

## Exact ROI contract

The default `exact_bottom_slices` mode is exact only for streams whose slice
layout is verified by this project:

- progressive H.264 in an AVCC MP4/MOV-family container;
- encoded with libx264 using five fixed horizontal slices per frame;
- a slice starts exactly at the bottom-20% macroblock boundary on every frame;
- VCL slices are unique and in increasing raster order.

For the current 1920x1080 source, the display ROI starts at `y=864`, or
`first_mb_in_slice=6480`. Every generated frame is required to have slice starts
`[0, 1680, 3240, 4920, 6480]`. The selected slice also contains the coded padding
between display height 1080 and coded height 1088.

This is slice-header parsing, not CABAC/CAVLC macroblock parsing. The generic
extractor does not yet parse arbitrary SPS crop offsets, MBAFF, or FMO, so an
unverified external H.264 stream must not be described as an exact ROI source.
Normal single-slice H.264 is rejected by default.

The old non-spatial baseline remains available only by explicit opt-in:

```bash
uv run h264-timing extract \
  /path/to/unverified.mp4 data/h264_timing/features/proxy \
  --spatial-mode payload_tail_proxy
```

## Dataset preparation

`prepare` is the canonical P1 workflow. Each source segment produces two videos
with the same source timeline and encoder settings:

- `subtitle_signal`: deterministic randomized burned-in cues;
- `clean_control`: the same source frames with no burned-in cue and empty labels.

Randomized cue durations are 0.5-5.0 seconds and gaps are 0.5-4.0 seconds.
Normalized subtitle payload hashes are assigned to fixed train/val/test
partitions, so the same text cannot occur across splits. The audit also records a
SHA-256 of the source video; strict resume rejects source, SRT, font, schedule,
slice-contract, or feature-setting changes. Generated videos use full-file SHA-256
instead of an edge sample, and every feature-cache array has its own SHA-256, so
resume and training reject changed videos, labels, or cached features.

The sample plan is CSV:

```csv
clip_id,split,start_seconds,end_seconds
source-train-01,train,541.520,663.210
source-val-01,val,7632.365,7755.840
```

Clips may not overlap. Different splits require the configured source-timeline
guard, 10 seconds by default.

Build all videos, labels, feature caches, the manifest, and the dataset audit:

```bash
uv run h264-timing prepare \
  data/h264_timing/source/MIAB-462.mp4 \
  data/h264_timing/source/MIAB-462.zh-CN.srt \
  data/h264_timing/window-training/window-clip-plan.csv \
  data/h264_timing/window-training \
  --source-group miab-462-window-training \
  --seed 2026
```

With no output present, this creates a new dataset. With complete output present,
the command resumes only after validating all recorded contracts. Use
`--overwrite` when intentionally rebuilding generated output.

The resulting manifest uses explicit pair metadata:

```json
{"video_id":"source-train-01-signal","source_group":"source-window-training","features":"features/source-train-01-signal","labels":"composite/labels/source-train-01-signal.csv","split":"train","source_time_offset_seconds":541.541,"synthesis_audit":"composite/videos/source-train-01-signal.audit.json","pair_id":"source-train-01","signal_validation_role":"subtitle_signal"}
{"video_id":"source-train-01-clean","source_group":"source-window-training","features":"features/source-train-01-clean","labels":"composite/labels/source-train-01-clean.csv","split":"train","source_time_offset_seconds":541.541,"synthesis_audit":"composite/videos/source-train-01-clean.audit.json","pair_id":"source-train-01","signal_validation_role":"clean_control"}
```

`dataset-audit.json` is the source of truth for pair counts, packet counts, cue
duration/gap distributions, active ratios, content-partition isolation, source
fingerprints, and the required validation mode.

## Features and model

With the default settings, each frame has 326 numeric features:

- 128 compressed-domain packet, slice, byte-count, histogram, entropy, timing,
  and temporal-delta features;
- 198 bottom-ROI gray/edge statistics and their signed/absolute temporal deltas.

The cache also stores 256 uniformly sampled byte tokens from the exact ROI slice.
Training is numeric-only by default. `--use-byte-branch` enables the optional
embedding/CNN byte branch.

The model uses a dilated TCN and a bidirectional GRU. At every anchor it emits a
segment score, paired start/end offsets, and auxiliary start/end event logits.
Training jointly optimizes focal proposal classification, paired Smooth L1
boundary regression, temporal IoU, and event heatmaps.

Postprocessing operates only on complete regressed segments: proposal-peak
selection, boundary-event confirmation/refinement, and temporal NMS. It has no
presence runs, hysteresis, gap merging, or independent start/end pairing. Score,
NMS, end refinement, event strength, and confirmation settings are calibrated on
train predictions; the start radius and event floor are fixed experiment
settings. Validation labels are not used in this search. The full resulting
configuration is stored in `best.pt` and `last.pt`.

## Training and inference

The current MIAB-462 dataset is one source and one encoding configuration, so it
must use diagnostic temporal validation:

```bash
uv run h264-timing train \
  data/h264_timing/window-training/manifest.jsonl \
  outputs/h264_timing/window-final \
  --epochs 15 \
  --batch-size 16 \
  --width 64 \
  --validation-mode diagnostic_temporal \
  --temporal-guard 10 \
  --device mps
```

For genuinely different source videos, assign stable source groups and use the
default `held_out` validation mode. Different transcodes of the same source must
retain the same source group.

Inference loads every calibrated postprocessing value from the checkpoint unless
that value is explicitly overridden on the command line:

```bash
uv run h264-timing infer \
  data/h264_timing/window-training/features/miab-462-val-01-signal \
  outputs/h264_timing/window-final/best.pt \
  outputs/h264_timing/window-final/miab-462-val-01-signal.csv \
  --labels data/h264_timing/window-training/composite/labels/miab-462-val-01-signal.csv
```

The command writes only interval CSV. It reports interval precision/recall/F1 at
temporal IoU 0.5, matched mean IoU, start/end MAE/p95/max, paired-boundary
precision/recall/F1 at one frame, 100 ms, and 250 ms, and false intervals per
minute when labels are supplied.

## Independent streaming mode

The direct-segment v5 model and its `train` / `infer` commands remain unchanged.
An additional streaming model is available for callers that cannot build a
feature-cache directory or know the total sample count before inference.

The streaming model uses causal dilated convolutions and a unidirectional GRU.
It emits subtitle-presence, start-event, and end-event probabilities for each
arriving sample. A stateful decoder opens and closes complete segments, emits a
segment as soon as its end is confirmed, and retains only fixed convolution/GRU
state plus constant-size decoder state. Input samples are discarded after they
are consumed.

Streaming checkpoints use a separate format and must be trained independently:

```bash
uv run h264-timing train-stream \
  data/h264_timing/streaming-training/manifest.jsonl \
  outputs/h264_timing/streaming-final \
  --epochs 15 \
  --batch-size 16 \
  --validation-mode diagnostic_temporal \
  --device mps
```

The Python inference entry accepts one sample or a chunk and owns all temporal
state until `close()`:

```python
from h264_timing.streaming import (
    StreamSample,
    StreamingSegmentDetector,
)

detector = StreamingSegmentDetector.from_checkpoint("outputs/stream/best.pt")
try:
    for sample in model_ready_samples:
        for segment in detector.push(
            StreamSample(
                timestamp_seconds=sample.timestamp_seconds,
                duration_seconds=sample.duration_seconds,
                features=sample.features,
                tokens=sample.tokens,
            )
        ):
            consume(segment)
finally:
    for segment in detector.close():
        consume(segment)
```

Samples must use strictly increasing presentation timestamps and the numeric
feature order stored in the checkpoint. Byte tokens are optional unless the
checkpoint enabled the byte branch. `push_many()` processes a supplied chunk in
one model call without changing results at chunk boundaries.

The command-line streaming entry accepts an H.264 video directly and emits
finalized segments as JSON Lines. MP4/MOV-family files are read directly;
other timestamped H.264 containers such as MKV and TS are losslessly remuxed
to a temporary MP4 before feature extraction:

```bash
uv run h264-timing stream-infer \
  outputs/h264_timing/streaming-final/best.pt input.mp4
```

Omit the video argument to read model-ready JSON Lines from stdin instead:

```bash
uv run h264-timing stream-infer \
  outputs/h264_timing/streaming-final/best.pt < live-samples.jsonl
```

Each sample object contains `timestamp_seconds`, `duration_seconds`, `features`,
and optional `tokens`. Send `{"type":"close"}` or end stdin to flush the active
tail and close the inference session. Video feature settings, visual settings,
and inference chunk size are loaded from the streaming checkpoint.

### Compressed-only streaming

The dedicated compressed stream family never uses ROI pixels at inference. Its
default expanded representation contains 498 numeric H.264 payload statistics
and 512 sampled ROI slice-payload bytes per frame. Build that cache family
without changing the existing streaming manifest:

```bash
uv run h264-timing prepare-compressed-stream \
  data/h264_timing/streaming-training/manifest.jsonl \
  data/h264_timing/compressed-streaming-training
```

Training may use the existing visual model as a teacher. The visual manifest is
read only for training supervision; it is not part of the student checkpoint's
input contract:

```bash
uv run h264-timing train-compressed-stream \
  data/h264_timing/compressed-streaming-training/manifest.jsonl \
  outputs/h264_timing/compressed-streaming-final-498 \
  --visual-teacher-checkpoint outputs/h264_timing/streaming-final/best.pt \
  --visual-teacher-manifest data/h264_timing/streaming-training/manifest.jsonl \
  --validation-mode diagnostic_temporal
```

`summary.json` contains a strict `quality_gate`. It passes only when validation
recall, interval F1, and one-frame boundary F1 are all 1.0 and the maximum
boundary drift is at most one frame. A checkpoint is still retained when the
gate fails so that an unsuccessful compressed-only experiment is inspectable;
it must not be reported as satisfying those deployment metrics.

Deployment uses the dedicated command. It performs container demux, packet
reads, NAL splitting, and slice-header-prefix parsing only; FFmpeg is used only
for lossless container remux when necessary, never for pixel decode:

```bash
uv run h264-timing compressed-stream-infer \
  outputs/h264_timing/compressed-streaming-final-498/best.pt input.mp4
```

### Streaming visual-only training comparison

The existing epoch-24 `outputs/h264_timing/streaming-final/best.pt` checkpoint
is the combined-input baseline. A separate visual-only model was trained from
scratch for 24 epochs with the same seed, data split, model width, temporal
layers, windowing, loss weights, and decoder calibration procedure. Its model
input structurally contains only the 198 decoded visual features; the 128
compressed numeric features and 256-byte branch are absent from both training
and inference.

Both best checkpoints were evaluated on all 10 validation records (34,636
frames and 113 labeled cues):

| Trained model | IoU50 precision | IoU50 recall | IoU50 F1 | One-frame F1 | Matched / target | False intervals |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Combined baseline | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 113 / 113 | 0 |
| Visual only | 1.000000 | 0.982301 | 0.991071 | 0.937500 | 111 / 113 | 0 |

The retrained visual-only model preserves almost all interval-detection quality,
so the current same-source task is predominantly solved by visual features.
Compressed inputs still provide a measurable benefit: two additional cues and
perfect one-frame boundary matching. The baseline was produced by staged
training while the visual-only run was trained from scratch, so this comparison
does not isolate feature domain from training schedule. Neither result proves
source-disjoint generalization.

## Verified local result

The command above was verified with the MIAB-462 P1 v2 dataset and checkpoint
format version 5. The dataset contains 12 train pairs and 5 validation pairs,
122,728 frames, 301 train cues, and 113 validation cues. The best checkpoint is
epoch 15. Its postprocessing configuration combines fixed experiment settings
with values selected from the train split:

```json
{
  "score_threshold": 0.075,
  "nms_iou_threshold": 0.95,
  "boundary_event_threshold": 0.2,
  "start_boundary_refinement_seconds": 0.6,
  "end_boundary_refinement_seconds": 1.2,
  "end_event_relative_threshold": 0.8,
  "require_boundary_events": true
}
```

The diagnostic temporal-validation result is:

| Metric | Result |
| --- | ---: |
| Overall IoU50 precision / recall / F1 | 0.353125 / **1.000000** / 0.521940 |
| Paired-boundary precision / recall / F1 at 1 frame | 0.353125 / **1.000000** / 0.521940 |
| Paired-boundary recall at 100 ms / 250 ms | **1.000000 / 1.000000** |
| Matched intervals | **113 / 113** |
| Matched mean IoU | 0.999864 |
| Start / end MAE | 0.000297 s / 0.000297 s |
| Start / end p95 | 0.0000033 s / 0.0000033 s |
| Start / end maximum error | 0.033366 s / 0.033367 s |
| Signal-only IoU50 precision / recall | 0.579487 / **1.000000** |
| Clean-control false intervals | 12.9793 per minute |

The model emitted 320 complete segments for 113 targets: all 113 targets matched,
with 207 false intervals. Recall and boundary accuracy meet this experiment's
target, while false positives remain the main unresolved quality limit.

## Interpretation boundary

The local 12-pair train / 5-pair validation plan is separated on the source
timeline, but it uses one source and one encoding configuration. The validation
set was also used during experiment development, so the numbers above are
diagnostic rather than an untouched final benchmark.

It does not establish performance on another movie, resolution, subtitle style,
codec profile, rate-control mode, or encoder. Cross-source claims require a
source-disjoint manifest and independent test inference.
