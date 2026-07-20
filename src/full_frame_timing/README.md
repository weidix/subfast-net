# Full-frame subtitle timing

`full-frame-timing` is independent from `h264-timing`. It decodes the complete
display frame and produces subtitle time intervals; it does not use a fixed
bottom crop, H.264 slice payloads, or OCR.

The source frame is aspect-preservingly resized and letterboxed to 256x144.
Compact edge statistics are computed over an 8x8 tiling and pooled without tile
coordinates, so the model input covers the whole display frame without fixing a
subtitle row. Pixel arrays are not stored.

Build full-frame features from the existing timing manifest and its videos:

```bash
uv run full-frame-timing prepare \
  data/h264_timing/streaming-training/manifest.jsonl \
  data/full_frame_timing/training
```

Train on the current same-source temporal split:

```bash
uv run full-frame-timing train \
  data/full_frame_timing/training/manifest.jsonl \
  outputs/full_frame_timing/final \
  --epochs 24 \
  --batch-size 64 \
  --validation-mode diagnostic_temporal \
  --device mps
```

Run inference on any FFmpeg-supported video and write interval CSV:

```bash
uv run full-frame-timing infer \
  input.mp4 \
  outputs/full_frame_timing/final/best.pt \
  output.csv
```

The current dataset uses one source video. Its validation result is diagnostic,
not evidence of cross-video or cross-style generalization.

## Verified local result

The full 24-epoch command above was run on the existing 35 train pairs and five
validation pairs. The resulting model has 189 numeric features, 189,961
parameters, no compressed-byte branch, and no fixed subtitle ROI. Epoch 24 was
selected as `best.pt`.

All 10 validation records contain 113 target intervals. The paired signal and
clean-control diagnostic result is:

| Metric | Result |
| --- | ---: |
| IoU50 precision / recall / F1 | 0.648000 / 0.716814 / 0.680672 |
| Matched / missed / false intervals | 81 / 32 / 44 |
| Matched mean IoU | 0.887641 |
| One-frame precision / recall / F1 | 0.312000 / 0.345133 / 0.327731 |
| 100 ms boundary F1 | 0.445378 |
| 250 ms boundary F1 | 0.495798 |
| Signal-only IoU50 F1 | 0.720000 |
| Clean-control false intervals | 13 |

An end-to-end `infer` run on `miab-462-val-01-signal.mp4` produced IoU50
precision / recall / F1 of 0.800000 / 0.923077 / 0.857143. These figures remain
same-source temporal diagnostics because all clips originate from MIAB-462.
