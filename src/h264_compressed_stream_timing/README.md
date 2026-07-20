# Compressed-only H.264 streaming timing

`h264_compressed_stream_timing` is the causal H.264 timing family whose
deployment input contains compressed-domain statistics and packet bytes only.
It never decodes ROI pixels at inference.

```bash
uv run h264-compressed-stream-timing prepare \
  data/h264_timing/streaming-training/manifest.jsonl \
  data/h264_timing/compressed-streaming-training

uv run h264-compressed-stream-timing train \
  data/h264_timing/compressed-streaming-training/manifest.jsonl \
  outputs/h264_timing/compressed-streaming-final

uv run h264-compressed-stream-timing infer \
  outputs/h264_timing/compressed-streaming-final/best.pt input.mp4
```

The preparation, quality gate, and optional visual-teacher details are in the
[H.264 timing guide](../h264_timing/README.md#compressed-only-streaming).
