# Visual H.264 streaming timing

`h264_stream_timing` is the causal, low-latency H.264 subtitle-timing family.
It consumes the shared timing cache contract and H.264 packet/visual feature
adapter, while its causal model, decoder, and trainer live in the reusable
`subtitle_timing_stream` package.

```bash
uv run h264-stream-timing train \
  data/h264_timing/streaming-training/manifest.jsonl \
  outputs/h264_timing/streaming-final \
  --epochs 15 --batch-size 16 --validation-mode diagnostic_temporal

uv run h264-stream-timing infer \
  outputs/h264_timing/streaming-final/best.pt input.mp4
```

The checkpoint input contract and stateful Python API are described in the
[direct H.264 guide](../h264_timing/README.md#independent-streaming-mode).
