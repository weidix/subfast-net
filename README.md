# subfast-net

PyTorch subtitle-geometry and subtitle-timing training projects. Inference
produces subtitle regions, presence scores, matching scores, or time intervals;
it does not perform OCR or text decoding.

## Setup

Python 3.13 or newer and [uv](https://docs.astral.sh/uv/) are required.

```bash
uv sync
```

Datasets belong under `data/` and checkpoints/exports under `outputs/`. Both
directories are local-only.

## Independent commands

| Training family | Command |
|---|---|
| Full-frame subtitle-region detector | `subfast-detector` |
| ROI subtitle presence | `subfast-roi-presence` |
| ROI presence and embedding | `subfast-roi-embedding` |
| ROI same-subtitle matcher | `subfast-roi-matcher` |
| Direct H.264 interval proposals | `h264-timing` |
| Visual H.264 causal timing | `h264-stream-timing train\|infer` |
| Compressed-only H.264 causal timing | `h264-compressed-stream-timing prepare\|train\|infer` |
| Decoded full-frame causal timing | `full-frame-timing` |
| Deployment export | `subfast-export unified\|coreml\|safetensors` |
| Dataset and review utilities | `subfast-tools` |

`subfast-net` remains an aggregate convenience CLI for the four visual model
families and exports. Each training project above has its own entry point.

```bash
uv run subfast-detector --help
uv run subfast-roi-presence --help
uv run subfast-roi-embedding --help
uv run subfast-roi-matcher --help
uv run h264-timing --help
uv run h264-stream-timing --help
uv run h264-compressed-stream-timing --help
uv run full-frame-timing --help
uv run subfast-export --help
```

## Source layout

```text
src/
├── subfast_detector/                 # full-frame region detector
├── subfast_roi_presence/             # ROI presence training family
├── subfast_roi_embedding/            # ROI presence + embedding family
├── subfast_roi_matcher/              # ROI pair-matching family
├── subfast_roi_data/                 # shared ROI samples, pairs, runtime helpers
├── subfast_shared/                   # shared geometry, normalization, layers, runtime
├── subfast_export/                   # reusable unified/Core ML/safetensors exporters
├── h264_timing/                      # direct H.264 proposal detector and feature/data flow
├── h264_stream_timing/               # visual H.264 stream-facing API and CLI
├── h264_compressed_stream_timing/    # compressed-only H.264 stream family
├── subtitle_timing_core/             # cache, manifest, labels, metrics, segment I/O
├── subtitle_timing_stream/           # reusable causal timing model, decoder, trainer
├── full_frame_timing/                # decoded full-frame timing family
└── tools/                            # dataset and review utilities
```

`subfast_shared`, `subfast_roi_data`, `subfast_export`,
`subtitle_timing_core`, and `subtitle_timing_stream` are deliberately reusable
packages. `full_frame_timing` depends only on the timing core/stream layers, not
on H.264 feature extraction.

Training contracts are documented in [Detector](docs/detector.md),
[ROI Presence](docs/roi-presence.md), [ROI Embedding](docs/roi-embedding.md),
and [ROI Matcher](docs/roi-matcher.md). The direct H.264, visual stream,
compressed stream, full-frame, and data-tool guides live beside their packages.
