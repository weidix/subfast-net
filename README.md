# subfast-net

PyTorch subtitle-geometry training projects. Inference produces subtitle
regions, presence scores, or matching scores; it does not perform OCR or text
decoding.

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
| Fast full-frame presence + rough region | `subfast-frame-presence` |
| ROI subtitle presence | `subfast-roi-presence` |
| ROI presence and embedding | `subfast-roi-embedding` |
| ROI same-subtitle matcher | `subfast-roi-matcher` |
| Deployment export | `subfast-export unified\|coreml\|safetensors` |
| Dataset and review utilities | `subfast-tools` |

`subfast-net` remains an aggregate convenience CLI for the four visual model
families and exports. Each training project above has its own entry point.

```bash
uv run subfast-detector --help
uv run subfast-frame-presence --help
uv run subfast-roi-presence --help
uv run subfast-roi-embedding --help
uv run subfast-roi-matcher --help
uv run subfast-export --help
```

## Source layout

```text
src/
├── subfast_detector/                 # full-frame region detector
├── subfast_frame_presence/           # fast presence + enclosing contour
├── subfast_roi_presence/             # ROI presence training family
├── subfast_roi_embedding/            # ROI presence + embedding family
├── subfast_roi_matcher/              # ROI pair-matching family
├── subfast_roi_data/                 # shared ROI samples, pairs, runtime helpers
├── subfast_shared/                   # shared geometry, normalization, layers, runtime
├── subfast_export/                   # reusable unified/Core ML/safetensors exporters
└── tools/                            # dataset and review utilities
```

`subfast_shared`, `subfast_roi_data`, and `subfast_export` are deliberately
reusable packages.

Training contracts are documented in [Detector](docs/detector.md),
[Frame Presence](docs/frame-presence.md),
[ROI Presence](docs/roi-presence.md), [ROI Embedding](docs/roi-embedding.md),
and [ROI Matcher](docs/roi-matcher.md). The data-tool guide lives beside its
package.
