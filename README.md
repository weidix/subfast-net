# subfast-net

PyTorch subtitle detection models and the data workflows that support them.
The project has three ROI training families, a full-frame detector, and an
H.264 compressed-domain timing detector. Inference produces subtitle geometry,
presence scores, or matching scores; it does not perform OCR or text decoding.

## Setup

Python 3.13 or newer and [uv](https://docs.astral.sh/uv/) are required.

```bash
uv sync
```

Datasets belong under `data/` and checkpoints/exports under `outputs/`.
Those directories are local-only.

## CLI

The repository exposes four independent command packages. `subfast-net`
manages detector and ROI models, `h264-timing` manages compressed-domain
timing detection, `full-frame-timing` manages decoded full-frame timing
detection, and `subfast-tools` manages dataset and review utilities.

```text
subfast-net
|-- train detector
|-- train presence
|-- train embedding
|-- train matcher
|-- validate embedding|matcher
|-- benchmark presence
`-- export unified|coreml|safetensors

subfast-tools
|-- build-samples|synthesize-samples
|-- prepare-roi|extract-craft
|-- labels-to-via|via-to-labels
`-- review-labels|review-roi

full-frame-timing
|-- extract
|-- prepare
|-- train
`-- infer
```

Inspect a command before running it:

```bash
uv run subfast-net --help
uv run subfast-net train detector --help
uv run subfast-net train presence --help
uv run subfast-net train embedding --help
uv run subfast-net train matcher --help
uv run h264-timing --help
uv run full-frame-timing --help
uv run subfast-tools --help
```

The H.264 family is a first-level subproject at `src/h264_timing`. Its `train`
model scores complete subtitle intervals; `train-stream` is an independent
causal model for low-latency decoding. Their checkpoints are not
interchangeable, and both are intentionally retained.

Training contracts and export formats are documented in [Detector](docs/detector.md),
[ROI Presence](docs/roi-presence.md), [ROI Embedding](docs/roi-embedding.md),
and [ROI Matcher](docs/roi-matcher.md).
The H.264 workflow has its own [guide](src/h264_timing/README.md).
The full-frame timing workflow has its own
[guide](src/full_frame_timing/README.md).
Dataset and review utilities are documented in [Tools](src/tools/command.md).

## Layout

```text
subfast-net/
|-- src/subfast_net/          # detector, ROI models, exports, and CLI
|   |-- detector/
|   |-- roi/
|   |   |-- presence/
|   |   |-- embedding/
|   |   `-- matcher/
|   `-- export/
|-- src/h264_timing/          # H.264 timing subproject (two model families)
|-- src/full_frame_timing/    # decoded full-frame timing CLI and model workflow
|-- src/tools/                # independent dataset and review tools package
|-- tests/                    # focused unit and smoke tests
|-- docs/                     # current guides and historical notes
|-- data/                     # local datasets (not tracked)
|-- outputs/                  # local checkpoints and exports (not tracked)
`-- pyproject.toml
```

`src` is a source layout directory, not an import namespace. Import model code
from `subfast_net`, H.264 code from `h264_timing`, and utility code from `tools`.
