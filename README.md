# subfast-net

A small PyTorch detector for locating subtitle regions in video frames. The project includes full-frame detection, ROI presence and matching models, held-out validation, dataset review tools, and runtime export commands.

## Setup

Python 3.13 or newer and [uv](https://docs.astral.sh/uv/) are required.

```bash
uv sync
```

Training data and generated artifacts are local-only. Put datasets under `data/`; commands write checkpoints and metrics under `outputs/`. Both directories are excluded from Git.

## Commands

```bash
# Show the default detector training options
uv run subfast-net --help

# Train the full-frame subtitle detector
uv run subfast-net train --help

# Train or validate ROI models
uv run subfast-net train-roi --help
uv run subfast-net validate-roi --help
uv run subfast-net train-presence --help
uv run subfast-net train-roi-pair --help
uv run subfast-net validate-roi-pair --help

# Export trained checkpoints
uv run subfast-net export-unified --help
uv run subfast-net export-coreml --help
uv run subfast-net export-safetensors --help
```

Exact training examples and output contracts are documented in [docs/training_commands.md](docs/training_commands.md), [docs/roi_presence_training.md](docs/roi_presence_training.md), and [docs/roi_pair_matcher_training.md](docs/roi_pair_matcher_training.md).

## Repository layout

- `src/`: models, datasets, losses, training, validation, postprocessing, and export code
- `tools/`: dataset preparation and browser-based review tools
- `tests/`: focused unit and smoke tests
- `docs/`: training, optimization, and deployment notes
