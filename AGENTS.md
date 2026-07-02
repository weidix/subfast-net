# AGENTS.md

## Project Goal

This project trains a small PyTorch model for detecting subtitle regions in video-frame images.

The model is a detector only. It must output subtitle geometry such as bounding boxes and/or masks. Do not add OCR, text recognition, subtitle parsing, translation, language modeling, or text decoding unless explicitly requested.

## Data Source

Use this project's own configured data paths. Do not assume data, configs, or behavior should come from another repository unless explicitly requested.

## Training Mode Reference

Training mode means which operators, targets, losses, thresholds, augmentations, validation logic, and postprocessing rules are used for training and evaluation.

Treat this Python project as the source of truth. Do not migrate, mirror, or preserve behavior from another repository unless the user explicitly asks for that comparison or port.

## Dependency Management

Use `uv` for dependency management.

Keep dependencies narrow and task-driven. Do not add speculative fallback libraries or unused optional stacks.

Current expected dependency roles:

- `torch`: model, tensor operations, training, inference
- `torchvision`: image transforms and vision utilities when useful
- `numpy`: array and numeric utilities
- `pillow`: image loading and simple image handling
- `opencv-python`: image visualization and drawing detection boxes for verification
- `tqdm`: compact progress reporting
- `pydantic`: typed config validation

Do not use `opencv-python` as a video extraction dependency unless that task is explicitly requested.

## Implementation Rules

- Keep the model small and subtitle-specific.
- Keep module boundaries clear: dataset loading, preprocessing, target generation, model, loss, training, validation, postprocessing, and visualization should not be mixed into one large file.
- Preserve coordinate-space definitions explicitly. Any resize, crop, padding, or alignment step must make its output coordinate space clear.
- The user dislikes large batches of incidental tests. Do not add tests by default, and do not add tests for trivial, mechanical, output-only, or low-risk changes.
- Add tests only when the logic is complex, error-prone, coordinate-sensitive, or needed to validate training/evaluation behavior. Keep any test scope minimal and directly tied to the change.
- Before claiming training quality, verify with held-out validation metrics, not just loss curves or self-overfit results.

## Development Style

- Keep changes narrow and directly tied to the current request.
- Do not run tests, full verification, or confirmation checks at the end of every task by default. Keep the execution flow short; run verification only when the change risk or user request warrants it.
- Prefer deleting irrelevant copied tests over keeping them around as noise.
- Prefer repo-backed facts and fresh verification over inference.
- Do not introduce unrelated architecture, tooling, or generated artifacts.
- Do not leave unused files, dependencies, or compatibility shims in the repository.
