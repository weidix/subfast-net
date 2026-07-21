# Frame Presence

`subfast_frame_presence` is a full-frame subtitle gate and conservative rough-region
detector. It does not perform OCR.

## Contract

- Input preprocessing: full-frame luma `256×144`, plus an automatically selected
  RGB detail view `256×32` derived from the same frame.
- Presence output: one logit per frame; the decision threshold is `0.5` after
  sigmoid.
- Region output: interleaved samples from a closed one-cell contour with shape
  `1×72×64`; the activation threshold is `0.5` after sigmoid. Alternating samples
  retain all four enclosing extrema while keeping active heatmap area below the
  labeled subtitle area.
- Region coordinates map directly to the source frame: `x / 64 × source_width`
  and `y / 72 × source_height`.
- The model uses two closely spaced geometric mappings. Expanded evidence is
  accepted only within one heatmap cell of compact evidence, which prevents a
  whole-frame region from satisfying containment by indiscriminate expansion.

## Prepare data

Training must include all six generated roots:

```bash
uv run subfast-frame-presence-prepare \
  --source-root data/generated_samples1 \
  --source-root data/generated_samples2 \
  --source-root data/generated_samples3 \
  --source-root data/generated_samples4 \
  --source-root data/generated_samples5 \
  --source-root data/generated_samples6 \
  --output data/frame_presence_train_cache

uv run subfast-frame-presence-prepare \
  --source-root data/validation_samples \
  --output data/frame_presence_validation_cache
```

Label masks are applied before caching. Every remaining validation sample is used;
there is no validation subsampling.

## Train

```bash
uv run subfast-frame-presence \
  --train-cache data/frame_presence_train_cache \
  --val-cache data/frame_presence_validation_cache \
  --output-dir outputs/frame_presence \
  --width 16 \
  --heatmap-threshold 0.5
```

The trainer validates presence separation, per-frame region containment, active
overflow, and full-frame-box avoidance. A separable presence head is affine
calibrated at the end of training to place the weakest positive near `0.95` and
the strongest negative near `0.05`.

## Export and benchmark

`subfast_export` 支持直接读取 `best.pt` 或 `best_inference.pt`，并保留三个输入
`images`、`focus`、`focus_mode` 与两个输出 `presence_logits`、`region_logits`：

```bash
uv run subfast-export safetensors \
  outputs/frame_presence/best_inference.pt \
  outputs/frame_presence/safetensors

uv run subfast-export unified \
  outputs/frame_presence/best_inference.pt \
  outputs/frame_presence/unified

uv run subfast-export coreml \
  outputs/frame_presence/best_inference.pt \
  outputs/frame_presence/model.mlpackage
```

```bash
uv run subfast-frame-presence-benchmark \
  outputs/frame_presence/best_inference.pt \
  --device mps \
  --coreml-output outputs/frame_presence/best.mlpackage \
  --coreml-maximum-batch-size 32 \
  --validation-cache data/frame_presence_validation_cache
```

传入单帧图片时，可同时保存原图、`64×72` 原始热力图、叠加图和字幕区域放大图：

```bash
uv run subfast-frame-presence-benchmark \
  outputs/frame_presence/best_inference.pt \
  --image path/to/frame.jpg \
  --visualization-output outputs/frame_presence/frame_heatmap.jpg \
  --batch-size 1
```

The Core ML export accepts flexible batch sizes from 1 through 32. The benchmark
reports preprocessing separately from model inference and revalidates the exported
model rather than assuming PyTorch/Core ML parity.
