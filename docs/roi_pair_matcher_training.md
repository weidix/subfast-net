# ROI Direct Pair Matcher

`train-roi-pair` 是独立的同字幕二分类训练路径。它直接输入两个 `256×64` ROI，输出同字幕概率；
不包含 presence head，也不输出 embedding descriptor。

## 训练

```bash
uv run subfast-net train-roi-pair \
  --train-root data/roi_samples1 \
  --train-root data/roi_samples2 \
  --train-root data/roi_samples3 \
  --train-root data/roi_samples4 \
  --train-root data/roi_samples5 \
  --train-root data/roi_samples6 \
  --val-root data/roi_validation_samples \
  --output-dir outputs/roi_pair_matcher_run \
  --resize-roi 256x64 \
  --batch-size 128 \
  --validation-batch-size 256 \
  --epochs 3 \
  --lr 0.001 \
  --min-lr 0.0003 \
  --seed 2028 \
  --device mps
```

默认判断阈值是 `0.5`。OCR 只用于构建可信负对，不进入模型输入。

## 验证

```bash
uv run subfast-net validate-roi-pair \
  outputs/roi_pair_matcher_run/best_inference.pt \
  --root data/roi_validation_samples \
  --resize-roi 256x64 \
  --batch-size 256 \
  --device mps
```

## 产物

- `best.pt`：模型、optimizer、设置和最佳指标，用于继续训练。
- `best_inference.pt`：仅推理所需权重、输入尺寸和阈值。
- `last.pt`：最后一个 epoch，可用 `--resume` 继续。
- `metrics.jsonl`：每个 epoch 的训练和完整验证指标。
- `best_pair_scores.jsonl`：最佳 checkpoint 对全部验证 pair 的逐对得分。
- `summary.json`：最佳 epoch、模型大小和最终指标。

当前已验证最佳产物是 `outputs/roi_pair_matcher_seed2028/best_inference.pt`：17,074 个可训练参数，
固定 `0.5` 阈值下 `FP=0`、`FN=0`、`pair_gap=+0.952548`。
