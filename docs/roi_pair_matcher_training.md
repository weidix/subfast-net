# ROI Direct Pair Matcher

`train-roi-pair` 是独立的同字幕二分类训练路径。它直接输入两个 `256×64` ROI；训练 forward 返回同字幕 logit 和辅助 mask，优化后的部署 runtime 只返回 logit，调用 sigmoid 后得到同字幕 score。不包含 presence head，也不输出 embedding descriptor。

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
  --output-dir outputs/roi_pair_matcher \
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
  outputs/roi_pair_matcher/best_inference.pt \
  --root data/roi_validation_samples \
  --resize-roi 256x64 \
  --batch-size 256 \
  --device mps
```

验证会先用普通 eval 模型计算完整 held-out 指标，再用同一组权重建立部署副本：融合 8 组 Conv-BatchNorm，并按 batch-1 输入 trace 固定推理图。`pair_forward_median_ms` 与 `pair_forward_p90_ms` 测量这个部署图。

计时口径是 FP32、batch 1、`256×64`、输入已在设备、40 次 warmup、500 次逐次 forward + MPS synchronize；不包含图片读取、resize、传输、sigmoid、阈值或 CPU 读取。

部署时必须通过优化 loader 在目标设备准备 runtime；portable checkpoint 本身仍保存普通 state dict：

```python
import torch
from pathlib import Path

from src.train_roi_pair import load_pair_inference_checkpoint

runtime, _ = load_pair_inference_checkpoint(
    Path("outputs/roi_pair_matcher/best_inference.pt"),
    torch.device("mps"),
)
pair_logit = runtime(left_roi, right_roi)
pair_score = pair_logit.sigmoid()
```

## 产物

- `best.pt`：模型、optimizer、设置和最佳指标，用于继续训练。
- `best_inference.pt`：仅推理所需权重、输入尺寸和阈值。
- `last.pt`：最后一个 epoch，可用 `--resume` 继续。
- `metrics.jsonl`：每个 epoch 的训练和完整验证指标。
- `best_pair_scores.jsonl`：最佳 checkpoint 对全部验证 pair 的逐对得分。
- `summary.json`：最佳 epoch、模型大小和最终指标。

旧 Top-K checkpoint 可以继续加载权重；`--resume` 会先用当前 width-peak 语义重新验证并刷新 best rank，避免混用两种 pooling 指标。

当前仓库产物 `outputs/roi_pair_matcher/best_inference.pt` 有 17,074 个可训练参数。将 width-tail 从 Top-K 均值改为宽度最大响应后，旧权重在完整 2,724 个 held-out pair 上仍为 `FP=0`、`FN=0`，`pair_gap=+0.956748`；刷新后的 `summary.json` 记录 Apple M4 / PyTorch 2.12.1 MPS median `0.560 ms`、P90 `0.661 ms`。
