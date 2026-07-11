# ROI Presence V3 训练

`train-presence` 是独立的 ROI 字幕存在性训练入口。V3 checkpoint 只包含 presence 专属 backbone、字幕区域 head 与连续区域证据聚合器。

## V3 的任务定义

V3 保留 `/4` dense region 输出和原有训练目标，同时将单张 MPS 推理图压缩为少量常规算子：

- backbone 使用两个无 normalization 的 `5×5 stride-2 Conv + SiLU`；同一样本不受 batch 组成或 train/eval 状态影响。
- head 使用 dilated `3×3` 卷积输出 dense subtitle-region logits。
- presence 证据先做 `5×5` 局部平均，再只在有效 ROI 内取最强连续区域；没有 Top-K、插值或 log-sum-exp。
- 所有正样本、空样本和带文字干扰负样本都参与区域 BCE；正样本另有 Dice 与横纵投影覆盖损失。
- 训练会为每个正样本按稳定 key 选择同来源的两个固定无字幕 donor：擦除标注字幕区域并要求 presence 降低，把完整字幕区域换到 donor 背景并将其作为明确正例，同时用第二个负 donor 检查拼接接缝不会触发 presence。反事实损失按完整 batch 归一化，不会让仅含一个正例的 batch 获得整项辅助损失权重。

总损失为：

```text
presence BCE
+ region_loss_weight × (balanced region BCE + Dice + projection extent)
+ counterfactual_loss_weight × (erased negative BCE + necessity margin
  + transplanted positive BCE + seam-control negative BCE)
```

## 数据语义边界

现有 `roi_samples1..6` 和 `roi_validation_samples` 中，正样本全部有文字框，负样本全部没有文字框；当前标签因此只能监督“标注文字区域/空区域”。数据中还存在被标成正例的网址水印。

所以 V3 会单独统计：

- `text_distractor_count`
- `text_distractor_fpr`
- `subtitle_specificity_evaluable`

只有验证集中存在“有文字框但 `has_subtitle=false`”的水印、UI、场景字或广告样本时，`subtitle_specificity_evaluable` 才会为 `1`。否则即使 `fp=0/fn=0`，也不能解释成模型已经学会区分字幕与其他文字。

可在 `segment_review.json` 中把非字幕文字设为 `has_subtitle=false`，同时保留对应 label box，使其成为 text-distractor negative。严格训练可增加：

```bash
--require-text-distractor-negatives
```

缺少这类训练或验证样本时，该参数会直接终止训练。

## 基本命令

```bash
uv run subfast-net train-presence \
  --train-root data/roi_samples1 \
  --train-root data/roi_samples2 \
  --train-root data/roi_samples3 \
  --train-root data/roi_samples4 \
  --train-root data/roi_samples5 \
  --train-root data/roi_samples6 \
  --val-root data/roi_validation_samples \
  --output-dir outputs/roi_presence_v3 \
  --resize-roi 512x64 \
  --epochs 30 \
  --batch-size 16
```

默认验证根目录是独立的 `data/roi_validation_samples`。若验证根目录与训练根目录重合，结果会被标记为非 held-out 证据。

## 尺寸与有效坐标

不同来源的 ROI 宽高比差异很大。V3 在指定 `--resize-roi` 时默认等比例缩放并居中 padding：

```bash
--resize-roi 512x64 --resize-mode letterbox
```

图像、字幕 mask 和 valid mask 使用同一输出坐标空间；区域损失和 presence 聚合都忽略 padding。训练与部署快路径显式传入预处理已经生成的 valid mask；单输入便利路径也可从精确零 padding 恢复 stride-4 mask。仅在明确需要旧式非等比拉伸时使用：

```bash
--resize-mode stretch
```

## MPS 推理基准

验收口径固定为 Apple M4、PyTorch 2.12.1、FP32、batch 1、`512×64`、显式 valid mask、输入已在 MPS；40 次 warmup、500 次计时，每次 forward 都同步，不含图片读取、resize、传输、sigmoid 或 CPU 读取。

| 模型 | MPS median | MPS P90 | 参数量 |
|---|---:|---:|---:|
| V2 | `2.08 ms` | `3.97 ms` | `104,707` |
| V3 | `0.43–0.45 ms` | `0.54–0.61 ms` | `14,339` |

无需 checkpoint 即可复测算子延迟；权重值不改变执行图：

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 uv run subfast-net benchmark-presence --device mps
```

也可把 V3 checkpoint 作为第一个参数传入。训练结束后 `summary.json` 会记录 `presence_forward_median_ms`、`presence_forward_p90_ms` 和完整计时口径。V3 改变了模型容量和聚合算子，必须从头训练并重新检查 held-out 指标；延迟达标本身不代表检测质量达标。

## 采样先验与 score

默认不再强制 batch 正负比例，而是每个 epoch 按数据原始分布遍历样本。若显式设置平衡采样：

```bash
--train-negative-ratio 0.35
```

presence BCE 会按采样先验和目标先验做 importance correction，避免 35% 负样本 batch 把 sigmoid 截距误当成真实数据先验。目标先验默认使用训练数据正样本比例，也可显式指定：

```bash
--score-positive-prior 0.30
```

checkpoint 会保存 `score_contract`，包括目标先验、固定决策阈值和字幕特异性是否可评估。`presence_score` 仍应理解为该先验下的 evidence score；只有在独立 calibration split 上做 affine calibration，并在未参与拟合的测试集报告 NLL/Brier 后，才应称为概率置信度。

固定阈值可通过以下参数设置，训练期间不会用每轮验证集的最佳阈值替换它：

```bash
--decision-threshold 0.5
```

## 验证证据

每轮同时报告：

- 固定阈值下的 F1、FP、FN 与 segment 级 recall。
- 正负最差 1% 均值、tail gap，以及逐样本跨 epoch 最大漂移和最差 1% 漂移；另报正例分数下降、负例分数上升的 adverse drift，避免把置信度改善误判成退化。
- Brier、NLL、ECE。
- region IoU、Dice、框内外响应差和最大响应落框率。
- 擦除字幕后的 score drop/翻转率、换背景后的 recall，以及负图接缝 control FPR。
- 同一样本单独推理与混批推理的最大 logit 差；V3 正常应接近数值误差。
- text-distractor 专项 FPR；无此类样本时明确标记不可评估。

最佳 checkpoint 先比较固定阈值下全局、普通字幕和短字幕的最弱 F1，再比较 text-distractor、区域定位、反事实必要性、尾部间隔和 Brier。不会再用 sigmoid 的单点 `presence_gap` 奖励饱和过置信。

每轮保存逐样本诊断：

```text
outputs/roi_presence_v3/
├── best.pt
├── best_inference.pt
├── last.pt
├── metrics.jsonl
├── best_presence_scores.jsonl
├── last_presence_scores.jsonl
└── summary.json
```

主产物与 `pair_matcher` 一致：`best.pt` 用于续训，`best_inference.pt` 仅包含推理所需权重与预处理契约，`last.pt` 是最后一轮续训点，`best_presence_scores.jsonl` 对应最佳 checkpoint，`last_presence_scores.jsonl` 保留最后一轮逐样本结果并供续训首轮计算漂移。

`last_presence_scores.jsonl` 包含稳定 sample key、segment、target、sample kind、原始 score、前一轮 score、漂移、区域定位、擦除 score 和换背景 score，可直接定位尾部异常。

## 旧 checkpoint

V3 的 backbone 和证据聚合算子均已改变，不能从 V1/V2 checkpoint 续训。`--resume` 会严格检查 `architecture_version`、输入尺寸模式、证据聚合参数和模型宽度；不匹配时直接报错。

## 主要参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--batch-size` | `16` | 训练与验证 batch size |
| `--epochs` | `1` | 本次执行的 epoch 数 |
| `--lr` | `3e-4` | AdamW 学习率 |
| `--train-negative-ratio` | 不强制 | 可选的 batch 负样本比例 |
| `--score-positive-prior` | 训练集先验 | score 对应的目标正样本先验 |
| `--region-loss-weight` | `1.0` | dense region 总损失权重 |
| `--region-dice-weight` | `1.0` | 正样本区域 Dice 权重 |
| `--region-projection-weight` | `0.25` | 字幕横纵覆盖权重 |
| `--text-distractor-weight` | `4.0` | 非字幕文字框负区域权重 |
| `--counterfactual-loss-weight` | `0.5` | 擦除/换背景约束权重 |
| `--counterfactual-margin` | `2.0` | 原图与擦除图的 logit margin |
| `--evidence-kernel-size` | `5` | 连续区域支持核，必须为大于 1 的奇数 |
| `--decision-threshold` | `0.5` | 固定验证/部署阈值 |
| `--resize-mode` | `letterbox` | `letterbox` 或 `stretch` |
| `--width` | `16` | 模型基础通道数 |

查看全部参数：

```bash
uv run subfast-net train-presence --help
```
