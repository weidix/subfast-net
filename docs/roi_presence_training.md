# ROI Presence 模型训练

`train-presence` 是独立的 ROI 字幕存在性训练入口。模型只判断 ROI 中是否存在字幕，checkpoint 仅包含 backbone 与 presence head，不包含 embedding head、embedding loss 或同字幕配对逻辑。

旧的 `train-roi` 双头训练入口已弃用。`train` 检测器训练入口也已标记为弃用。

## 基本命令

```bash
uv run subfast-net train-presence \
  --train-root data/roi_samples6 \
  --val-root data/roi_validation_samples \
  --output-dir outputs/roi_presence_run \
  --epochs 10 \
  --batch-size 16
```

多个训练集通过重复传入 `--train-root` 合并：

```bash
uv run subfast-net train-presence \
  --train-root data/roi_samples1 \
  --train-root data/roi_samples2 \
  --train-root data/roi_samples3 \
  --train-root data/roi_samples4 \
  --train-root data/roi_samples5 \
  --train-root data/roi_samples6 \
  --val-root data/roi_validation_samples \
  --output-dir outputs/roi_presence_full \
  --epochs 30 \
  --batch-size 16 \
  --train-negative-ratio 0.35
```

训练集与验证集应分离。若 `--val-root` 与任一训练根目录相同，CLI 会提示该结果不能作为 held-out 质量依据。

## 数据与尺寸

输入使用 `tools/prepare_roi_samples.py` 生成的 ROI 数据目录。每个目录需要包含：

- `summary.json`
- `annotations.jsonl`
- ROI 图片
- `labels/` 下对应的标签文件

多个数据目录的 ROI 尺寸不同时，必须显式统一尺寸：

```bash
--resize-roi 256x64
```

尺寸格式为 `WIDTHxHEIGHT`。图片和可选的字幕 mask 会使用同一输出坐标空间。

## 类别采样

默认训练 batch 中无字幕样本比例为 `0.35`。可以使用正样本比例或负样本比例配置，但同一作用域只能指定一种：

```bash
--train-negative-ratio 0.35
--val-negative-ratio 0.50
```

或：

```bash
--train-positive-ratio 0.65
--val-positive-ratio 0.50
```

验证比例只有在同时设置 `--max-val-samples`、需要裁剪验证集时才会改变样本组成。

## 短字幕监督

可提高 OCR 归一化后长度不超过两个字符的正样本权重：

```bash
--short-positive-loss-weight 2.0
```

也可以启用短字幕 textness map 的辅助 mask loss：

```bash
--short-positive-mask-loss-weight 0.1
```

辅助 mask loss 只用于 presence 模型的局部 textness 监督，不会创建额外的推理输出头。

## 断点续训

`--resume` 可以指向 checkpoint 文件、单个 epoch 输出目录或完整训练输出目录：

```bash
uv run subfast-net train-presence \
  --train-root data/roi_samples6 \
  --val-root data/roi_validation_samples \
  --output-dir outputs/roi_presence_run \
  --resume outputs/roi_presence_run \
  --epochs 10
```

续训时 `--epochs` 表示本次继续执行的 epoch 数。例如已有 10 个 epoch，再传入 `--epochs 10`，本次会训练 epoch 11 至 20。

Presence CLI 只接受 `model_type=roi_presence` 的 checkpoint，不加载旧双头 checkpoint。

## 输出文件

```text
outputs/roi_presence_run/
├── best.pt
├── best_presence.pt
├── metrics.jsonl
├── summary.json
└── epoch_outputs/
    └── epoch_0001/
        ├── model.pt
        └── metrics.json
```

- `best.pt` 与 `best_presence.pt`：当前最佳 presence checkpoint。
- `metrics.jsonl`：训练 step 和每轮验证指标。
- `summary.json`：最佳轮次、checkpoint 路径及验证摘要。
- `epoch_outputs/`：每轮完整 checkpoint 和指标。

每轮验证额外输出基于 sigmoid 概率的分离度：

- `presence_min_positive_score`：最难正样本分数。
- `presence_max_negative_score`：最难负样本分数。
- `presence_gap = min_positive - max_negative`。大于 `0` 表示验证集存在零错误阈值，小于等于 `0` 表示正负分数仍有重叠。
- `presence_roc_auc`、正负样本尾部分位数、`presence_best_f1_threshold` 和 `presence_best_f1` 用于区分整体排序质量、尾部重叠和固定 `0.5` 阈值问题。

最佳 checkpoint 仍以全局、普通字幕和短字幕三个 presence F1 的平均值为首要标准；F1 相同时选择 `presence_gap` 更大的轮次。正式质量结论应使用独立验证集上的指标。

## 主要参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--batch-size` | `16` | 训练与验证 batch size |
| `--epochs` | `1` | 本次执行的 epoch 数 |
| `--lr` | `3e-4` | AdamW 学习率 |
| `--weight-decay` | `1e-4` | AdamW weight decay |
| `--train-negative-ratio` | `0.35` | 每个训练 batch 的目标负样本比例 |
| `--val-negative-ratio` | 不限制 | 验证集裁剪时的目标负样本比例 |
| `--presence-topk-ratio` | `0.05` | textness map 中用于聚合 presence logit 的最高响应比例 |
| `--width` | `32` | 模型基础通道数 |
| `--num-workers` | `0` | DataLoader worker 数量 |
| `--device` | `auto` | `auto`、`cpu`、`mps` 或 `cuda` |
| `--max-train-samples` | 不限制 | 限制训练样本数量，适合冒烟验证 |
| `--max-val-samples` | 不限制 | 限制验证样本数量 |

查看全部参数：

```bash
uv run subfast-net train-presence --help
```
