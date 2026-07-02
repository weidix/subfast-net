# Sample Builder 命令

最常用：

```powershell
python tools/build_samples.py data\xxx.mp4 -o data\generated_samples --boxed-images --filter-region 0,600,1280,720
```

参数：

```text
data\xxx.mp4                 视频文件
-o data\generated_samples    输出目录
--boxed-images               输出带框预览图
--filter-region x1,y1,x2,y2  只保留这个区域内的检测框
--save-empty                 没检测到字幕也保存图片
--frame-stride 30            每 30 帧取 1 帧
--max-frames 100             最多生成 100 张
--start-frame 3000           seek 到第 3000 个解码帧附近开始，用于中断后续跑
--det-model-name MODEL       使用指定 PaddleOCR 检测模型，默认 PP-OCRv5_server_det
--det-limit-side-len 960     检测前最长边缩放尺寸，越小越快
--det-batch-size 4           每次批量检测 4 个抽样帧，通常比逐张 det 更快；显存/内存不够就调小
--video-backend ffmpeg       使用 OpenCV 的 FFmpeg 视频解码后端，默认 opencv
--log-every 30               每处理 30 个抽样帧输出一次 decode/det/write 耗时汇总；0 只输出最终汇总
```

续跑示例：

```powershell
python tools/build_samples.py data\xxx.mp4 -o data\generated_samples --start-frame 3000 --boxed-images --filter-region 0,600,1280,720
```

说明：

```text
--start-frame N              通过视频后端 seek 到解码帧索引 N 开始；如果 annotations.jsonl 已存在，会追加写入
--frame-stride 30            仍按原始帧索引取模筛选，例如 start-frame=3000 时会处理 3000、3030、3060...
progress/summary             avg_det_ms 高说明 PaddleOCR 检测慢；avg_decode_ms 或 decode_s 高说明视频读取慢；avg_write_ms 高通常是 boxed_images 或磁盘写入慢
```

切换 PaddleOCR 检测模型示例：

```powershell
python tools/build_samples.py data\xxx.mp4 -o data\generated_samples --det-model-name PP-OCRv4_server_det
```

使用 FFmpeg 视频解码后端示例：

```powershell
python tools/build_samples.py data\xxx.mp4 -o data\generated_samples --video-backend ffmpeg
```

说明：

```text
--video-backend opencv       当前默认行为，使用 OpenCV 默认 VideoCapture 后端
--video-backend ffmpeg       显式使用 cv2.CAP_FFMPEG；通常对部分编码、色彩和时间戳处理更稳
```

输出：

```text
images\        原图
labels\        YOLO 标签
boxed_images\  带框预览图
annotations.jsonl
```

VIA 标注转换：

```powershell
# labels -> VGG Image Annotator JSON
python tools/via_labels.py labels-to-via --labels-dir data\generated_samples\labels --images-dir data\generated_samples\images --annotations data\generated_samples\annotations.jsonl -o data\generated_samples\via_annotations.json

# VGG Image Annotator JSON -> labels
python tools/via_labels.py via-to-labels data\generated_samples\via_annotations.json --labels-dir data\generated_samples\labels
```

说明：

```text
labels-to-via     将现有 YOLO labels 转成 VIA 可导入的 JSON
via-to-labels     将 VIA 修正后的 JSON 转回 YOLO labels
class             固定为 subtitle
空标签图片        保留为空 .txt，可作为 hard negative
```

区域辅助工具：

```text
tools\filter_region_helper.html
```

选框审阅和屏蔽工具：

```powershell
python tools/review_generated_labels.py --samples-dir data\generated_samples
```

如果系统没有全局 `python`，使用项目虚拟环境：

```powershell
tools\.venv\Scripts\python.exe tools\review_generated_labels.py --samples-dir data\generated_samples
```

参数：

```text
--samples-dir DIR    generated_samples 目录，要求包含 images\ 和 labels\
--host 127.0.0.1     本地 Web 服务监听地址
--port 8765          本地 Web 服务端口
--no-open            启动后不自动打开浏览器
```

界面过滤：

```text
面积过小             找出面积小于阈值的框，单位：像素²，例如 2000
宽度小于             找出宽度小于阈值的框，单位：像素，例如 80
宽度大于             找出宽度大于阈值的框，单位：像素，例如 900
高度小于             找出高度小于阈值的框，单位：像素，例如 20
高度大于             找出高度大于阈值的框，单位：像素，例如 120
显示无区域图像       勾选后显示没有 label 区域的图片；默认过滤掉
显示已屏蔽框         勾选后仍显示已屏蔽的选框
```

快捷键：

```text
N / →                下一张图片
P / ←                上一张图片
J / ↓                下一个选框，循环选择
K / ↑                上一个选框，循环选择
Delete / M           屏蔽或取消屏蔽当前选框
A                    屏蔽当前图片中匹配过滤条件的所有选框
X                    屏蔽或取消屏蔽当前图像，屏蔽后不进入训练
B                    进入或退出新增选框模式，拖拽空白区域创建选框
Ctrl+A               全选当前图片的所有选框
鼠标拖拽             移动当前选框；拖拽四角调整大小
Ctrl+方向键          以 1px 微移当前选框；Alt 可切换为 10px
Ctrl+Shift+方向键    调整当前选框宽高；Alt 可切换为 10px
R                    重新载入
Esc                  输入框失焦或取消多选
```

说明：

```text
调整选框会直接写回 labels\*.txt 中对应的 YOLO 标签行。
屏蔽不会删除 labels\*.txt 中的 YOLO 标签行。
屏蔽图像不会删除 labels\*.txt，状态写入 generated_samples\label_masks.json 的 __image__.drop_image。
新增选框会追加写入 labels\*.txt。
屏蔽状态写入 generated_samples\label_masks.json。
训练读取 labels 时会读取 label_masks.json，忽略被屏蔽选框并跳过被屏蔽图像。
```
