from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


DEFAULT_DET_MODEL_NAME = "PP-OCRv5_server_det"


@dataclass(frozen=True)
class Detection:
    polygon: list[list[float]]
    score: float | None = None


@dataclass
class VideoStats:
    decoded_frames: int = 0
    selected_frames: int = 0
    detected_frames: int = 0
    kept_frames: int = 0
    detections: int = 0
    decode_seconds: float = 0.0
    detect_seconds: float = 0.0
    write_seconds: float = 0.0
    start_seconds: float = 0.0

    def begin(self) -> None:
        self.start_seconds = time.perf_counter()

    def elapsed_seconds(self) -> float:
        return max(time.perf_counter() - self.start_seconds, 1e-9)


@dataclass
class PendingFrame:
    frame_index: int
    image: Any
    width: int
    height: int
    offset_x: int
    offset_y: int


def parse_video_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract video frames and build subtitle detector samples with PaddleOCR text detection."
    )
    parser.add_argument("videos", nargs="+", help="Input video files.")
    parser.add_argument(
        "-o",
        "--output",
        default="data/generated_samples",
        help="Output dataset directory.",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=30,
        help="Keep one frame every N decoded frames.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Maximum kept frames per video. 0 means unlimited.",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="Skip decoded frames before this frame index. Useful for resuming interrupted jobs.",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.5,
        help="Drop detections below this confidence when the model returns scores.",
    )
    parser.add_argument(
        "--roi",
        default=None,
        help="Optional crop region as x1,y1,x2,y2 in source-frame pixels.",
    )
    parser.add_argument(
        "--filter-region",
        default=None,
        help="Only keep detections whose bbox center is inside x1,y1,x2,y2.",
    )
    parser.add_argument(
        "--save-empty",
        action="store_true",
        help="Save frames even when no subtitle/text region is detected.",
    )
    parser.add_argument(
        "--yolo-labels",
        action="store_true",
        default=True,
        help="Also write YOLO txt labels with class 0 for each bbox.",
    )
    parser.add_argument(
        "--no-yolo-labels",
        action="store_false",
        dest="yolo_labels",
        help="Do not write YOLO txt labels.",
    )
    parser.add_argument(
        "--det-limit-side-len",
        type=int,
        default=960,
        help="Resize the longest side for text detection. Lower values are faster.",
    )
    parser.add_argument(
        "--det-model-name",
        default=DEFAULT_DET_MODEL_NAME,
        help="PaddleOCR text detection model name.",
    )
    parser.add_argument(
        "--video-backend",
        choices=("opencv", "ffmpeg"),
        default="opencv",
        help="Video decode backend. Use ffmpeg when OpenCV's default decoder gives poor results.",
    )
    parser.add_argument(
        "--boxed-images",
        action="store_true",
        help="Also write preview images with detected boxes drawn on them.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=30,
        help="Print progress every N selected frames. Use 1 for per-frame logs, 0 for summary only.",
    )
    parser.add_argument(
        "--det-batch-size",
        type=int,
        default=1,
        help="Run text detection on N selected frames at once. Larger values can improve throughput.",
    )
    return parser.parse_args(argv)


def frame_index_selected(frame_index: int, frame_stride: int) -> bool:
    if frame_stride <= 0:
        raise ValueError("--frame-stride must be greater than 0")
    return frame_index % frame_stride == 0


def parse_roi(value: str | None) -> tuple[int, int, int, int] | None:
    if value is None:
        return None
    parts = [int(part.strip()) for part in value.split(",")]
    if len(parts) != 4:
        raise ValueError("--roi must use x1,y1,x2,y2")
    x1, y1, x2, y2 = parts
    if x2 <= x1 or y2 <= y1:
        raise ValueError("--roi requires x2 > x1 and y2 > y1")
    return x1, y1, x2, y2


def normalize_quad(
    quad: Iterable[Iterable[float]], width: int, height: int
) -> list[list[float]]:
    max_x = float(max(width - 1, 0))
    max_y = float(max(height - 1, 0))
    points: list[list[float]] = []
    for point in quad:
        x, y = point
        points.append([min(max(float(x), 0.0), max_x), min(max(float(y), 0.0), max_y)])
    if len(points) != 4:
        raise ValueError("detection polygon must contain exactly four points")
    return points


def yolo_bbox_from_quad(
    detection: Detection, width: int, height: int
) -> list[int | float]:
    xs = [point[0] for point in detection.polygon]
    ys = [point[1] for point in detection.polygon]
    x_min = min(xs)
    x_max = max(xs)
    y_min = min(ys)
    y_max = max(ys)
    return [
        0,
        round(((x_min + x_max) / 2.0) / width, 6),
        round(((y_min + y_max) / 2.0) / height, 6),
        round((x_max - x_min) / width, 6),
        round((y_max - y_min) / height, 6),
    ]


def detection_center(detection: Detection) -> tuple[float, float]:
    xs = [point[0] for point in detection.polygon]
    ys = [point[1] for point in detection.polygon]
    return (min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0


def detection_bbox(
    detection: Detection, offset_x: int = 0, offset_y: int = 0
) -> tuple[float, float, float, float]:
    xs = [point[0] + offset_x for point in detection.polygon]
    ys = [point[1] + offset_y for point in detection.polygon]
    return min(xs), min(ys), max(xs), max(ys)


def bbox_intersects_region(
    bbox: tuple[float, float, float, float], region: tuple[int, int, int, int]
) -> bool:
    bbox_x1, bbox_y1, bbox_x2, bbox_y2 = bbox
    region_x1, region_y1, region_x2, region_y2 = region
    return (
        bbox_x1 <= region_x2
        and bbox_x2 >= region_x1
        and bbox_y1 <= region_y2
        and bbox_y2 >= region_y1
    )


def filter_detections_by_region(
    detections: list[Detection],
    region: tuple[int, int, int, int] | None,
    offset_x: int = 0,
    offset_y: int = 0,
) -> list[Detection]:
    if region is None:
        return detections
    return [
        detection
        for detection in detections
        if bbox_intersects_region(detection_bbox(detection, offset_x, offset_y), region)
    ]


def load_cv2() -> Any:
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Missing dependency: install opencv-python.") from exc
    return cv2


def make_text_detection_options(
    limit_side_len: int = 960, model_name: str = DEFAULT_DET_MODEL_NAME
) -> dict[str, Any]:
    return {
        "model_name": model_name,
        "enable_mkldnn": False,
        "limit_side_len": limit_side_len,
    }


def create_text_detector(limit_side_len: int, model_name: str) -> Any:
    try:
        from paddleocr import TextDetection  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Missing dependency: install paddleocr.") from exc

    return TextDetection(**make_text_detection_options(limit_side_len, model_name))


def detect_text_regions(detector: Any, image: Any, width: int, height: int) -> list[Detection]:
    result = detector.predict(image)
    return parse_paddle_detections(result, width, height)


def parse_paddle_detections(result: Any, width: int, height: int) -> list[Detection]:
    detections: list[Detection] = []
    for polygon, score in iter_paddle_polygons(result):
        if polygon is None:
            continue
        detections.append(Detection(normalize_quad(polygon, width, height), score))
    return detections


def iter_paddle_polygons(result: Any) -> Iterable[tuple[list[list[float]] | None, float | None]]:
    if result is None:
        return
    if hasattr(result, "tolist"):
        result = result.tolist()
    if isinstance(result, dict):
        if "dt_polys" in result:
            polygons = result["dt_polys"]
            if hasattr(polygons, "tolist"):
                polygons = polygons.tolist()
            scores = result.get("dt_scores")
            if scores is None:
                scores = []
            if hasattr(scores, "tolist"):
                scores = scores.tolist()
            for index, polygon in enumerate(polygons):
                if hasattr(polygon, "tolist"):
                    polygon = polygon.tolist()
                score = scores[index] if index < len(scores) else None
                if hasattr(score, "item"):
                    score = score.item()
                yield polygon, float(score) if isinstance(score, (int, float)) else None
            return
        yield extract_polygon_and_score(result)
        return
    if isinstance(result, list):
        if is_quad(result):
            yield result, None
            return
        if is_scored_polygon(result):
            polygon = result[0]
            score = result[1]
            yield polygon, float(score) if isinstance(score, (int, float)) else None
            return
        for item in result:
            yield from iter_paddle_polygons(item)
        return
    yield None, None


def flatten_paddle_result(result: Any) -> Iterable[Any]:
    if result is None:
        return []
    if isinstance(result, dict):
        for key in ("dt_polys", "rec_polys", "boxes"):
            if key in result:
                return result[key]
        return []
    if isinstance(result, list) and len(result) == 1 and isinstance(result[0], list):
        return result[0]
    return result


def extract_polygon_and_score(item: Any) -> tuple[list[list[float]] | None, float | None]:
    if hasattr(item, "tolist"):
        item = item.tolist()
    if isinstance(item, dict):
        polygon = None
        for key in ("points", "poly", "bbox"):
            if key in item:
                polygon = item[key]
                break
        score = item.get("score")
        return polygon, score
    if not isinstance(item, list):
        return None, None
    if len(item) == 4 and all(is_point(point) for point in item):
        return item, None
    if item and isinstance(item[0], list):
        polygon = item[0]
        score = None
        if len(item) > 1 and isinstance(item[1], (int, float)):
            score = float(item[1])
        return polygon, score
    return None, None


def is_quad(value: Any) -> bool:
    return isinstance(value, list) and len(value) == 4 and all(is_point(point) for point in value)


def is_scored_polygon(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) >= 2
        and is_quad(value[0])
        and isinstance(value[1], (int, float))
    )


def is_point(value: Any) -> bool:
    return isinstance(value, (list, tuple)) and len(value) == 2


def write_yolo_label(path: Path, detections: list[Detection], width: int, height: int) -> None:
    lines = [
        " ".join(str(value) for value in yolo_bbox_from_quad(detection, width, height))
        for detection in detections
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_boxed_image(cv2: Any, path: Path, image: Any, detections: list[Detection]) -> None:
    import numpy as np

    boxed = image.copy()
    for detection in detections:
        points = [[int(round(x)), int(round(y))] for x, y in detection.polygon]
        cv2.polylines(boxed, [np.array(points, dtype=np.int32)], True, (0, 255, 0), 2)
    cv2.imwrite(str(path), boxed)


def sample_stem(video_prefix: str, frame_index: int) -> str:
    return f"{video_prefix}_f{frame_index:08d}"


def build_samples(args: argparse.Namespace) -> int:
    cv2 = load_cv2()
    if args.start_frame < 0:
        raise ValueError("--start-frame must be greater than or equal to 0")
    if args.det_batch_size <= 0:
        raise ValueError("--det-batch-size must be greater than 0")
    detector = create_text_detector(args.det_limit_side_len, args.det_model_name)
    output = Path(args.output)
    image_dir = output / "images"
    label_dir = output / "labels"
    boxed_dir = output / "boxed_images"
    image_dir.mkdir(parents=True, exist_ok=True)
    if args.yolo_labels:
        label_dir.mkdir(parents=True, exist_ok=True)
    if args.boxed_images:
        boxed_dir.mkdir(parents=True, exist_ok=True)

    roi = parse_roi(args.roi)
    filter_region = parse_roi(args.filter_region)
    annotations_path = output / "annotations.jsonl"
    annotations_mode = "a" if args.start_frame > 0 and annotations_path.exists() else "w"
    kept_total = 0

    with annotations_path.open(annotations_mode, encoding="utf-8") as annotations:
        for video_number, video_arg in enumerate(args.videos, start=1):
            print(f"processing video: {video_arg}", flush=True)
            kept_total += process_video(
                cv2=cv2,
                detector=detector,
                video_path=Path(video_arg),
                video_prefix=f"video{video_number:04d}",
                image_dir=image_dir,
                label_dir=label_dir,
                boxed_dir=boxed_dir,
                annotations=annotations,
                frame_stride=args.frame_stride,
                start_frame=args.start_frame,
                max_frames=args.max_frames,
                min_score=args.min_score,
                roi=roi,
                filter_region=filter_region,
                video_backend=args.video_backend,
                save_empty=args.save_empty,
                yolo_labels=args.yolo_labels,
                boxed_images=args.boxed_images,
                log_every=args.log_every,
                det_batch_size=args.det_batch_size,
            )

    print(f"generated {kept_total} samples in {output}")
    return 0


def process_video(
    *,
    cv2: Any,
    detector: Any,
    video_path: Path,
    video_prefix: str,
    image_dir: Path,
    label_dir: Path,
    boxed_dir: Path,
    annotations: Any,
    frame_stride: int,
    start_frame: int,
    max_frames: int,
    min_score: float,
    roi: tuple[int, int, int, int] | None,
    filter_region: tuple[int, int, int, int] | None,
    video_backend: str,
    save_empty: bool,
    yolo_labels: bool,
    boxed_images: bool,
    log_every: int,
    det_batch_size: int,
) -> int:
    if not video_path.exists():
        raise FileNotFoundError(f"video not found: {video_path}")

    capture = open_video_capture(cv2, video_path, video_backend)
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")

    frame_index = seek_video_capture(cv2, capture, start_frame)
    stats = VideoStats()
    stats.begin()
    pending: list[PendingFrame] = []
    try:
        while True:
            if not frame_index_selected(frame_index, frame_stride):
                decode_start = time.perf_counter()
                ok = capture.grab()
                stats.decode_seconds += time.perf_counter() - decode_start
                if not ok:
                    break
                stats.decoded_frames += 1
                frame_index += 1
                continue
            if max_frames and stats.kept_frames >= max_frames:
                break

            decode_start = time.perf_counter()
            ok, frame = capture.read()
            stats.decode_seconds += time.perf_counter() - decode_start
            if not ok:
                break
            stats.decoded_frames += 1
            stats.selected_frames += 1

            sample_image, offset_x, offset_y = crop_frame(frame, roi)
            height, width = sample_image.shape[:2]
            pending.append(
                PendingFrame(
                    frame_index=frame_index,
                    image=sample_image,
                    width=width,
                    height=height,
                    offset_x=offset_x,
                    offset_y=offset_y,
                )
            )
            if len(pending) >= det_batch_size:
                process_pending_frames(
                    cv2=cv2,
                    detector=detector,
                    pending=pending,
                    video_path=video_path,
                    image_dir=image_dir,
                    label_dir=label_dir,
                    boxed_dir=boxed_dir,
                    annotations=annotations,
                    video_prefix=video_prefix,
                    min_score=min_score,
                    filter_region=filter_region,
                    save_empty=save_empty,
                    yolo_labels=yolo_labels,
                    boxed_images=boxed_images,
                    log_every=log_every,
                    stats=stats,
                )
                pending.clear()
            frame_index += 1
        if pending:
            process_pending_frames(
                cv2=cv2,
                detector=detector,
                pending=pending,
                video_path=video_path,
                image_dir=image_dir,
                label_dir=label_dir,
                boxed_dir=boxed_dir,
                annotations=annotations,
                video_prefix=video_prefix,
                min_score=min_score,
                filter_region=filter_region,
                save_empty=save_empty,
                yolo_labels=yolo_labels,
                boxed_images=boxed_images,
                log_every=log_every,
                stats=stats,
            )
    finally:
        capture.release()
    print_video_summary(video_path, stats)
    return stats.kept_frames


def process_pending_frames(
    *,
    cv2: Any,
    detector: Any,
    pending: list[PendingFrame],
    video_path: Path,
    image_dir: Path,
    label_dir: Path,
    boxed_dir: Path,
    annotations: Any,
    video_prefix: str,
    min_score: float,
    filter_region: tuple[int, int, int, int] | None,
    save_empty: bool,
    yolo_labels: bool,
    boxed_images: bool,
    log_every: int,
    stats: VideoStats,
) -> None:
    detect_start = time.perf_counter()
    batch_results = detect_text_regions_batch(detector, pending)
    stats.detect_seconds += time.perf_counter() - detect_start
    stats.detected_frames += len(pending)

    for pending_frame, detections in zip(pending, batch_results, strict=True):
        detections = [
            detection
            for detection in detections
            if detection.score is None or detection.score >= min_score
        ]
        detections = filter_detections_by_region(
            detections,
            filter_region,
            offset_x=pending_frame.offset_x,
            offset_y=pending_frame.offset_y,
        )
        stats.detections += len(detections)
        if not detections and not save_empty:
            print_video_progress(video_path, pending_frame.frame_index, stats, log_every)
            continue

        stem = sample_stem(video_prefix, pending_frame.frame_index)
        image_path = image_dir / f"{stem}.jpg"
        write_start = time.perf_counter()
        cv2.imwrite(str(image_path), pending_frame.image)
        if yolo_labels:
            write_yolo_label(
                label_dir / f"{stem}.txt",
                detections,
                pending_frame.width,
                pending_frame.height,
            )
        boxed_path = None
        if boxed_images:
            boxed_path = boxed_dir / f"{stem}.jpg"
            write_boxed_image(cv2, boxed_path, pending_frame.image, detections)

        annotations.write(
            json.dumps(
                {
                    "image": str(image_path.as_posix()),
                    "source_video": str(video_path),
                    "frame_index": pending_frame.frame_index,
                    "image_width": pending_frame.width,
                    "image_height": pending_frame.height,
                    "roi_offset": [pending_frame.offset_x, pending_frame.offset_y],
                    "filter_region": list(filter_region) if filter_region else None,
                    "boxed_image": str(boxed_path.as_posix()) if boxed_path else None,
                    "detections": [
                        {"polygon": detection.polygon, "score": detection.score}
                        for detection in detections
                    ],
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        stats.write_seconds += time.perf_counter() - write_start
        stats.kept_frames += 1
        print_video_progress(video_path, pending_frame.frame_index, stats, log_every)


def detect_text_regions_batch(
    detector: Any, pending: list[PendingFrame]
) -> list[list[Detection]]:
    if len(pending) == 1:
        frame = pending[0]
        return [detect_text_regions(detector, frame.image, frame.width, frame.height)]
    results = detector.predict(
        [frame.image for frame in pending], batch_size=len(pending)
    )
    return [
        parse_paddle_detections(result, frame.width, frame.height)
        for result, frame in zip(results, pending, strict=True)
    ]


def print_video_progress(
    video_path: Path, frame_index: int, stats: VideoStats, log_every: int
) -> None:
    if log_every <= 0:
        return
    if stats.selected_frames % log_every != 0 and log_every != 1:
        return
    print(
        f"progress {video_path.name}: frame={frame_index} "
        f"selected={stats.selected_frames} kept={stats.kept_frames} "
        f"decoded_fps={stats.decoded_frames / stats.elapsed_seconds():.2f} "
        f"selected_fps={stats.selected_frames / stats.elapsed_seconds():.2f} "
        f"avg_decode_ms={average_ms(stats.decode_seconds, stats.decoded_frames):.2f} "
        f"avg_det_ms={average_ms(stats.detect_seconds, stats.detected_frames):.2f} "
        f"avg_write_ms={average_ms(stats.write_seconds, stats.kept_frames):.2f} "
        f"decode_s={stats.decode_seconds:.1f} det_s={stats.detect_seconds:.1f} "
        f"write_s={stats.write_seconds:.1f}",
        flush=True,
    )


def print_video_summary(video_path: Path, stats: VideoStats) -> None:
    elapsed = stats.elapsed_seconds()
    print(
        f"summary {video_path.name}: decoded={stats.decoded_frames} "
        f"selected={stats.selected_frames} detected={stats.detected_frames} "
        f"kept={stats.kept_frames} detections={stats.detections} "
        f"elapsed_s={elapsed:.1f} decoded_fps={stats.decoded_frames / elapsed:.2f} "
        f"selected_fps={stats.selected_frames / elapsed:.2f} "
        f"decode_s={stats.decode_seconds:.1f} det_s={stats.detect_seconds:.1f} "
        f"write_s={stats.write_seconds:.1f} "
        f"avg_decode_ms={average_ms(stats.decode_seconds, stats.decoded_frames):.2f} "
        f"avg_det_ms={average_ms(stats.detect_seconds, stats.detected_frames):.2f} "
        f"avg_write_ms={average_ms(stats.write_seconds, stats.kept_frames):.2f}",
        flush=True,
    )


def average_ms(total_seconds: float, count: int) -> float:
    if count <= 0:
        return 0.0
    return total_seconds * 1000.0 / count


def open_video_capture(cv2: Any, video_path: Path, video_backend: str) -> Any:
    if video_backend == "opencv":
        return cv2.VideoCapture(str(video_path))
    if video_backend == "ffmpeg":
        if not hasattr(cv2, "CAP_FFMPEG"):
            raise RuntimeError("OpenCV build does not expose the FFmpeg video backend.")
        return cv2.VideoCapture(str(video_path), cv2.CAP_FFMPEG)
    raise ValueError("--video-backend must be either opencv or ffmpeg")


def seek_video_capture(cv2: Any, capture: Any, start_frame: int) -> int:
    if start_frame <= 0:
        return 0
    if not hasattr(cv2, "CAP_PROP_POS_FRAMES"):
        print(
            "warning: OpenCV build does not expose CAP_PROP_POS_FRAMES; "
            f"falling back to sequential skip before frame {start_frame}",
            flush=True,
        )
        return 0
    if not capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame):
        print(
            f"warning: video backend could not seek to frame {start_frame}; "
            "falling back to sequential skip",
            flush=True,
        )
        return 0
    return start_frame


def crop_frame(frame: Any, roi: tuple[int, int, int, int] | None) -> tuple[Any, int, int]:
    if roi is None:
        return frame, 0, 0
    x1, y1, x2, y2 = roi
    return frame[y1:y2, x1:x2], x1, y1


def main(argv: list[str] | None = None) -> int:
    args = parse_video_args(argv)
    try:
        return build_samples(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
