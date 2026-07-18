import unittest

import numpy as np

from subfast_net.detector.geometry import Box, letterbox_shape, yolo_to_box
from subfast_net.detector.metrics import evaluate_image
from subfast_net.detector.postprocess import logits_to_boxes
from subfast_net.detector.targets import build_targets


class GeometryTests(unittest.TestCase):
    def test_yolo_box_converts_to_pixel_box_and_clips(self):
        box = yolo_to_box((0, 0.5, 0.5, 0.25, 0.5), width=200, height=100)
        self.assertEqual(box, Box(75.0, 25.0, 125.0, 75.0))

    def test_letterbox_shape_keeps_aspect_and_alignment(self):
        shape = letterbox_shape(width=1920, height=1080, size=256, stride=32)
        self.assertEqual(shape.resized_width, 256)
        self.assertEqual(shape.resized_height, 144)
        self.assertEqual(shape.padded_width, 256)
        self.assertEqual(shape.padded_height, 160)
        self.assertAlmostEqual(shape.scale_x, 256 / 1920)
        self.assertAlmostEqual(shape.scale_y, 144 / 1080)


class TargetTests(unittest.TestCase):
    def test_build_targets_creates_region_kernel_and_training_mask(self):
        targets = build_targets(
            width=32,
            height=16,
            boxes=[Box(4, 4, 20, 12)],
            ignore_regions=[Box(0, 0, 4, 4)],
            kernel_scale=0.25,
        )
        self.assertEqual(targets.region.shape, (16, 32))
        self.assertEqual(targets.kernel.shape, (16, 32))
        self.assertEqual(targets.training_mask[1, 1], 0.0)
        self.assertEqual(targets.region[6, 6], 1.0)
        self.assertEqual(targets.kernel[8, 12], 1.0)
        self.assertLess(targets.kernel.sum(), targets.region.sum())

    def test_build_targets_adds_eroded_kernel_interior_for_wide_subtitle(self):
        targets = build_targets(
            width=100,
            height=40,
            boxes=[Box(10, 10, 90, 30)],
            kernel_scale=0.1,
            pooling_size=9,
        )

        self.assertGreater(targets.kernel.sum(), 400)
        self.assertLess(targets.kernel.sum(), targets.region.sum())


class PostprocessTests(unittest.TestCase):
    def test_logits_to_boxes_grows_kernel_seed_inside_region(self):
        region = np.full((16, 32), -8.0, dtype=np.float32)
        kernel = np.full((16, 32), -8.0, dtype=np.float32)
        region[4:12, 4:20] = 8.0
        kernel[7:9, 10:14] = 8.0

        boxes = logits_to_boxes(region, kernel, region_threshold=0.5, kernel_threshold=0.5, min_size=2)

        self.assertEqual(len(boxes), 1)
        self.assertEqual(boxes[0].box, Box(4.0, 4.0, 20.0, 12.0))
        self.assertGreater(boxes[0].score, 0.8)

    def test_logits_to_boxes_refines_wide_component_to_high_confidence_core(self):
        region = np.full((4, 16), -8.0, dtype=np.float32)
        kernel = np.full((4, 16), -8.0, dtype=np.float32)
        region[1, :] = 1.0
        region[1, 5:11] = 4.0
        kernel[1, 7:9] = 4.0

        boxes = logits_to_boxes(region, kernel, region_threshold=0.5, kernel_threshold=0.5, min_size=1)

        self.assertEqual(len(boxes), 1)
        self.assertEqual(boxes[0].box, Box(5.0, 1.0, 11.0, 2.0))

    def test_logits_to_boxes_suppresses_upper_candidates_when_bottom_subtitle_exists(self):
        region = np.full((20, 20), -8.0, dtype=np.float32)
        kernel = np.full((20, 20), -8.0, dtype=np.float32)
        for x1, y1, x2, y2 in [(8, 4, 12, 6), (6, 17, 14, 19)]:
            region[y1:y2, x1:x2] = 4.0
            kernel[y1:y2, x1:x2] = 4.0

        boxes = logits_to_boxes(region, kernel, region_threshold=0.5, kernel_threshold=0.5, min_size=1)

        self.assertEqual(len(boxes), 1)
        self.assertGreaterEqual(boxes[0].box.y1, 17.0)


class MetricTests(unittest.TestCase):
    def test_evaluate_image_matches_by_iou(self):
        result = evaluate_image(
            predictions=[Box(0, 0, 10, 10), Box(20, 20, 30, 30)],
            targets=[Box(1, 1, 11, 11)],
            iou_threshold=0.5,
        )
        self.assertEqual(result.true_positive, 1)
        self.assertEqual(result.false_positive, 1)
        self.assertEqual(result.false_negative, 0)

    def test_evaluate_image_merges_adjacent_subtitle_lines(self):
        result = evaluate_image(
            predictions=[Box(22, 47, 73, 54)],
            targets=[Box(24.3, 50.0, 71.6, 53.8), Box(22.8, 47.3, 72.8, 50.4)],
            iou_threshold=0.5,
        )
        self.assertEqual(result.true_positive, 1)
        self.assertEqual(result.false_positive, 0)
        self.assertEqual(result.false_negative, 0)

    def test_evaluate_image_merges_same_row_subtitle_fragments(self):
        result = evaluate_image(
            predictions=[Box(171, 265, 345, 288)],
            targets=[Box(172.3, 269.6, 268.3, 285.9), Box(276.3, 269.1, 339.7, 285.9)],
            iou_threshold=0.5,
        )

        self.assertEqual(result.true_positive, 1)
        self.assertEqual(result.false_positive, 0)
        self.assertEqual(result.false_negative, 0)


if __name__ == "__main__":
    unittest.main()
