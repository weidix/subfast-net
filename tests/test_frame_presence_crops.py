import random
import unittest
from pathlib import Path

from subfast_frame_presence.dataset import FramePresenceSample, _crop_boxes, _random_crop_box
from subfast_shared.geometry import Box


class FramePresenceCropTests(unittest.TestCase):
    def test_random_crop_keeps_target_box_and_clips_only_intersections(self) -> None:
        target = Box(40.9, 70.7, 60.1, 80.2)
        sample = FramePresenceSample(
            image_path=Path("frame.jpg"),
            label_path=Path("frame.txt"),
            sample_id="frame",
            root=Path("samples"),
            source_size=(100, 100),
            boxes=(target,),
            ignored_boxes=(),
        )

        crop = _random_crop_box(sample, rng=random.Random(7), min_scale=0.5, max_scale=0.5)
        left, top, right, bottom = crop
        self.assertEqual(right - left, 50)
        self.assertEqual(bottom - top, 50)
        self.assertLessEqual(left, target.x1)
        self.assertLessEqual(target.x2, right)
        self.assertLessEqual(top, target.y1)
        self.assertLessEqual(target.y2, bottom)

        cropped = _crop_boxes((target, Box(0.0, 0.0, 10.0, 10.0)), crop)
        self.assertEqual(
            cropped,
            (Box(target.x1 - left, target.y1 - top, target.x2 - left, target.y2 - top),),
        )
