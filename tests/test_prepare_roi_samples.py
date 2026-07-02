import sys
import unittest
import tempfile
import json
from argparse import Namespace
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import tools.prepare_roi_samples as roi_prep


class PrepareRoiSamplesTests(unittest.TestCase):
    def test_all_samples_use_global_label_union_region(self):
        top_sample = roi_prep.SourceSample(
            stem="top",
            image_path=Path("sample.jpg"),
            label_path=Path("sample.txt"),
            boxes=[
                roi_prep.LabelBox(
                    index=0,
                    class_id="0",
                    cx=0.20,
                    cy=0.25,
                    width=0.10,
                    height=0.10,
                    raw="0 0.20 0.25 0.10 0.10",
                )
            ],
            width=100,
            height=80,
            annotation={},
        )
        bottom_sample = roi_prep.SourceSample(
            stem="bottom",
            image_path=Path("sample.jpg"),
            label_path=Path("sample.txt"),
            boxes=[
                roi_prep.LabelBox(
                    index=0,
                    class_id="0",
                    cx=0.60,
                    cy=0.75,
                    width=0.20,
                    height=0.10,
                    raw="0 0.60 0.75 0.20 0.10",
                )
            ],
            width=100,
            height=80,
            annotation={},
        )
        empty_sample = roi_prep.SourceSample(
            stem="empty",
            image_path=Path("empty.jpg"),
            label_path=Path("empty.txt"),
            boxes=[],
            width=100,
            height=80,
            annotation={},
        )

        samples = [top_sample, bottom_sample, empty_sample]
        roi_width, roi_height = roi_prep.common_roi_size(samples)
        anchor = roi_prep.common_roi_anchor(samples)
        top_crop = roi_prep.crop_box_for_sample(top_sample, roi_width=roi_width, roi_height=roi_height, anchor=anchor)
        bottom_crop = roi_prep.crop_box_for_sample(bottom_sample, roi_width=roi_width, roi_height=roi_height, anchor=anchor)
        empty_crop = roi_prep.crop_box_for_sample(empty_sample, roi_width=roi_width, roi_height=roi_height, anchor=anchor)

        self.assertEqual((roi_width, roi_height), (55, 48))
        self.assertEqual(anchor, (42, 40))
        self.assertEqual(top_crop, roi_prep.PixelBox(15, 16, 70, 64))
        self.assertEqual(bottom_crop, roi_prep.PixelBox(15, 16, 70, 64))
        self.assertEqual(empty_crop, roi_prep.PixelBox(15, 16, 70, 64))

    def test_recognize_sample_text_orders_same_line_boxes_left_to_right(self):
        class RecognizerStub:
            def predict(self, path):
                with Image.open(path) as image:
                    red, green, blue = image.convert("RGB").getpixel((0, 0))
                return {"rec_text": "左" if red > blue else "右"}

        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "sample.jpg"
            image = Image.new("RGB", (100, 40), (0, 0, 0))
            for x in range(10, 30):
                for y in range(12, 24):
                    image.putpixel((x, y), (255, 0, 0))
            for x in range(60, 80):
                for y in range(10, 22):
                    image.putpixel((x, y), (0, 0, 255))
            image.save(image_path)

            sample = roi_prep.SourceSample(
                stem="sample",
                image_path=image_path,
                label_path=Path(tmpdir) / "sample.txt",
                boxes=[],
                width=100,
                height=40,
                annotation={},
            )
            text = roi_prep.recognize_sample_text(
                RecognizerStub(),
                sample,
                [
                    roi_prep.PixelBox(60, 10, 80, 22),
                    roi_prep.PixelBox(10, 12, 30, 24),
                ],
            )

        self.assertEqual(text, "左右")

    def test_prepare_outputs_roi_ocr_without_segment_markers(self):
        class RecognizerStub:
            def predict(self, path):
                return {"rec_text": "字幕"}

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source"
            output = root / "roi"
            (source / "images").mkdir(parents=True)
            (source / "labels").mkdir()
            Image.new("RGB", (80, 40), (20, 40, 90)).save(source / "images" / "sample.jpg")
            (source / "labels" / "sample.txt").write_text(
                "0 0.500000 0.500000 0.500000 0.500000\n",
                encoding="utf-8",
            )

            original_create = roi_prep.create_text_recognizer
            roi_prep.create_text_recognizer = lambda _model_name: RecognizerStub()
            try:
                status = roi_prep.prepare_roi_samples(
                    Namespace(
                        samples_dir=source,
                        output=output,
                        keep_empty=False,
                        copy_labels=False,
                        include_dropped_images=False,
                    )
                )
            finally:
                roi_prep.create_text_recognizer = original_create

            row = json.loads((output / "annotations.jsonl").read_text(encoding="utf-8").strip())
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))

        self.assertEqual(status, 0)
        self.assertNotIn("segment_marker", row)
        self.assertNotIn("segment_marker_method", row)
        self.assertEqual(row["ocr_text"], "字幕")
        self.assertEqual(row["subtitle_presence_method"], "source_label_box_ocr")
        self.assertEqual(summary["subtitle_presence_method"], "source_label_box_ocr")
        self.assertNotIn("segment_marker_method", summary)


if __name__ == "__main__":
    unittest.main()
