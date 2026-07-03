import json
import tempfile
import unittest
from pathlib import Path

from src.config import TrainSettings
from src.dataset import SubtitleDataset, apply_label_masks
from src.train import format_epoch_summary, parse_args, resolve_resume_checkpoint, run_training


class TrainingSmokeTests(unittest.TestCase):
    def test_dataset_loads_project_data(self):
        dataset = SubtitleDataset([Path("data/generated_samples1")], image_size=128, max_samples=4)
        self.assertEqual(len(dataset), 4)
        item = dataset[0]
        self.assertEqual(item["image"].shape[0], 3)
        self.assertEqual(item["region"].shape[0], 1)
        self.assertEqual(item["training_mask"].shape, item["region"].shape)

    def test_label_masks_can_drop_image_and_add_box(self):
        boxes = []
        kept, ignore_regions, drop = apply_label_masks(
            "sample",
            boxes,
            {
                "sample": {
                    "__image__": {"drop_image": True},
                    "__add_1": {"add_bbox": [10, 20, 30, 40]},
                }
            },
            100,
            80,
        )

        self.assertTrue(drop)
        self.assertEqual(ignore_regions, [])
        self.assertEqual(len(kept), 1)
        self.assertEqual((kept[0].x1, kept[0].y1, kept[0].x2, kept[0].y2), (10, 20, 30, 40))

    def test_dataset_skips_dropped_images_and_counts_added_boxes_as_labeled(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "images").mkdir()
            (root / "labels").mkdir()

            from PIL import Image

            Image.new("RGB", (64, 48), "black").save(root / "images" / "drop.jpg")
            Image.new("RGB", (64, 48), "black").save(root / "images" / "added.jpg")
            (root / "labels" / "drop.txt").write_text("0 0.5 0.5 0.2 0.2\n")
            (root / "labels" / "added.txt").write_text("")
            (root / "label_masks.json").write_text(
                json.dumps(
                    {
                        "version": 1,
                        "items": {
                            "drop": {"__image__": {"drop_image": True}},
                            "added": {"__add_1": {"add_bbox": [8, 10, 40, 30]}},
                        },
                    }
                )
            )

            dataset = SubtitleDataset([root], image_size=64)

            self.assertEqual([sample.sample_id for sample in dataset.samples], ["added"])
            self.assertEqual(dataset.summary.labeled, 1)
            self.assertEqual(dataset.summary.empty, 0)

    def test_training_settings_define_explicit_sample_roots(self):
        settings = TrainSettings(
            train_roots=[Path("data/generated_samples1"), Path("data/generated_samples2")],
            val_root=Path("data/validation_samples"),
        )
        self.assertEqual(settings.train_roots, [Path("data/generated_samples1"), Path("data/generated_samples2")])
        self.assertEqual(settings.val_root, Path("data/validation_samples"))

    def test_parse_args_accepts_multiple_train_roots(self):
        settings = parse_args(
            [
                "--train-root",
                "data/generated_samples1",
                "--train-root",
                "data/generated_samples2",
                "--val-root",
                "data/validation_samples",
            ]
        )
        self.assertEqual(settings.train_roots, [Path("data/generated_samples1"), Path("data/generated_samples2")])
        self.assertEqual(settings.val_root, Path("data/validation_samples"))

    def test_parse_args_accepts_validation_thresholds(self):
        settings = parse_args(
            [
                "--region-threshold",
                "0.95",
                "--kernel-threshold",
                "0.9",
                "--iou-threshold",
                "0.6",
                "--max-detection-width-ratio",
                "0.8",
            ]
        )

        self.assertEqual(settings.region_threshold, 0.95)
        self.assertEqual(settings.kernel_threshold, 0.9)
        self.assertEqual(settings.iou_threshold, 0.6)
        self.assertEqual(settings.max_detection_width_ratio, 0.8)

    def test_parse_args_accepts_resume_path(self):
        settings = parse_args(["--resume", "outputs/pytorch_run/epoch_outputs/epoch_0003"])

        self.assertEqual(settings.resume, Path("outputs/pytorch_run/epoch_outputs/epoch_0003"))

    def test_resolve_resume_checkpoint_uses_latest_epoch_checkpoint_from_output_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            old_checkpoint = output_dir / "epoch_outputs" / "epoch_0001" / "model.pt"
            latest_checkpoint = output_dir / "epoch_outputs" / "epoch_0003" / "model.pt"
            old_checkpoint.parent.mkdir(parents=True)
            latest_checkpoint.parent.mkdir(parents=True)
            old_checkpoint.write_bytes(b"old")
            latest_checkpoint.write_bytes(b"latest")

            self.assertEqual(resolve_resume_checkpoint(output_dir), latest_checkpoint)

    def test_one_epoch_training_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            metrics = run_training(
                TrainSettings(
                    output_dir=Path(tmp),
                    image_size=96,
                    batch_size=2,
                    epochs=1,
                    max_train_samples=4,
                    max_val_samples=2,
                    device="cpu",
                )
            )
            self.assertIn("train_loss", metrics)
            self.assertIn("f1", metrics)
            self.assertTrue((Path(tmp) / "best.pt").exists())

    def test_training_resumes_from_epoch_checkpoint_and_appends_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            first_metrics = run_training(
                TrainSettings(
                    output_dir=output_dir,
                    image_size=96,
                    batch_size=2,
                    epochs=1,
                    max_train_samples=4,
                    max_val_samples=2,
                    save_epoch_outputs=False,
                    device="cpu",
                )
            )
            checkpoint_dir = output_dir / "epoch_outputs" / "epoch_0001"
            resumed_metrics = run_training(
                TrainSettings(
                    output_dir=output_dir,
                    image_size=96,
                    batch_size=2,
                    epochs=2,
                    max_train_samples=4,
                    max_val_samples=2,
                    save_epoch_outputs=False,
                    resume=checkpoint_dir,
                    device="cpu",
                )
            )

            self.assertEqual(resumed_metrics["epoch"], 2.0)
            self.assertGreater(resumed_metrics["step"], first_metrics["step"])
            validation_epochs = [
                json.loads(line)["epoch"]
                for line in (output_dir / "metrics.jsonl").read_text().splitlines()
                if json.loads(line).get("record_type") == "validation"
            ]
            self.assertEqual(validation_epochs, [1.0, 2.0])

    def test_format_epoch_summary_includes_train_and_validation_metrics(self):
        text = format_epoch_summary(
            epoch=1,
            total_epochs=3,
            metrics={
                "train_loss": 0.123456,
                "train_region_bce": 0.01,
                "train_kernel_bce": 0.02,
                "train_region_dice": 0.03,
                "train_kernel_dice": 0.04,
                "val_loss": 0.234567,
                "precision": 0.9,
                "recall": 0.8,
                "f1": 0.847059,
                "true_positive": 17.0,
                "false_positive": 2.0,
                "false_negative": 3.0,
            },
        )
        self.assertEqual(
            text,
            "\n".join(
                [
                    "epoch 1/3",
                    "  loss: train=0.1235 val=0.2346",
                    "  train_parts: region_bce=0.0100 kernel_bce=0.0200 region_dice=0.0300 kernel_dice=0.0400",
                    "  validation: precision=0.9000 recall=0.8000 f1=0.8471 tp=17 fp=2 fn=3",
                ]
            ),
        )


if __name__ == "__main__":
    unittest.main()
