import json
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from src.roi_config import RoiTrainSettings
from src.roi_dataset import RoiPresenceEmbeddingDataset, collate_roi_batch
from src.roi_loss import roi_presence_embedding_loss
from src.roi_pairs import select_embedding_pairs
from src.roi_model import RoiPresenceEmbeddingModel
from src.train_roi import parse_args, run_training


def write_roi_dataset(root: Path, *, size: tuple[int, int] = (32, 16)) -> None:
    (root / "images").mkdir(parents=True)
    annotations = []
    rows = [
        ("a0", True, "seg-a", 0, "same subtitle"),
        ("a1", True, "seg-a", 30, "same subtitle"),
        ("b0", True, "seg-b", 60, "different subtitle"),
        ("e0", False, "empty", 90, ""),
    ]
    for name, has_subtitle, segment, frame_index, ocr_text in rows:
        Image.new("RGB", size, "white" if has_subtitle else "black").save(root / "images" / f"{name}.jpg")
        annotations.append(
            {
                "image": f"images/{name}.jpg",
                "source_annotation": {"source_video": "video-1", "frame_index": frame_index},
                "image_width": size[0],
                "image_height": size[1],
                "roi_size": [size[0], size[1]],
                "has_subtitle": has_subtitle,
                "segment_marker": segment,
                "ocr_text": ocr_text,
                "ocr_text_normalized": ocr_text,
            }
        )
    (root / "annotations.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in annotations),
        encoding="utf-8",
    )
    (root / "summary.json").write_text(
        json.dumps({"version": 1, "roi_size": list(size), "samples": 4, "positive": 3, "empty": 1}),
        encoding="utf-8",
    )


class RoiPresenceEmbeddingTests(unittest.TestCase):
    def test_embedding_pair_selection_uses_only_local_subtitle_pairs(self):
        selection = select_embedding_pairs(
            presence=torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0]),
            segment_ids=["seg-a", "seg-a", "seg-b", "seg-c", "empty", "seg-d", "seg-a"],
            roots=["root-a", "root-a", "root-a", "root-b", "root-a", "root-a", "root-a"],
            video_ids=["video-1", "video-1", "video-1", "video-1", "video-1", "video-2", "video-1"],
            frame_indices=[100, 130, 160, 160, 190, 190, 400],
            ocr_texts=["alpha", "alpha", "bravo", "charlie", "", "delta", "alpha"],
            frame_window=30,
            ocr_negative_enabled=False,
            ocr_negative_max_similarity=0.2,
        )

        self.assertEqual([(pair.i, pair.j, pair.same) for pair in selection.pairs], [(0, 1, True), (1, 2, False)])
        self.assertEqual(selection.local_positive_pairs, 1)
        self.assertEqual(selection.local_negative_pairs, 1)
        self.assertEqual(selection.ocr_negative_pairs, 0)
        self.assertGreater(selection.skipped_pairs, 0)

    def test_embedding_pair_selection_adds_only_strong_ocr_negative_pairs(self):
        selection = select_embedding_pairs(
            presence=torch.ones(6),
            segment_ids=["seg-a", "seg-b", "seg-b", "seg-d", "seg-a", "seg-e"],
            roots=["root-a", "root-a", "root-a", "root-a", "root-a", "root-b"],
            video_ids=["video-1"] * 6,
            frame_indices=[0, 300, 600, 900, 1200, 1500],
            ocr_texts=["今晚吃饭", "abcXYZ987", "今晚吃飯", "", "xyz", "abcXYZ987"],
            frame_window=30,
            ocr_negative_enabled=True,
            ocr_negative_max_similarity=0.2,
        )

        self.assertEqual([(pair.i, pair.j, pair.same) for pair in selection.pairs], [(0, 1, False)])
        self.assertEqual(selection.local_positive_pairs, 0)
        self.assertEqual(selection.local_negative_pairs, 0)
        self.assertEqual(selection.ocr_negative_pairs, 1)

    def test_dataset_reads_presence_and_segment_supervision(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_roi_dataset(root)

            dataset = RoiPresenceEmbeddingDataset([root])
            batch = collate_roi_batch([dataset[0], dataset[3]])

            self.assertEqual(dataset.summary.roi_size, (32, 16))
            self.assertEqual(dataset.summary.positive, 3)
            self.assertEqual(batch.images.shape, (2, 3, 16, 32))
            self.assertEqual(batch.presence.tolist(), [1.0, 0.0])
            self.assertEqual(batch.segment_ids, ["seg-a", "empty"])

    def test_dataset_requires_explicit_resize_for_mismatched_roots(self):
        with tempfile.TemporaryDirectory() as tmp:
            root_a = Path(tmp) / "a"
            root_b = Path(tmp) / "b"
            write_roi_dataset(root_a, size=(32, 16))
            write_roi_dataset(root_b, size=(40, 20))

            with self.assertRaises(ValueError):
                RoiPresenceEmbeddingDataset([root_a, root_b])

            dataset = RoiPresenceEmbeddingDataset([root_a, root_b], resize_roi=(32, 16))
            self.assertEqual(dataset.summary.roi_size, (32, 16))

    def test_validation_limit_keeps_same_segment_pairs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_roi_dataset(root)

            dataset = RoiPresenceEmbeddingDataset(
                [root],
                max_samples=3,
                empty_ratio=1 / 3,
                segment_aware_limit=True,
            )

            self.assertEqual(dataset.summary.total, 3)
            self.assertEqual(dataset.summary.positive, 2)
            self.assertEqual(dataset.summary.empty, 1)
            self.assertEqual(dataset.summary.same_segment_pairs, 1)
            self.assertEqual([sample.segment_id for sample in dataset.samples if sample.has_subtitle], ["seg-a", "seg-a"])

    def test_model_returns_normalized_embedding(self):
        model = RoiPresenceEmbeddingModel(width=8, embedding_dim=128)
        presence_logit, embedding = model(torch.randn(2, 3, 16, 32))

        self.assertEqual(presence_logit.shape, (2,))
        self.assertEqual(embedding.shape, (2, 128))
        self.assertTrue(torch.allclose(embedding.norm(dim=1), torch.ones(2), atol=1e-5))

    def test_hybrid_lite_model_returns_normalized_embedding(self):
        model = RoiPresenceEmbeddingModel(
            width=8,
            embedding_dim=128,
            embedding_head_type="hybrid_lite",
            embedding_sequence_channels=16,
        )
        presence_logit, embedding = model(torch.randn(2, 3, 16, 32))

        self.assertEqual(presence_logit.shape, (2,))
        self.assertEqual(embedding.shape, (2, 128))
        self.assertTrue(torch.allclose(embedding.norm(dim=1), torch.ones(2), atol=1e-5))

    def test_embedding_loss_skips_without_two_positive_samples(self):
        presence_logit = torch.zeros(2, requires_grad=True)
        embedding = torch.nn.functional.normalize(torch.randn(2, 128, requires_grad=True), dim=1)
        loss = roi_presence_embedding_loss(
            presence_logit,
            embedding,
            torch.tensor([1.0, 0.0]),
            ["seg-a", "empty"],
            roots=["root-a", "root-a"],
            video_ids=["video-1", "video-1"],
            frame_indices=[0, 30],
            ocr_texts=["subtitle", ""],
            embedding_loss_weight=1.0,
            embedding_pair_frame_window=30,
            embedding_ocr_negative_enabled=True,
            embedding_ocr_negative_max_similarity=0.2,
            embedding_temperature=0.1,
        )

        self.assertEqual(loss.embedding_pairs, 0)
        self.assertEqual(float(loss.embedding_loss.detach()), 0.0)
        loss.total.backward()
        self.assertIsNotNone(presence_logit.grad)

    def test_parse_args_uses_roi_training_names(self):
        settings = parse_args(
            [
                "--train-root",
                "data/roi_samples1",
                "--val-root",
                "data/roi_samples6",
                "--resize-roi",
                "128x32",
                "--embedding-loss-weight",
                "0.5",
                "--embedding-head",
                "hybrid_lite",
                "--embedding-sequence-channels",
                "16",
            ]
        )

        self.assertEqual(settings.train_roots, [Path("data/roi_samples1")])
        self.assertEqual(settings.val_root, Path("data/roi_samples6"))
        self.assertEqual(settings.resize_roi, (128, 32))
        self.assertEqual(settings.embedding_loss_weight, 0.5)
        self.assertEqual(settings.embedding_head_type, "hybrid_lite")
        self.assertEqual(settings.embedding_sequence_channels, 16)

    def test_parse_args_accepts_positive_and_negative_ratio_only(self):
        positive_settings = parse_args(["--positive-ratio", "0.7"])
        negative_settings = parse_args(["--negative-ratio", "0.2"])

        self.assertEqual(positive_settings.negative_ratio, 0.30000000000000004)
        self.assertEqual(negative_settings.negative_ratio, 0.2)
        with self.assertRaises(SystemExit):
            parse_args(["--train-empty-sample-ratio", "0.35"])

    def test_parse_args_accepts_validation_ratio(self):
        settings = parse_args(["--val-positive-ratio", "0.6"])

        self.assertEqual(settings.val_negative_ratio, 0.4)

    def test_one_epoch_roi_training_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "roi"
            output = Path(tmp) / "out"
            write_roi_dataset(root)

            metrics = run_training(
                RoiTrainSettings(
                    train_roots=[root],
                    val_root=root,
                    output_dir=output,
                    batch_size=2,
                    epochs=1,
                    max_train_samples=4,
                    max_val_samples=4,
                    width=8,
                    device="cpu",
                )
            )

            self.assertIn("presence_f1", metrics)
            self.assertIn("embedding_pair_accuracy", metrics)
            self.assertTrue((output / "best.pt").exists())


if __name__ == "__main__":
    unittest.main()
