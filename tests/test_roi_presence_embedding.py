import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image

from src.roi_config import RoiTrainSettings
from src.roi_dataset import RoiPresenceEmbeddingDataset, RoiSample, collate_roi_batch
from src.roi_loss import (
    EmbeddingPairMemory,
    balance_embedding_pairs,
    embedding_margin_loss,
    roi_presence_embedding_loss,
    supervised_contrastive_embedding_loss,
)
from src.roi_metrics import embedding_metrics
from src.roi_pairs import select_embedding_pairs
from src.roi_model import LocalContrastEmbeddingHead, LocalContrastResidual, LocalTextnessPresenceHead, RoiPresenceEmbeddingModel
from src.train_roi import (
    configure_training_phase,
    format_epoch_summary,
    parse_args,
    run_training,
    should_save_roi_best_checkpoint,
    short_positive_mask_loss,
    training_summary,
    training_phases,
    validation_overlaps_training,
)
from src.roi_sampler import RoiBalancedBatchSampler


def write_roi_dataset(root: Path, *, size: tuple[int, int] = (32, 16)) -> None:
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir()
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
                "source_roi": [10, 20, 10 + size[0], 20 + size[1]],
                "source_subtitle_boxes": [[10, 20, 10 + size[0], 20 + size[1]]] if has_subtitle else [],
                "has_subtitle": has_subtitle,
                "segment_marker": segment,
                "ocr_text": ocr_text,
                "ocr_text_normalized": ocr_text,
            }
        )
        label_text = "0 0.250000 0.437500 0.250000 0.375000\n" if has_subtitle else ""
        (root / "labels" / f"{name}.txt").write_text(label_text, encoding="utf-8")
    (root / "annotations.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in annotations),
        encoding="utf-8",
    )
    (root / "summary.json").write_text(
        json.dumps({"version": 1, "roi_size": list(size), "samples": 4, "positive": 3, "empty": 1}),
        encoding="utf-8",
    )


def make_sampler_samples(
    *, positives: int, negatives: int, repeated_segments: bool = True
) -> list[RoiSample]:
    samples: list[RoiSample] = []
    for index in range(positives):
        segment_number = index // 2 if repeated_segments else index
        samples.append(
            RoiSample(
                image_path=Path(f"positive-{index}.jpg"),
                label_path=Path(f"positive-{index}.txt"),
                sample_id=f"positive-{index}",
                root=Path("root"),
                has_subtitle=True,
                segment_id=f"segment-{segment_number}",
                video_id="video",
                frame_index=segment_number * 300 + index % 2 * 30,
                ocr_text=f"subtitle-{segment_number}",
                annotation={},
            )
        )
    for index in range(negatives):
        samples.append(
            RoiSample(
                image_path=Path(f"negative-{index}.jpg"),
                label_path=Path(f"negative-{index}.txt"),
                sample_id=f"negative-{index}",
                root=Path("root"),
                has_subtitle=False,
                segment_id=f"empty-{index}",
                video_id="video",
                frame_index=10000 + index * 30,
                ocr_text="",
                annotation={},
            )
        )
    return samples


class RoiPresenceEmbeddingTests(unittest.TestCase):
    def test_balanced_batch_sampler_realizes_presence_ratio_and_covers_every_sample(self):
        samples = make_sampler_samples(positives=6, negatives=10)
        sampler = RoiBalancedBatchSampler(samples, batch_size=4, negative_ratio=0.5, seed=7)
        batches = list(sampler)

        self.assertTrue(all(sum(not samples[index].has_subtitle for index in batch) == 2 for batch in batches))
        self.assertEqual({index for batch in batches for index in batch}, set(range(len(samples))))

    def test_balanced_batch_sampler_is_reproducible_and_changes_by_epoch(self):
        samples = make_sampler_samples(positives=8, negatives=8)
        left = RoiBalancedBatchSampler(samples, batch_size=4, negative_ratio=0.5, seed=11)
        right = RoiBalancedBatchSampler(samples, batch_size=4, negative_ratio=0.5, seed=11)

        self.assertEqual(list(left), list(right))
        left.set_epoch(1)
        self.assertNotEqual(list(left), list(right))

    def test_balanced_batch_sampler_places_valid_same_segment_pair_in_every_batch(self):
        samples = make_sampler_samples(positives=8, negatives=8, repeated_segments=True)
        sampler = RoiBalancedBatchSampler(samples, batch_size=4, negative_ratio=0.5, seed=13)
        for batch in sampler:
            positives = [samples[index] for index in batch if samples[index].has_subtitle]
            self.assertTrue(
                any(
                    left.root == right.root
                    and left.segment_id == right.segment_id
                    for offset, left in enumerate(positives)
                    for right in positives[offset + 1 :]
                )
            )

    def test_balance_embedding_pairs_keeps_all_positives_and_hardest_negatives(self):
        similarities = torch.tensor([0.8, 0.7, 0.9, 0.6, 0.2, -0.1])
        targets = torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0, 0.0])

        selection = balance_embedding_pairs(similarities, targets, negative_ratio=0.5)

        self.assertEqual(selection.indices.tolist(), [0, 1, 2, 3])
        self.assertEqual(selection.candidate_positive_pairs, 2)
        self.assertEqual(selection.candidate_negative_pairs, 4)
        self.assertEqual(selection.selected_positive_pairs, 2)
        self.assertEqual(selection.selected_negative_pairs, 2)

    def test_balance_embedding_pairs_bounds_negative_only_batch(self):
        similarities = torch.tensor([0.1, 0.9, 0.4])
        targets = torch.zeros(3)

        selection = balance_embedding_pairs(similarities, targets, negative_ratio=0.5)

        self.assertEqual(selection.indices.tolist(), [1])

    def test_embedding_margin_loss_enforces_tolerance_around_fixed_threshold(self):
        similarities = torch.tensor([0.80, 0.20, 0.60, 0.40])
        targets = torch.tensor([1.0, 0.0, 1.0, 0.0])

        loss = embedding_margin_loss(similarities, targets)

        self.assertAlmostEqual(float(loss), 0.9009242, places=6)

    def test_embedding_margin_loss_balances_positive_and_negative_pairs(self):
        similarities = torch.tensor([0.50] + [0.0] * 100)
        targets = torch.tensor([1.0] + [0.0] * 100)

        loss = embedding_margin_loss(similarities, targets)

        self.assertAlmostEqual(float(loss), 0.4, places=6)

    def test_embedding_margin_loss_keeps_hard_negative_from_easy_pair_dilution(self):
        similarities = torch.tensor([0.50] + [0.0] * 99)
        targets = torch.zeros(100)

        loss = embedding_margin_loss(similarities, targets)

        self.assertAlmostEqual(float(loss), 0.8, places=6)

    def test_supervised_contrastive_embedding_loss_uses_same_segment_positives(self):
        embedding = torch.nn.functional.normalize(
            torch.tensor(
                [
                    [1.0, 0.0],
                    [0.8, 0.2],
                    [0.0, 1.0],
                ]
            ),
            dim=1,
        )

        loss = supervised_contrastive_embedding_loss(
            embedding,
            torch.ones(3),
            ["seg-a", "seg-a", "seg-b"],
            ["root", "root", "root"],
            temperature=0.1,
        )

        self.assertGreater(float(loss), 0.0)

    def test_embedding_metrics_calibrates_best_threshold_without_strict_zip_error(self):
        embedding = torch.nn.functional.normalize(torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.8, 0.6]]), dim=1)

        metrics = embedding_metrics(
            embedding,
            torch.ones(3),
            ["seg-a", "seg-b", "seg-a"],
            roots=["root", "root", "root"],
            video_ids=["video", "video", "video"],
            ocr_texts=["alpha", "bravo", "alpha"],
            adjacent_segment_ids=[frozenset({"seg-b"}), frozenset({"seg-a"}), frozenset({"seg-b"})],
            ocr_negative_enabled=False,
            ocr_negative_max_similarity=0.2,
            ocr_negative_ratio=0.3,
            threshold=0.5,
        )

        self.assertAlmostEqual(metrics["embedding_best_f1"], 1.0)

    def test_positive_embedding_memory_connects_same_root_segment_across_batches(self):
        memory = EmbeddingPairMemory(ocr_negative_max_similarity=0.2, ocr_negative_ratio=0.3)
        first = torch.tensor([[1.0, 0.0]], requires_grad=True)
        second = torch.tensor([[0.0, 1.0]], requires_grad=True)

        first_loss, first_pairs = memory.loss_and_update(
            first, torch.ones(1), ["segment-a"], ["root-a"], ["video-a"], ["same text"], [frozenset()]
        )
        second_loss, second_pairs = memory.loss_and_update(
            second, torch.ones(1), ["segment-a"], ["root-a"], ["video-b"], ["same text"], [frozenset()]
        )

        self.assertEqual(first_pairs, 0)
        self.assertEqual(float(first_loss.detach()), 0.0)
        self.assertEqual(second_pairs, 1)
        self.assertAlmostEqual(float(second_loss.detach()), 0.9, places=6)

    def test_embedding_pair_memory_connects_local_negative_across_batches(self):
        memory = EmbeddingPairMemory(ocr_negative_max_similarity=0.2, ocr_negative_ratio=0.3)
        first = torch.tensor([[1.0, 0.0]])
        second = torch.tensor([[1.0, 0.0]], requires_grad=True)

        memory.loss_and_update(
            first,
            torch.ones(1),
            ["segment-a"],
            ["root-a"],
            ["video-a"],
            ["alpha text"],
            [frozenset({"segment-b"})],
        )
        loss, pairs = memory.loss_and_update(
            second,
            torch.ones(1),
            ["segment-b"],
            ["root-a"],
            ["video-a"],
            ["bravo text"],
            [frozenset({"segment-a"})],
        )

        self.assertEqual(pairs, 1)
        self.assertAlmostEqual(float(loss.detach()), 1.8, places=6)

    def test_embedding_pair_memory_connects_ocr_negative_across_batches(self):
        memory = EmbeddingPairMemory(ocr_negative_max_similarity=0.2, ocr_negative_ratio=0.3)
        first = torch.tensor([[1.0, 0.0]])
        second = torch.tensor([[1.0, 0.0]], requires_grad=True)

        memory.loss_and_update(first, torch.ones(1), ["segment-a"], ["root-a"], ["video-a"], ["alpha text"], [frozenset()])
        loss, pairs = memory.loss_and_update(
            second, torch.ones(1), ["segment-b"], ["root-a"], ["video-b"], ["9876 xyz"], [frozenset()]
        )

        self.assertEqual(pairs, 1)
        self.assertAlmostEqual(float(loss.detach()), 1.8, places=6)

    def test_embedding_pair_memory_limits_ocr_negatives_by_ratio(self):
        memory = EmbeddingPairMemory(ocr_negative_max_similarity=0.2, ocr_negative_ratio=0.0)
        first = torch.tensor([[1.0, 0.0]])
        second = torch.tensor([[1.0, 0.0]], requires_grad=True)

        memory.loss_and_update(first, torch.ones(1), ["segment-a"], ["root-a"], ["video-a"], ["alpha text"], [frozenset()])
        loss, pairs = memory.loss_and_update(
            second, torch.ones(1), ["segment-b"], ["root-a"], ["video-b"], ["9876 xyz"], [frozenset()]
        )

        self.assertEqual(pairs, 0)
        self.assertEqual(float(loss.detach()), 0.0)

    def test_local_contrast_residual_removes_constant_background(self):
        operator = LocalContrastResidual(kernel_size=3)
        subtitle = torch.tensor(
            [[[[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, -2.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]]]]
        )

        dark_background = operator(subtitle + 3.0)
        bright_background = operator(subtitle + 30.0)

        self.assertTrue(torch.allclose(dark_background, bright_background, atol=1e-4))

    def test_embedding_pair_selection_uses_only_local_subtitle_pairs(self):
        selection = select_embedding_pairs(
            presence=torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0]),
            segment_ids=["seg-a", "seg-a", "seg-b", "seg-c", "empty", "seg-d", "seg-a"],
            roots=["root-a", "root-a", "root-a", "root-b", "root-a", "root-a", "root-a"],
            video_ids=["video-1", "video-1", "video-1", "video-1", "video-1", "video-2", "video-1"],
            ocr_texts=["alpha", "alpha", "bravo", "charlie", "", "delta", "alpha"],
            adjacent_segment_ids=[
                frozenset({"seg-b"}),
                frozenset({"seg-b"}),
                frozenset({"seg-a"}),
                frozenset(),
                frozenset(),
                frozenset(),
                frozenset({"seg-b"}),
            ],
            ocr_negative_enabled=False,
            ocr_negative_max_similarity=0.2,
            ocr_negative_ratio=0.3,
        )

        self.assertEqual(
            [(pair.i, pair.j, pair.same) for pair in selection.pairs],
            [(0, 1, True), (0, 2, False), (0, 6, True), (1, 2, False), (1, 6, True), (2, 6, False)],
        )
        self.assertEqual(selection.local_positive_pairs, 3)
        self.assertEqual(selection.local_negative_pairs, 3)
        self.assertEqual(selection.ocr_negative_pairs, 0)
        self.assertGreater(selection.skipped_pairs, 0)

    def test_embedding_pair_selection_adds_only_strong_ocr_negative_pairs(self):
        selection = select_embedding_pairs(
            presence=torch.ones(6),
            segment_ids=["seg-a", "seg-b", "seg-b", "seg-d", "seg-a", "seg-e"],
            roots=["root-a", "root-a", "root-a", "root-a", "root-a", "root-b"],
            video_ids=["video-1"] * 6,
            ocr_texts=["今晚吃饭", "abcXYZ987", "今晚吃飯", "", "xyz", "abcXYZ987"],
            adjacent_segment_ids=[frozenset() for _ in range(6)],
            ocr_negative_enabled=True,
            ocr_negative_max_similarity=0.2,
            ocr_negative_ratio=0.3,
        )

        self.assertEqual([(pair.i, pair.j, pair.same) for pair in selection.pairs], [(0, 4, True), (1, 2, True), (0, 1, False)])
        self.assertEqual(selection.local_positive_pairs, 2)
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
            self.assertEqual(batch.subtitle_masks.shape, (2, 1, 16, 32))
            self.assertEqual(batch.presence.tolist(), [1.0, 0.0])
            self.assertEqual(batch.segment_ids, ["seg-a", "empty"])
            self.assertEqual(float(batch.subtitle_masks[0].sum()), 48.0)
            self.assertEqual(float(batch.subtitle_masks[1].sum()), 0.0)

    def test_dataset_maps_current_roi_labels_after_resize(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_roi_dataset(root, size=(32, 16))

            dataset = RoiPresenceEmbeddingDataset([root], resize_roi=(16, 8))
            item = dataset[0]

            mask = item["subtitle_mask"]
            self.assertEqual(mask.shape, (1, 8, 16))
            self.assertEqual(float(mask.sum()), 12.0)
            self.assertTrue(bool(mask[0, 2:5, 2:6].all()))

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

    def test_presence_head_uses_top_local_textness_responses(self):
        head = LocalTextnessPresenceHead(feature_dim=1, hidden_dim=1, topk_ratio=0.25)
        head.local[1] = torch.nn.ReLU()
        with torch.no_grad():
            head.local[0].weight.zero_()
            head.local[0].bias.zero_()
            head.local[0].weight[:, :, 1, 1] = 1.0
            head.textness.weight.fill_(1.0)
            head.textness.bias.zero_()
        feature_map = torch.tensor([[[[8.0, 0.0, 0.0, 0.0], [4.0, 0.0, 0.0, 0.0]]]])

        logit = head(feature_map)

        self.assertEqual(logit.shape, (1,))
        self.assertAlmostEqual(float(logit.item()), 6.0)

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

    def test_local_contrast_model_returns_normalized_embedding(self):
        model = RoiPresenceEmbeddingModel(
            width=8,
            embedding_dim=128,
            embedding_head_type="local_contrast",
            embedding_sequence_channels=16,
        )

        presence_logit, embedding = model(torch.randn(2, 3, 16, 32))

        self.assertEqual(presence_logit.shape, (2,))
        self.assertEqual(embedding.shape, (2, 128))
        self.assertTrue(torch.allclose(embedding.norm(dim=1), torch.ones(2), atol=1e-5))

    def test_local_contrast_fusion_preserves_gap_embedding_at_initialization(self):
        head = LocalContrastEmbeddingHead(feature_dim=16, embedding_dim=32, sequence_channels=8)

        self.assertEqual(head.sequence[0].in_channels, 16)
        self.assertTrue(torch.equal(head.fusion.weight[:, :32], torch.eye(32)))
        self.assertTrue(torch.equal(head.fusion.weight[:, 32:], torch.zeros(32, 64)))
        self.assertTrue(torch.equal(head.fusion.bias, torch.zeros(32)))

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
            ocr_texts=["subtitle", ""],
            embedding_loss_weight=1.0,
            adjacent_segment_ids=[frozenset(), frozenset()],
            embedding_ocr_negative_enabled=True,
            embedding_ocr_negative_max_similarity=0.2,
            embedding_ocr_negative_ratio=0.3,
            embedding_temperature=0.1,
        )

        self.assertEqual(loss.embedding_pairs, 0)
        self.assertEqual(float(loss.embedding_loss.detach()), 0.0)
        loss.total.backward()
        self.assertIsNotNone(presence_logit.grad)

    def test_presence_loss_accepts_sample_weights(self):
        presence_logit = torch.zeros(3, requires_grad=True)
        embedding = torch.nn.functional.normalize(torch.randn(3, 128, requires_grad=True), dim=1)
        loss = roi_presence_embedding_loss(
            presence_logit,
            embedding,
            torch.tensor([1.0, 1.0, 0.0]),
            ["seg-a", "seg-b", "empty"],
            presence_loss_weights=torch.tensor([3.0, 1.0, 1.0]),
            roots=["root-a", "root-a", "root-a"],
            video_ids=["video-1", "video-1", "video-1"],
            ocr_texts=["好", "subtitle", ""],
            embedding_loss_weight=0.0,
            adjacent_segment_ids=[frozenset({"seg-b"}), frozenset({"seg-a"}), frozenset()],
            embedding_ocr_negative_enabled=True,
            embedding_ocr_negative_max_similarity=0.2,
            embedding_ocr_negative_ratio=0.3,
            embedding_temperature=0.1,
        )

        self.assertAlmostEqual(float(loss.presence_loss.detach()), float(torch.log(torch.tensor(2.0)) * 5.0 / 3.0))

    def test_short_positive_mask_loss_targets_only_short_positive_masks(self):
        textness = torch.zeros(3, 1, 2, 4, requires_grad=True)
        masks = torch.zeros(3, 1, 8, 16)
        masks[0, :, 2:6, 4:12] = 1.0
        masks[2, :, 2:6, 4:12] = 1.0
        presence = torch.tensor([1.0, 1.0, 1.0])

        loss = short_positive_mask_loss(
            textness,
            masks,
            presence,
            ["好", "long subtitle", "无"],
            weight=2.0,
        )

        self.assertAlmostEqual(float(loss.detach()), float(torch.log(torch.tensor(2.0)) * 2.0))

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
                "--short-positive-loss-weight",
                "2.5",
                "--short-positive-mask-loss-weight",
                "1.25",
                "--presence-topk-ratio",
                "0.02",
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
        self.assertEqual(settings.short_positive_loss_weight, 2.5)
        self.assertEqual(settings.short_positive_mask_loss_weight, 1.25)
        self.assertEqual(settings.presence_topk_ratio, 0.02)
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

    def test_parse_args_accepts_embedding_negative_ratio(self):
        settings = parse_args(["--negative-ratio", "0.5", "--embedding-negative-ratio", "0.4"])

        self.assertEqual(settings.negative_ratio, 0.5)
        self.assertEqual(settings.embedding_negative_ratio, 0.4)

    def test_parse_args_accepts_three_stage_schedule(self):
        settings = parse_args(
            ["--presence-epochs", "2", "--embedding-epochs", "3", "--joint-epochs", "4", "--joint-lr", "0.00002"]
        )

        self.assertEqual([phase.name for phase in training_phases(settings)], ["presence", "embedding", "joint"])
        self.assertEqual([phase.end_epoch for phase in training_phases(settings)], [2, 5, 9])
        self.assertEqual(settings.joint_learning_rate, 0.00002)

    def test_parse_args_accepts_presence_only_schedule(self):
        settings = parse_args(["--presence-epochs", "3", "--embedding-epochs", "0", "--joint-epochs", "0"])

        phases = training_phases(settings)
        self.assertEqual([phase.name for phase in phases], ["presence"])
        self.assertEqual(phases[0].start_epoch, 1)
        self.assertEqual(phases[0].end_epoch, 3)

    def test_parse_args_rejects_empty_schedule(self):
        with self.assertRaises(SystemExit):
            parse_args(["--presence-epochs", "0", "--embedding-epochs", "0", "--joint-epochs", "0"])

    def test_training_phase_freezes_exact_module_groups(self):
        model = RoiPresenceEmbeddingModel(width=8, embedding_dim=16)

        configure_training_phase(model, "presence")
        self.assertTrue(all(parameter.requires_grad for parameter in model.backbone.parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in model.presence_head.parameters()))
        self.assertFalse(any(parameter.requires_grad for parameter in model.embedding_head.parameters()))

        configure_training_phase(model, "embedding")
        self.assertFalse(any(parameter.requires_grad for parameter in model.backbone.parameters()))
        self.assertFalse(any(parameter.requires_grad for parameter in model.presence_head.parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in model.embedding_head.parameters()))

        configure_training_phase(model, "joint")
        self.assertTrue(all(parameter.requires_grad for parameter in model.parameters()))

    def test_validation_overlap_detects_same_resolved_root(self):
        settings = RoiTrainSettings(train_roots=[Path("data/roi_samples6")], val_root=Path("data/roi_samples6"))

        self.assertTrue(validation_overlaps_training(settings))
        separate = settings.model_copy(update={"val_root": Path("data/roi_validation_samples")})
        self.assertFalse(validation_overlaps_training(separate))

    def test_best_checkpoint_score_accepts_embedding_gain(self):
        best = {
            "global_presence_f1": 0.95,
            "normal_presence_f1": 0.95,
            "short_presence_f1": 0.70,
            "global_embedding_acc": 0.80,
            "normal_embedding_acc": 0.80,
            "style_hard_negative_embedding_acc": 0.60,
            "hard_negative_sim": 0.42,
        }
        current = dict(best, global_embedding_acc=0.83, normal_embedding_acc=0.82)

        self.assertTrue(should_save_roi_best_checkpoint(current, best))

    def test_best_checkpoint_gate_saves_when_short_presence_improves_without_core_regression(self):
        best = {
            "global_presence_f1": 0.95,
            "normal_presence_f1": 0.95,
            "short_presence_f1": 0.70,
            "global_embedding_acc": 0.80,
            "normal_embedding_acc": 0.80,
            "style_hard_negative_embedding_acc": 0.60,
            "hard_negative_sim": 0.42,
        }
        current = dict(best, short_presence_f1=0.73, global_embedding_acc=0.81)

        self.assertTrue(should_save_roi_best_checkpoint(current, best))

    def test_best_checkpoint_score_allows_compensating_presence_tradeoff(self):
        best = {
            "global_presence_f1": 0.95,
            "normal_presence_f1": 0.95,
            "short_presence_f1": 0.70,
            "global_embedding_acc": 0.80,
            "normal_embedding_acc": 0.80,
            "style_hard_negative_embedding_acc": 0.60,
            "hard_negative_sim": 0.42,
        }
        current = dict(best, normal_presence_f1=0.91, short_presence_f1=0.78)

        self.assertTrue(should_save_roi_best_checkpoint(current, best))

    def test_best_checkpoint_score_does_not_use_margin_without_accuracy_gain(self):
        best = {
            "global_presence_f1": 0.95,
            "normal_presence_f1": 0.95,
            "short_presence_f1": 0.70,
            "global_embedding_acc": 0.80,
            "normal_embedding_acc": 0.80,
            "style_hard_negative_embedding_acc": 0.60,
            "hard_negative_sim_p90": 0.42,
            "hard_margin": 0.10,
        }
        current = dict(best, hard_margin=0.12, hard_negative_sim_p90=0.41)

        self.assertFalse(should_save_roi_best_checkpoint(current, best))

    def test_format_epoch_summary_groups_roi_metrics(self):
        text = format_epoch_summary(
            epoch=2,
            total_epochs=5,
            metrics={
                "train_loss": 0.123456,
                "train_presence_loss": 0.02,
                "train_embedding_loss": 0.03,
                "val_loss": 0.234567,
                "presence_f1": 0.91,
                "presence_accuracy": 0.92,
                "presence_tp": 46.0,
                "presence_fp": 3.0,
                "presence_fn": 2.0,
                "presence_tn": 29.0,
                "embedding_pair_accuracy": 0.83,
                "embedding_false_positive_pairs": 7.0,
                "embedding_false_negative_pairs": 4.0,
                "embedding_same_similarity": 0.71,
                "embedding_diff_similarity": 0.22,
                "hard_negative_sim_p50": 0.31,
                "hard_negative_sim_p90": 0.41,
                "hard_negative_sim_p95": 0.45,
                "same_sim_p50": 0.72,
                "same_sim_p10": 0.52,
                "hard_margin": 0.11,
                "checkpoint_score": 0.88,
                "best_checkpoint_score": 0.90,
                "best_epoch": 1.0,
                "training_phase": "joint",
            },
        )

        self.assertEqual(
            text,
            "\n".join(
                [
                    "roi epoch 2/5 phase=joint",
                    "  loss: train=0.1235 presence=0.0200 embedding=0.0300 val=0.2346",
                    "  presence: f1=0.9100 accuracy=0.9200 tp=46 fp=3 fn=2 tn=29",
                    "  embedding: acc=0.8300 fp=7 fn=4 same=0.7100 diff=0.2200 hard_margin=0.1100 gap=0.0000 best_threshold=0.0000",
                    "  similarity: same_p10=0.5200 same_p50=0.7200 hard_neg_p50=0.3100 hard_neg_p90=0.4100 hard_neg_p95=0.4500",
                    "  score: current=0.8800 best=0.9000 best_epoch=1",
                ]
            ),
        )

    def test_training_summary_points_to_best_outputs_without_last_metrics_blob(self):
        settings = RoiTrainSettings(
            output_dir=Path("outputs/roi_presence_only"),
            presence_epochs=3,
            embedding_epochs=0,
            joint_epochs=0,
        )
        summary = training_summary(
            settings,
            completed_epoch=30,
            total_epochs=30,
            best_metrics={
                "epoch": 3.0,
                "step": 4974.0,
                "training_phase": "presence",
                "checkpoint_score": 1.0,
                "presence_f1": 1.0,
                "presence_accuracy": 1.0,
                "presence_precision": 1.0,
                "presence_recall": 1.0,
                "presence_tp": 486.0,
                "presence_fp": 0.0,
                "presence_fn": 0.0,
                "presence_tn": 625.0,
            },
        )

        self.assertEqual(summary["record_type"], "roi_training_summary")
        self.assertEqual(summary["best_epoch"], 3)
        self.assertEqual(summary["best_checkpoint"], "outputs/roi_presence_only/best.pt")
        self.assertEqual(summary["best_epoch_metrics"], "outputs/roi_presence_only/epoch_outputs/epoch_0003/metrics.json")
        self.assertNotIn("last_metrics", summary)

    def test_presence_training_skips_embedding_pair_memory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "roi"
            output = Path(tmp) / "out"
            write_roi_dataset(root)

            with patch("src.train_roi.EmbeddingPairMemory") as memory_class:
                metrics = run_training(
                    RoiTrainSettings(
                        train_roots=[root],
                        val_root=root,
                        output_dir=output,
                        batch_size=4,
                        presence_epochs=1,
                        embedding_epochs=0,
                        joint_epochs=0,
                        max_train_samples=4,
                        max_val_samples=4,
                        width=8,
                        device="cpu",
                    )
                )

            memory_class.assert_not_called()
            self.assertEqual(metrics["training_phase"], "presence")
            self.assertEqual(metrics["train_embedding_memory_loss"], 0.0)
            self.assertEqual(metrics["train_embedding_memory_pairs"], 0.0)
            step_records = [
                json.loads(line)
                for line in (output / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
                if json.loads(line).get("record_type") == "roi_train_step"
            ]
            self.assertTrue(step_records)
            self.assertTrue(all(record["embedding_memory_loss"] == 0.0 for record in step_records))
            self.assertTrue(all(record["embedding_memory_pairs"] == 0.0 for record in step_records))

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
                    batch_size=4,
                    presence_epochs=1,
                    embedding_epochs=1,
                    joint_epochs=1,
                    max_train_samples=4,
                    max_val_samples=4,
                    width=8,
                    device="cpu",
                )
            )

            self.assertIn("presence_f1", metrics)
            self.assertIn("embedding_pair_accuracy", metrics)
            self.assertIn("global_presence_f1", metrics)
            self.assertIn("normal_presence_f1", metrics)
            self.assertIn("short_presence_f1", metrics)
            self.assertIn("global_embedding_acc", metrics)
            self.assertIn("normal_embedding_acc", metrics)
            self.assertIn("style_hard_negative_embedding_acc", metrics)
            self.assertIn("hard_negative_sim", metrics)
            self.assertIn("hard_negative_sim_p50", metrics)
            self.assertIn("hard_negative_sim_p90", metrics)
            self.assertIn("hard_negative_sim_p95", metrics)
            self.assertIn("same_sim_p50", metrics)
            self.assertIn("same_sim_p10", metrics)
            self.assertIn("hard_margin", metrics)
            self.assertIn("train_presence_negative_ratio", metrics)
            self.assertIn("train_embedding_negative_ratio", metrics)
            self.assertIn("train_embedding_positive_pairs", metrics)
            self.assertIn("train_embedding_negative_pairs", metrics)
            self.assertEqual(metrics["training_phase"], "joint")
            self.assertTrue((output / "best_presence.pt").exists())
            self.assertTrue((output / "best_embedding.pt").exists())
            self.assertTrue((output / "best_joint.pt").exists())
            self.assertTrue((output / "best.pt").exists())
            self.assertTrue((output / "error_pairs" / "epoch_0003.jsonl").exists())
            self.assertTrue((output / "error_pairs" / "epoch_0003.html").exists())


if __name__ == "__main__":
    unittest.main()
