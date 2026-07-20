from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from subfast_roi_data.data import RoiPairBatch, RoiPairDataset, RoiSample, collate_pair_batch
from subfast_roi_data.pairs import (
    RoiPair,
    RoiPairEpochSchedule,
    RoiPairPools,
    RoiPairSelection,
    build_pair_epoch_schedule,
    select_pairs,
)


def write_roi_dataset(root: Path, *, size: tuple[int, int] = (32, 16)) -> None:
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir()
    rows = [
        ("a0", True, "seg-a", 0, "same subtitle"),
        ("a1", True, "seg-a", 30, "same subtitle"),
        ("b0", True, "seg-b", 60, "different subtitle"),
        ("e0", False, "empty", 90, ""),
    ]
    annotations = []
    for name, has_subtitle, segment_id, frame_index, ocr_text in rows:
        Image.new("RGB", size, "white" if has_subtitle else "black").save(
            root / "images" / f"{name}.jpg"
        )
        annotations.append(
            {
                "image": f"images/{name}.jpg",
                "source_annotation": {
                    "source_video": "video-1",
                    "frame_index": frame_index,
                },
                "roi_size": list(size),
                "has_subtitle": has_subtitle,
                "segment_marker": segment_id,
                "ocr_text": ocr_text,
            }
        )
        label = "0 0.250000 0.437500 0.250000 0.375000\n" if has_subtitle else ""
        (root / "labels" / f"{name}.txt").write_text(label, encoding="utf-8")
    (root / "annotations.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in annotations),
        encoding="utf-8",
    )
    (root / "summary.json").write_text(
        json.dumps(
            {
                "version": 1,
                "roi_size": list(size),
                "samples": len(rows),
                "positive": 3,
                "empty": 1,
            }
        ),
        encoding="utf-8",
    )


def make_scheduler_samples(*, segments: int = 3, per_segment: int = 4) -> list[RoiSample]:
    segment_texts = ("alpha caption", "bravo message", "charlie sentence")
    samples: list[RoiSample] = []
    for segment_index in range(segments):
        for offset in range(per_segment):
            sample_index = segment_index * per_segment + offset
            samples.append(
                RoiSample(
                    image_path=Path(f"positive-{sample_index}.jpg"),
                    label_path=Path(f"positive-{sample_index}.txt"),
                    sample_id=f"positive-{sample_index}",
                    root=Path("root"),
                    has_subtitle=True,
                    segment_id=f"segment-{segment_index}",
                    video_id="video",
                    frame_index=segment_index * 100 + offset,
                    ocr_text=segment_texts[segment_index],
                    annotation={},
                )
            )
    return samples


class RoiPairTests(unittest.TestCase):
    def test_dataset_collates_direct_matcher_supervision(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_roi_dataset(root)

            dataset = RoiPairDataset([root])
            batch = collate_pair_batch([dataset[0], dataset[3]])

        self.assertIsInstance(batch, RoiPairBatch)
        self.assertEqual(dataset.summary.roi_size, (32, 16))
        self.assertEqual(dataset.summary.same_segment_pairs, 1)
        self.assertEqual(batch.images.shape, (2, 3, 16, 32))
        self.assertEqual(batch.subtitle_masks.shape, (2, 1, 16, 32))
        self.assertEqual(batch.presence.tolist(), [1.0, 0.0])
        self.assertEqual(batch.segment_ids, ["seg-a", "empty"])
        self.assertEqual(batch.adjacent_segment_ids, [frozenset({"seg-b"}), frozenset()])
        self.assertEqual(float(batch.subtitle_masks[0].sum()), 48.0)

    def test_pair_selection_uses_only_valid_local_pairs(self) -> None:
        selection = select_pairs(
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

        self.assertIsInstance(selection, RoiPairSelection)
        self.assertEqual(
            [(pair.i, pair.j, pair.same) for pair in selection.pairs],
            [(0, 1, True), (0, 2, False), (0, 6, True), (1, 2, False), (1, 6, True), (2, 6, False)],
        )
        self.assertEqual(selection.local_positive_pairs, 3)
        self.assertEqual(selection.local_negative_pairs, 3)

    def test_pair_schedule_balances_pairs_and_preserves_roi_identity(self) -> None:
        samples = make_scheduler_samples()
        schedule = build_pair_epoch_schedule(
            samples,
            batch_size=4,
            negative_ratio=0.5,
            ocr_negative_enabled=False,
            ocr_negative_max_similarity=0.2,
            ocr_negative_ratio=0.3,
            seed=17,
            epoch=1,
        )
        global_pairs = {
            tuple(sorted((batch.sample_indices[pair.i], batch.sample_indices[pair.j])))
            for batch in schedule.batches
            for pair in batch.pairs
            if pair.same
        }
        pair_ids = [pair.pair_id for batch in schedule.batches for pair in batch.pairs]

        self.assertIsInstance(schedule, RoiPairEpochSchedule)
        self.assertEqual(schedule.positive_pair_count, 12)
        self.assertEqual(schedule.negative_pair_count, 12)
        self.assertEqual(schedule.unique_positive_pair_count, 12)
        self.assertEqual(schedule.unique_negative_pair_count, 12)
        self.assertEqual(schedule.unique_positive_roi_count, schedule.total_positive_roi_count)
        self.assertTrue(all(len(batch.sample_indices) <= 4 for batch in schedule.batches))
        self.assertTrue(all(pair_id.startswith("root|positive-") for pair_id in pair_ids))
        for segment_index in range(3):
            start = segment_index * 4
            self.assertIn((start, start + 3), global_pairs)

    def test_pair_schedule_reserves_ocr_negative_budget(self) -> None:
        samples = make_scheduler_samples()
        pools = RoiPairPools(
            positive_pairs=tuple(
                RoiPair(i=index, j=index + 1, same=True, source="local")
                for index in range(0, 12, 2)
            ),
            local_negative_pairs=tuple(
                RoiPair(i=index, j=index + 2, same=False, source="local")
                for index in range(6)
            ),
            ocr_negative_pairs=tuple(
                RoiPair(i=index, j=index + 6, same=False, source="ocr")
                for index in range(6)
            ),
            total_positive_roi_count=12,
        )

        schedule = build_pair_epoch_schedule(
            samples,
            batch_size=4,
            negative_ratio=0.5,
            ocr_negative_enabled=True,
            ocr_negative_max_similarity=0.2,
            ocr_negative_ratio=0.5,
            seed=29,
            epoch=1,
            pair_pools=pools,
        )
        negative_sources = [
            pair.source
            for batch in schedule.batches
            for pair in batch.pairs
            if not pair.same
        ]

        self.assertEqual(negative_sources.count("local"), 3)
        self.assertEqual(negative_sources.count("ocr"), 3)


if __name__ == "__main__":
    unittest.main()
