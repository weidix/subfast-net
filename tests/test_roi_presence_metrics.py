from __future__ import annotations

import unittest

import torch
from torch import nn
from torch.nn import functional as F

from subfast_net.roi.presence.config import RoiPresenceTrainSettings
from subfast_net.roi.presence.dataset import RoiPresenceDataset
from subfast_net.roi.presence.loss import counterfactual_presence_loss, subtitle_region_loss
from subfast_net.roi.presence.metrics import (
    checkpoint_rank,
    presence_metrics,
    text_distractor_metrics,
)
from subfast_net.roi.presence.model import CoherentEvidencePooling, RoiPresenceModel
from subfast_net.roi.presence.train import make_training_loader


class RoiPresenceMetricsTest(unittest.TestCase):
    def test_presence_metrics_report_probability_separation(self) -> None:
        metrics = presence_metrics(
            torch.logit(torch.tensor([0.90, 0.70, 0.40, 0.10])),
            torch.tensor([1.0, 1.0, 0.0, 0.0]),
        )

        self.assertAlmostEqual(metrics["presence_min_positive_score"], 0.70, places=6)
        self.assertAlmostEqual(metrics["presence_max_negative_score"], 0.40, places=6)
        self.assertAlmostEqual(metrics["presence_gap"], 0.30, places=6)
        self.assertEqual(metrics["presence_zero_error_threshold_exists"], 1.0)
        self.assertEqual(metrics["presence_roc_auc"], 1.0)
        self.assertEqual(metrics["presence_best_f1"], 1.0)

    def test_checkpoint_rank_prefers_localized_causal_evidence_after_presence_f1(self) -> None:
        base = {
            "global_presence_f1": 0.9,
            "normal_presence_f1": 0.9,
            "short_presence_f1": 0.9,
            "region_pointing_accuracy": 0.5,
            "counterfactual_score_drop_lower_tail_1pct": 0.1,
            "presence_tail_gap_1pct": 0.2,
            "presence_brier": 0.1,
        }

        self.assertGreater(
            checkpoint_rank(dict(base, region_pointing_accuracy=0.9)),
            checkpoint_rank(base),
        )
        self.assertGreater(
            checkpoint_rank(dict(base, global_presence_f1=0.91, region_pointing_accuracy=0.0)),
            checkpoint_rank(base),
        )

    def test_auc_does_not_hide_fixed_threshold_false_negatives(self) -> None:
        metrics = presence_metrics(
            torch.logit(torch.tensor([0.49, 0.48, 0.20, 0.10])),
            torch.tensor([1.0, 1.0, 0.0, 0.0]),
        )

        self.assertEqual(metrics["presence_roc_auc"], 1.0)
        self.assertEqual(metrics["presence_fn"], 2.0)
        self.assertLess(metrics["presence_positive_lower_tail_mean_1pct"], 0.5)

    def test_text_distractor_slice_is_not_reported_as_empty_background(self) -> None:
        logits = torch.logit(torch.tensor([0.9, 0.8, 0.1]))
        presence = torch.tensor([1.0, 0.0, 0.0])
        candidate_masks = torch.zeros(3, 1, 2, 2)
        candidate_masks[:2, :, 0, 0] = 1.0

        metrics, selected = text_distractor_metrics(logits, presence, candidate_masks)

        self.assertEqual(selected.tolist(), [False, True, False])
        self.assertEqual(metrics["text_distractor_fpr"], 1.0)
        self.assertEqual(metrics["subtitle_specificity_evaluable"], 1.0)


class RoiPresenceOperatorTest(unittest.TestCase):
    def test_counterfactual_transplant_is_supervised_as_a_fixed_positive_target(self) -> None:
        original = torch.tensor([20.0], requires_grad=True)
        erased = torch.tensor([-20.0], requires_grad=True)
        transplanted = torch.tensor([-2.0], requires_grad=True)
        seam = torch.tensor([-20.0], requires_grad=True)

        loss = counterfactual_presence_loss(
            original,
            erased,
            transplanted,
            seam,
            margin=2.0,
        )
        loss.total.backward()

        self.assertGreater(float(loss.sufficiency.detach()), 2.0)
        self.assertLess(float(transplanted.grad.detach()), 0.0)
        self.assertEqual(float(original.grad.detach()), 0.0)

    def test_training_loader_drops_a_tiny_shuffled_remainder(self) -> None:
        class SizedDataset:
            samples: list[object] = []

            def __len__(self) -> int:
                return 18

        settings = RoiPresenceTrainSettings(batch_size=16)

        loader, sampler = make_training_loader(SizedDataset(), settings)  # type: ignore[arg-type]

        self.assertIsNone(sampler)
        self.assertTrue(loader.drop_last)
        self.assertEqual(len(loader), 1)

    def test_letterbox_resize_keeps_image_mask_and_valid_coordinates_aligned(self) -> None:
        dataset = object.__new__(RoiPresenceDataset)
        dataset.resize_roi = (40, 40)
        dataset.resize_mode = "letterbox"
        image = torch.zeros(3, 10, 20)
        mask = torch.zeros(1, 10, 20)
        mask[:, 2:8, 5:15] = 1.0
        valid = torch.ones(1, 10, 20)

        resized_image, resized_mask, resized_valid = dataset._resize_tensors(image, mask, valid)

        self.assertEqual(resized_image.shape, (3, 40, 40))
        self.assertEqual(resized_mask.shape, (1, 40, 40))
        self.assertTrue(bool((resized_valid[:, :10] == 0.0).all()))
        self.assertTrue(bool((resized_valid[:, 10:30] == 1.0).all()))
        self.assertTrue(bool((resized_valid[:, 30:] == 0.0).all()))
        self.assertTrue(bool((resized_mask * (1.0 - resized_valid) == 0.0).all()))

        model = RoiPresenceModel(width=8).eval()
        resized_image[:, 10:30] = torch.randn_like(resized_image[:, 10:30])
        torch.testing.assert_close(
            model(resized_image.unsqueeze(0)),
            model(resized_image.unsqueeze(0), resized_valid.unsqueeze(0)),
        )

    def test_model_is_batch_context_and_train_eval_invariant(self) -> None:
        torch.manual_seed(7)
        model = RoiPresenceModel(width=8)
        sample = torch.randn(1, 3, 32, 64)
        companions = torch.randn(3, 3, 32, 64) * 8.0 + 20.0

        model.train()
        alone = model(sample)
        mixed = model(torch.cat([sample, companions]))[:1]
        model.eval()
        evaluated = model(sample)

        self.assertFalse(any(isinstance(module, nn.modules.batchnorm._BatchNorm) for module in model.modules()))
        torch.testing.assert_close(alone, mixed, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(alone, evaluated, atol=1e-6, rtol=1e-6)

    def test_coherent_pooling_rejects_an_isolated_peak(self) -> None:
        pooling = CoherentEvidencePooling(kernel_size=3)
        isolated = torch.full((1, 1, 9, 9), -5.0)
        coherent = isolated.clone()
        isolated[:, :, 4, 4] = 9.0
        coherent[:, :, 3:6, 3:6] = 9.0

        self.assertGreater(
            float(pooling(coherent).detach()),
            float(pooling(isolated).detach()) + 3.0,
        )

    def test_coherent_pooling_keeps_all_invalid_inputs_finite(self) -> None:
        pooling = CoherentEvidencePooling(kernel_size=5)
        logit = pooling(torch.randn(1, 1, 4, 8), torch.zeros(1, 1, 4, 8))

        self.assertTrue(bool(torch.isfinite(logit).all()))
        self.assertTrue(bool(torch.isfinite(F.binary_cross_entropy_with_logits(logit, torch.zeros(1)))))

    def test_coherent_pooling_uses_partial_support_for_thin_valid_regions(self) -> None:
        pooling = CoherentEvidencePooling(kernel_size=5)
        valid = torch.zeros(1, 1, 16, 32)
        valid[:, :, 6:10] = 1.0
        low = torch.zeros(1, 1, 16, 32)
        high = low.clone()
        high[:, :, 6:10] = 2.0

        self.assertGreater(
            float(pooling(high, valid).detach()),
            float(pooling(low, valid).detach()) + 1.0,
        )

    def test_model_valid_mask_matches_dense_area_coordinates(self) -> None:
        model = RoiPresenceModel(width=8)
        input_mask = torch.zeros(1, 1, 64, 512)
        input_mask[:, :, 18:46] = 1.0
        region_logits = torch.zeros(1, 1, 16, 128)

        actual = model.downsample_valid_mask(input_mask, region_logits)
        expected = (F.interpolate(input_mask, size=(16, 128), mode="area") > 0.5).to(actual.dtype)
        torch.testing.assert_close(actual, expected)

    def test_dense_region_loss_supervises_positive_extent_and_text_distractors(self) -> None:
        logits = torch.zeros(2, 1, 4, 8, requires_grad=True)
        masks = torch.zeros(2, 1, 4, 8)
        masks[:, :, 1:3, 2:6] = 1.0
        presence = torch.tensor([1.0, 0.0])

        loss = subtitle_region_loss(
            logits,
            masks,
            presence,
            dice_weight=0.0,
            projection_weight=0.0,
            text_distractor_weight=4.0,
        )
        loss.total.backward()

        self.assertLess(float(logits.grad[0, 0, 1, 2]), 0.0)
        self.assertGreater(float(logits.grad[0, 0, 0, 0]), 0.0)
        self.assertGreater(float(logits.grad[1, 0, 1, 2]), 0.0)


if __name__ == "__main__":
    unittest.main()
