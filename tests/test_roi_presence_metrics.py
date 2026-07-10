from __future__ import annotations

import unittest

import torch

from src.roi_presence_metrics import checkpoint_rank, presence_metrics


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

    def test_checkpoint_rank_uses_gap_only_after_presence_f1(self) -> None:
        base = {
            "global_presence_f1": 0.9,
            "normal_presence_f1": 0.9,
            "short_presence_f1": 0.9,
            "presence_gap": -0.2,
        }

        self.assertGreater(checkpoint_rank(dict(base, presence_gap=0.1)), checkpoint_rank(base))
        self.assertGreater(
            checkpoint_rank(dict(base, global_presence_f1=0.91, presence_gap=-0.5)),
            checkpoint_rank(base),
        )


if __name__ == "__main__":
    unittest.main()
