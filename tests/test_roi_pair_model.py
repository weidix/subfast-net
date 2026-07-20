from __future__ import annotations

import unittest

import torch

from subfast_roi_matcher.model import (
    RoiPairMatcher,
    fuse_pair_matcher_for_inference,
    trace_pair_matcher_for_inference,
)
from subfast_shared.vision import IMAGENET_MEAN, IMAGENET_STD


class RoiPairModelTest(unittest.TestCase):
    def test_pair_features_ignore_hue_and_saturation(self) -> None:
        torch.manual_seed(7)
        left_value = torch.rand(2, 1, 16, 32) * 0.7 + 0.2
        right_value = torch.rand_like(left_value) * 0.7 + 0.2

        def normalize(red: torch.Tensor, green: torch.Tensor, blue: torch.Tensor) -> torch.Tensor:
            rgb = torch.cat((red, green, blue), dim=1)
            return (rgb - IMAGENET_MEAN) / IMAGENET_STD

        left_a = normalize(left_value, left_value * 0.15, left_value * 0.55)
        right_a = normalize(right_value * 0.35, right_value, right_value * 0.60)
        left_b = normalize(left_value * 0.70, left_value, left_value * 0.05)
        right_b = normalize(right_value, right_value * 0.10, right_value * 0.80)

        actual = RoiPairMatcher.pair_features(left_a, right_a)
        recolored = RoiPairMatcher.pair_features(left_b, right_b)

        torch.testing.assert_close(recolored, actual, atol=1e-6, rtol=1e-6)

    def test_fused_and_traced_inference_preserve_eval_outputs(self) -> None:
        torch.manual_seed(11)
        model = RoiPairMatcher().eval()
        left = torch.randn(2, 3, 32, 64)
        right = torch.randn_like(left)

        with torch.inference_mode():
            expected = model(left, right)
            fused = fuse_pair_matcher_for_inference(model)(left, right)
            traced = trace_pair_matcher_for_inference(model, left, right)(left, right)

        torch.testing.assert_close(fused[0], expected[0], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(fused[1], expected[1], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(traced, expected[0], atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
