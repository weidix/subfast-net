from __future__ import annotations

import unittest

import torch

from src.roi_pair_model import (
    RoiPairMatcher,
    fuse_pair_matcher_for_inference,
    trace_pair_matcher_for_inference,
)


class RoiPairModelTest(unittest.TestCase):
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
