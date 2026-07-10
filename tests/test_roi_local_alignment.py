from __future__ import annotations

import unittest

import torch

from src.roi_local_alignment import extreme_gap_loss, local_alignment_similarity


def _tokens(count: int = 10) -> torch.Tensor:
    return torch.eye(count).unsqueeze(0)


class RoiLocalAlignmentTests(unittest.TestCase):
    def test_bottom_token_aggregation_amplifies_one_token_difference(self) -> None:
        left = _tokens()
        identical = local_alignment_similarity(left, left, position_penalty=0.0)
        changed = left.clone()
        changed[:, 4] = -changed[:, 4]
        changed_score = local_alignment_similarity(left, changed, position_penalty=0.0)

        self.assertTrue(torch.allclose(identical, torch.ones_like(identical)))
        self.assertLess(changed_score.item(), 0.85)

    def test_band_rejects_an_identical_token_outside_allowed_position(self) -> None:
        left = _tokens()
        shifted = torch.roll(left, shifts=4, dims=1)

        narrow = local_alignment_similarity(left, shifted, bandwidth=3, position_penalty=0.0)
        unrestricted = local_alignment_similarity(left, shifted, bandwidth=9, position_penalty=0.0)

        self.assertLess(narrow.item(), unrestricted.item())

    def test_extreme_gap_loss_focuses_on_worst_pair_tails(self) -> None:
        scores = torch.tensor([0.9, 0.4, 0.1, 0.8], requires_grad=True)
        targets = torch.tensor([1.0, 1.0, 0.0, 0.0])
        loss = extreme_gap_loss(scores, targets, tail_ratio=0.5)
        loss.backward()

        self.assertIsNotNone(scores.grad)
        assert scores.grad is not None
        self.assertLess(scores.grad[1].item(), 0.0)
        self.assertGreater(scores.grad[3].item(), 0.0)


if __name__ == "__main__":
    unittest.main()
