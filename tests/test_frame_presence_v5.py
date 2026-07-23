import unittest

import torch

from subfast_frame_presence.dataset import aligned_resize_size, collate_frame_presence_batch
from subfast_frame_presence.loss import FramePresenceLossInput, frame_presence_macro_loss
from subfast_frame_presence.model import FramePresenceModel


class FramePresenceV5Tests(unittest.TestCase):
    def test_scale_then_nearest_16_alignment(self):
        self.assertEqual(aligned_resize_size((1920, 1080), 0.25), (480, 272))

    def test_collate_splits_exact_sizes_without_padding(self):
        def item(width: int, height: int, label: float) -> dict[str, object]:
            return {
                "image": torch.zeros(3, height, width),
                "subtitle_mask": torch.zeros(1, height, width),
                "supervision_mask": torch.ones(1, height, width),
                "presence": torch.tensor(label),
                "sample_id": f"{width}x{height}",
                "root": "root",
                "image_path": "image.jpg",
                "sample_type": "full_frame",
                "resize_scale": 0.25,
                "output_size": (width, height),
            }

        batch = collate_frame_presence_batch([item(480, 272, 1.0), item(320, 176, 0.0)])

        self.assertEqual(batch.size, 2)
        self.assertEqual(len(batch.micro_batches), 2)
        self.assertEqual([micro.output_size for micro in batch.micro_batches], [(480, 272), (320, 176)])

    def test_macro_loss_matches_one_batch_global_reduction(self):
        logits = torch.tensor([2.0, -1.0, 1.0, -3.0], requires_grad=True)
        regions = torch.randn(4, 1, 4, 4, requires_grad=True)
        presence = torch.tensor([1.0, 0.0, 1.0, 0.0])
        masks = torch.zeros(4, 1, 32, 32)
        masks[0, :, 8:16, 8:24] = 1.0
        masks[2, :, 16:24, 4:20] = 1.0
        supervision = torch.ones_like(masks)
        kwargs = {
            "region_loss_weight": 1.0,
            "region_dice_weight": 0.5,
            "margin_loss_weight": 0.5,
            "positive_logit_margin": 4.0,
            "negative_logit_margin": -4.0,
        }

        combined = frame_presence_macro_loss(
            [FramePresenceLossInput(logits, regions, presence, masks, supervision)],
            **kwargs,
        )
        split = frame_presence_macro_loss(
            [
                FramePresenceLossInput(logits[:1], regions[:1], presence[:1], masks[:1], supervision[:1]),
                FramePresenceLossInput(logits[1:], regions[1:], presence[1:], masks[1:], supervision[1:]),
            ],
            **kwargs,
        )

        torch.testing.assert_close(split.total, combined.total)
        torch.testing.assert_close(split.region_bce, combined.region_bce)

    def test_scheme_a_has_no_normalization_and_scheme_b_supports_batch_one(self):
        scheme_a = FramePresenceModel(width=8, normalization="none")
        self.assertFalse(any(isinstance(module, (torch.nn.BatchNorm2d, torch.nn.GroupNorm)) for module in scheme_a.modules()))
        scheme_b = FramePresenceModel(width=8, normalization="group_norm").eval()
        self.assertEqual(tuple(scheme_b(torch.rand(1, 3, 16, 16)).shape), (1,))


if __name__ == "__main__":
    unittest.main()
