#!/usr/bin/env python3

import unittest

import torch

try:
    from .evaluate_fgw_plan_transfer import (
        _agreement,
        _aligned_acoustic_features,
        _alignment_geometry,
    )
except ImportError:
    from evaluate_fgw_plan_transfer import (
        _agreement,
        _aligned_acoustic_features,
        _alignment_geometry,
    )


class TestFgwPlanTransfer(unittest.TestCase):
    def test_identity_agreement(self):
        alignment = torch.eye(4)
        result = _agreement(alignment, alignment, relative_threshold=0.1)
        self.assertAlmostEqual(result["barycenter_mad"], 0.0)
        self.assertAlmostEqual(result["support_iou"], 1.0)
        self.assertAlmostEqual(result["total_variation"], 0.0)

    def test_diagonal_geometry(self):
        diagonal = torch.eye(4)
        reverse = diagonal.flip(1)
        diag_metrics = _alignment_geometry(diagonal, 0.1, 0.12)
        reverse_metrics = _alignment_geometry(reverse, 0.1, 0.12)
        self.assertAlmostEqual(diag_metrics["diag_mean_abs_dev"], 0.0)
        self.assertAlmostEqual(diag_metrics["offdiag_mass"], 0.0)
        self.assertGreater(
            reverse_metrics["diag_mean_abs_dev"],
            diag_metrics["diag_mean_abs_dev"],
        )
        self.assertGreater(reverse_metrics["offdiag_mass"], 0.0)

    def test_training_acoustic_frame_selection(self):
        feature = torch.arange(20, dtype=torch.float32).unsqueeze(1)
        selected = _aligned_acoustic_features(
            feature,
            raw_num_frames=18,
            encoder_length=5,
            subsampling_factor=4,
        )
        self.assertEqual(selected.squeeze(1).tolist(), [2.0, 6.0, 10.0, 14.0, 17.0])


if __name__ == "__main__":
    unittest.main()
