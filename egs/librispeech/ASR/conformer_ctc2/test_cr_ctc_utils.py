#!/usr/bin/env python3

import unittest

import torch

from cr_ctc_utils import (
    cr_ctc_consistency_loss,
    linear_warmup_weight,
    with_gradient_scale,
)


class TestCrCtcConsistencyLoss(unittest.TestCase):
    def test_identical_views_have_zero_loss(self):
        logits = torch.randn(2, 5, 7)
        lengths = torch.tensor([5, 3])
        loss = cr_ctc_consistency_loss(
            logits, logits, lengths, lengths, True, 1.0
        )
        self.assertTrue(torch.allclose(loss, torch.zeros_like(loss), atol=1.0e-6))

    def test_padding_is_ignored(self):
        logits_a = torch.randn(2, 5, 7)
        logits_b = logits_a.clone()
        logits_b[1, 3:] = torch.randn_like(logits_b[1, 3:]) * 20.0
        lengths = torch.tensor([5, 3])
        loss = cr_ctc_consistency_loss(
            logits_a, logits_b, lengths, lengths, True, 1.0
        )
        self.assertLess(abs(float(loss[1])), 1.0e-6)

    def test_stop_gradient_updates_both_views(self):
        logits_a = torch.randn(2, 5, 7, requires_grad=True)
        logits_b = torch.randn(2, 5, 7, requires_grad=True)
        lengths = torch.tensor([5, 4])
        loss = cr_ctc_consistency_loss(
            logits_a, logits_b, lengths, lengths, True, 1.0
        ).sum()
        loss.backward()
        for grad in (logits_a.grad, logits_b.grad):
            self.assertIsNotNone(grad)
            self.assertTrue(torch.isfinite(grad).all())
            self.assertGreater(float(grad.abs().sum()), 0.0)


class TestCrCtcIntegrationHelpers(unittest.TestCase):
    def test_gradient_scale_preserves_forward_and_detaches_at_zero(self):
        value = torch.randn(4, requires_grad=True)
        routed = with_gradient_scale(value, 0.0)
        self.assertTrue(torch.equal(routed, value))
        (routed.sum() + value.sum() * 0.0).backward()
        self.assertTrue(torch.equal(value.grad, torch.zeros_like(value)))

    def test_linear_warmup(self):
        self.assertEqual(linear_warmup_weight(99, 100, 3000, 0.1), 0.0)
        self.assertEqual(linear_warmup_weight(100, 100, 3000, 0.1), 0.0)
        self.assertAlmostEqual(
            linear_warmup_weight(1600, 100, 3000, 0.1), 0.05
        )
        self.assertEqual(linear_warmup_weight(3100, 100, 3000, 0.1), 0.1)
        self.assertEqual(linear_warmup_weight(0, 0, 0, 0.1), 0.1)


if __name__ == "__main__":
    unittest.main()
