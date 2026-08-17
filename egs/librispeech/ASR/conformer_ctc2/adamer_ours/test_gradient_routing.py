#!/usr/bin/env python3

import unittest

import torch

from gradient_routing import with_gradient_scale


class TestGradientRouting(unittest.TestCase):
    def test_forward_value_is_unchanged(self):
        value = torch.randn(4, requires_grad=True)
        for scale in (0.0, 0.25, 1.0):
            routed = with_gradient_scale(value, scale)
            self.assertTrue(torch.equal(routed, value))

    def test_zero_scale_detaches_gradient(self):
        value = torch.randn(4, requires_grad=True)
        routed = with_gradient_scale(value, 0.0)
        (routed.sum() + value.sum() * 0.0).backward()
        self.assertTrue(torch.equal(value.grad, torch.zeros_like(value)))

    def test_fractional_scale_scales_gradient(self):
        value = torch.randn(4, requires_grad=True)
        with_gradient_scale(value, 0.25).sum().backward()
        self.assertTrue(torch.allclose(value.grad, torch.full_like(value, 0.25)))


if __name__ == "__main__":
    unittest.main()
