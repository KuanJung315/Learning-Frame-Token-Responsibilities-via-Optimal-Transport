#!/usr/bin/env python3

import unittest

import torch

from evaluate_timit_alignment import (
    _assign_phones_to_words,
    _encoder_aligned_acoustic_features,
    _interval_error_sums,
    plan_agreement,
)
from ot_fgw import vi_fgw_loss_v2, vi_fgw_loss_v2_batched
from ot_prior_v2 import vi_ot_loss_v2


class TestTimitFgw(unittest.TestCase):
    def test_lambda_zero_matches_ot(self):
        torch.manual_seed(0)
        logits = torch.randn(12, 40)
        log_probs = torch.log_softmax(logits, dim=-1)
        alpha = torch.sigmoid(torch.randn(12))
        labels = torch.tensor([1, 2, 3, 4, 5])
        acoustic = torch.randn(12, 80)
        kwargs = dict(
            log_p_nonblank=log_probs,
            alpha=alpha,
            labels=labels,
            column_marginal_type="acoustic",
            eps=0.05,
            iters=7,
            beta_pos=50.0,
        )
        expected = vi_ot_loss_v2(**kwargs)
        actual = vi_fgw_loss_v2(
            **kwargs, acoustic_features=acoustic, lambda_gw=0.0
        )
        self.assertTrue(torch.allclose(expected, actual, atol=1.0e-7))

    def test_batched_fgw_matches_variable_length_loop(self):
        torch.manual_seed(2026)
        batch_size, max_frames, max_tokens = 3, 9, 5
        frame_lens = torch.tensor([9, 6, 8])
        label_lens = torch.tensor([5, 3, 4])
        log_probs = torch.log_softmax(torch.randn(batch_size, max_frames, 40), -1)
        alpha = torch.sigmoid(torch.randn(batch_size, max_frames))
        labels = torch.randint(1, 41, (batch_size, max_tokens))
        acoustic = torch.randn(batch_size, max_frames, 80)
        kwargs = dict(
            column_marginal_type="acoustic",
            alpha_smooth_mix=0.1,
            token_prior_sigma=0.15,
            token_prior_score_temp=1.0,
            token_prior_floor=0.05,
            eps=0.05,
            iters=20,
            beta_pos=50.0,
            lambda_gw=0.2,
            n_outer=3,
            return_plan=True,
        )
        actual_loss, actual_plan = vi_fgw_loss_v2_batched(
            log_p_nonblank=log_probs,
            alpha=alpha,
            labels=labels,
            frame_lens=frame_lens,
            label_lens=label_lens,
            acoustic_features=acoustic,
            **kwargs,
        )
        for i in range(batch_size):
            num_frames = int(frame_lens[i])
            num_tokens = int(label_lens[i])
            expected_loss, expected_plan = vi_fgw_loss_v2(
                log_p_nonblank=log_probs[i, :num_frames],
                alpha=alpha[i, :num_frames],
                labels=labels[i, :num_tokens],
                acoustic_features=acoustic[i, :num_frames],
                **kwargs,
            )
            self.assertTrue(
                torch.allclose(actual_loss[i], expected_loss, atol=2.0e-5),
                (actual_loss[i], expected_loss),
            )
            self.assertTrue(
                torch.allclose(
                    actual_plan[i, :num_frames, :num_tokens],
                    expected_plan,
                    atol=2.0e-5,
                )
            )

    def test_positive_lambda_without_outer_step_keeps_ot_marginal_and_plan(self):
        torch.manual_seed(9)
        log_probs = torch.log_softmax(torch.randn(10, 40), dim=-1)
        alpha = torch.sigmoid(torch.randn(10))
        labels = torch.tensor([1, 7, 3, 9])
        acoustic = torch.randn(10, 80)
        kwargs = dict(
            log_p_nonblank=log_probs,
            alpha=alpha,
            labels=labels,
            column_marginal_type="acoustic",
            eps=0.05,
            iters=30,
            beta_pos=50.0,
            return_plan=True,
        )
        expected_loss, expected_plan = vi_ot_loss_v2(**kwargs)
        actual_loss, actual_plan = vi_fgw_loss_v2(
            **kwargs,
            acoustic_features=acoustic,
            lambda_gw=0.2,
            n_outer=0,
        )
        self.assertTrue(torch.allclose(actual_loss, expected_loss, atol=1.0e-7))
        self.assertTrue(torch.allclose(actual_plan, expected_plan, atol=1.0e-7))

    def test_training_frame_selection(self):
        feature = torch.arange(20, dtype=torch.float32).unsqueeze(1)
        selected = _encoder_aligned_acoustic_features(
            feature, raw_num_frames=18, encoder_length=5, subsampling_factor=4
        )
        self.assertEqual(
            selected.squeeze(1).tolist(), [2.0, 6.0, 10.0, 14.0, 17.0]
        )

    def test_interval_metrics(self):
        values = _interval_error_sums(
            [0.0, 1.0], [1.0, 2.0], [0.1, 0.9], [0.8, 2.2]
        )
        self.assertEqual(values["count"], 2)
        self.assertAlmostEqual(values["onset_abs_sum_ms"], 200.0)
        self.assertAlmostEqual(values["offset_abs_sum_ms"], 400.0)
        self.assertAlmostEqual(values["boundary_abs_sum_ms"], 300.0)
        self.assertAlmostEqual(values["duration_abs_sum_ms"], 600.0)
        self.assertAlmostEqual(values["gold_duration_sum_ms"], 2000.0)
        self.assertAlmostEqual(values["predicted_duration_sum_ms"], 2000.0)

    def test_phone_word_assignment_and_identity_agreement(self):
        groups, unassigned = _assign_phones_to_words(
            [0.0, 0.2, 0.4, 0.6],
            [0.2, 0.4, 0.6, 0.8],
            [0.0, 0.4],
            [0.4, 0.8],
        )
        self.assertEqual(groups, [[0, 1], [2, 3]])
        self.assertEqual(unassigned, 0)
        agreement = plan_agreement(torch.eye(4), torch.eye(4))
        self.assertAlmostEqual(agreement["barycenter_mad"], 0.0)
        self.assertAlmostEqual(agreement["support_iou"], 1.0)
        self.assertAlmostEqual(agreement["total_variation"], 0.0)


if __name__ == "__main__":
    unittest.main()
