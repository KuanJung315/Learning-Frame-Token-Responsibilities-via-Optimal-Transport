#!/usr/bin/env python3

import unittest

import torch

from evaluate_mfa_alignment import (
    _word_rows,
    ctc_viterbi_align,
    summarize,
    token_spans_to_words,
)


class TestMfaAlignmentEvaluation(unittest.TestCase):
    def test_ctc_viterbi_alignment(self):
        logits = torch.tensor(
            [
                [8.0, 0.0, 0.0],
                [0.0, 8.0, 0.0],
                [8.0, 0.0, 0.0],
                [0.0, 0.0, 8.0],
                [8.0, 0.0, 0.0],
            ]
        )
        spans, score = ctc_viterbi_align(logits.log_softmax(dim=-1), [1, 2])
        self.assertEqual(spans, [(1, 2), (3, 4)])
        self.assertLessEqual(score, 0.0)

    def test_repeated_ctc_label_requires_blank(self):
        logits = torch.tensor(
            [
                [0.0, 8.0],
                [8.0, 0.0],
                [0.0, 8.0],
            ]
        )
        spans, _ = ctc_viterbi_align(logits.log_softmax(dim=-1), [1, 1])
        self.assertEqual(spans, [(0, 1), (2, 3)])

    def test_merge_token_spans_into_word_spans(self):
        word_spans = token_spans_to_words(
            token_spans=[(1, 2), (3, 5), (6, 7)],
            token_ranges=[(0, 2), (2, 3)],
            seconds_per_frame=0.04,
            duration=1.0,
        )
        self.assertEqual(word_spans, [(0.04, 0.20), (0.24, 0.28)])

    def test_paper_matched_utterance_macro_wbe(self):
        rows = []
        for model, predicted in (
            ("baseline", [(0.1, 0.9), (1.2, 1.8)]),
            ("candidate", [(0.0, 1.0), (1.0, 2.0)]),
        ):
            rows.extend(
                _word_rows(
                    dataset="dev-clean",
                    cut_id="utt-1",
                    model_name=model,
                    reference=[("A", 0.0, 1.0), ("B", 1.0, 2.0)],
                    predicted=predicted,
                    path_score=-1.0,
                    num_frames=50,
                )
            )
        result = summarize(
            rows,
            candidate_name="candidate",
            bootstrap_samples=20,
            bootstrap_seed=0,
        )
        baseline = result["datasets"]["combined"]["baseline"]
        candidate = result["datasets"]["combined"]["candidate"]
        self.assertAlmostEqual(
            baseline["utterance_macro_metrics"]["boundary_abs_error_ms"]["mean"],
            150.0,
        )
        self.assertAlmostEqual(
            candidate["utterance_macro_metrics"]["boundary_abs_error_ms"]["mean"],
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
