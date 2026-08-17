from pathlib import Path

import torch

from evaluate_mfa_phone_alignment import (
    _encoder_aligned_raw_features,
    _input_order_transcripts,
    _normalized_reference_words,
    summarize_geometry,
)
from word_phone_graph_compiler import PhoneTranscript


def transcript(word: str, phone_id: int) -> PhoneTranscript:
    return PhoneTranscript(
        words=[word],
        phones=[word],
        phone_ids=[phone_id],
        word_phone_spans=[(0, 1)],
        oov_words=[],
    )


def main() -> None:
    raw = torch.arange(30, dtype=torch.float32).reshape(10, 3)
    aligned = _encoder_aligned_raw_features(raw, 9, 3, 4)
    assert torch.equal(aligned, raw[torch.tensor([2, 6, 8])])

    first = transcript("FIRST", 1)
    second = transcript("SECOND", 2)
    assert _input_order_transcripts([first, second], [1, 0], 2) == [
        second,
        first,
    ]

    class Compiler:
        @staticmethod
        def normalize_words(text):
            return [text.replace("'", "").upper()]

    reference = [("don't", 0.0, 0.1), ("stop", 0.1, 0.2)]
    assert _normalized_reference_words(Compiler(), reference) == [
        "DONT",
        "STOP",
    ]

    rows = []
    for model, w1, iou in (
        ("baseline", 3.0, 0.2),
        ("candidate", 2.0, 0.4),
    ):
        rows.append(
            {
                "dataset": "dev-clean",
                "cut_id": "utt",
                "model": model,
                "plan_ctc_w1_frames": w1,
                "plan_ctc_barycenter_mae_frames": w1,
                "plan_ctc_support_iou": iou,
                "plan_diagonal_deviation": 0.3,
                "ctc_diagonal_deviation": 0.4,
            }
        )
    summary = summarize_geometry(rows, "candidate")
    combined = summary["combined"]
    assert (
        combined["paired"]["plan_ctc_w1_frames"][
            "candidate_minus_baseline"
        ]
        == -1.0
    )
    assert (
        combined["paired"]["plan_ctc_support_iou"][
            "candidate_minus_baseline"
        ]
        == 0.2
    )
    print("Libri phone MFA helper tests passed")


if __name__ == "__main__":
    main()
