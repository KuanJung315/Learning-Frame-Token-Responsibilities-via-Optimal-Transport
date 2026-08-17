import torch

from evaluate_timit_zero_shot import (
    _ctc_viterbi_token_frames,
    _exact_match_pairs,
    _frames_to_intervals,
    normalize_phone,
)


def main() -> None:
    # Best path: blank, A, A, blank, B.
    log_probs = torch.full((5, 3), -10.0)
    for frame, token in enumerate([0, 1, 1, 0, 2]):
        log_probs[frame, token] = 0.0
    assert _ctc_viterbi_token_frames(log_probs, [1, 2]) == [(1, 2), (4, 4)]

    starts, ends = _frames_to_intervals(
        [(1, 2), (4, 4)], duration=0.25, subsampling_factor=4
    )
    assert torch.allclose(torch.tensor(starts), torch.tensor([0.055, 0.175]))
    assert torch.allclose(torch.tensor(ends), torch.tensor([0.135, 0.215]))

    assert normalize_phone("AH0") == "ah"
    assert _exact_match_pairs(["a", "b", "c"], ["a", "x", "c"]) == [
        (0, 0),
        (2, 2),
    ]
    print("TIMIT zero-shot helper tests passed")


if __name__ == "__main__":
    main()
