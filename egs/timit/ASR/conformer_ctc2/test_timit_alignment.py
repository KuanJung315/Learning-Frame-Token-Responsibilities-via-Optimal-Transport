import torch

from evaluate_timit_alignment import (
    add_frame_accuracy,
    boundary_frame_token_ids,
    frame_to_seconds,
    gold_frame_token_ids,
    isotonic_non_decreasing,
    plan_frame_token_ids,
)


def test_encoder_frame_timing():
    assert frame_to_seconds(0) == 0.035
    assert frame_to_seconds(1) == 0.075
    assert frame_to_seconds(10) == 0.435


def test_weighted_isotonic_projection():
    projected = isotonic_non_decreasing(
        values=[0.1, 0.5, 0.3, 0.8],
        weights=[1.0, 1.0, 1.0, 1.0],
    )
    assert projected == [0.1, 0.4, 0.4, 0.8]


def test_gold_frame_token_ids_uses_boundary_intervals():
    labels = torch.tensor([10, 20, 30])
    got = gold_frame_token_ids(
        labels=labels,
        gold_boundaries_sec=[0.1, 0.2],
        num_frames=6,
    )
    assert got.tolist() == [10, 10, 20, 20, 20, 30]


def test_plan_and_boundary_frame_token_ids():
    labels = torch.tensor([5, 7, 9])
    plan = torch.tensor(
        [
            [0.8, 0.1, 0.1],
            [0.2, 0.7, 0.1],
            [0.1, 0.2, 0.7],
        ]
    )
    assert plan_frame_token_ids(plan, labels).tolist() == [5, 7, 9]
    assert boundary_frame_token_ids(
        labels=labels,
        boundaries_sec=[0.06, 0.10],
        num_frames=3,
    ).tolist() == [5, 7, 9]


def test_add_frame_accuracy():
    row = {}
    add_frame_accuracy(
        row,
        "method",
        predicted_token_ids=torch.tensor([1, 2, 4, 4]),
        gold_token_ids=torch.tensor([1, 2, 3, 4]),
    )
    assert row["method__frame_correct"] == 3
    assert row["method__frame_count"] == 4
    assert row["method__frame_accuracy"] == 0.75


if __name__ == "__main__":
    test_encoder_frame_timing()
    test_weighted_isotonic_projection()
    test_gold_frame_token_ids_uses_boundary_intervals()
    test_plan_and_boundary_frame_token_ids()
    test_add_frame_accuracy()
    print("All TIMIT alignment tests passed.")
