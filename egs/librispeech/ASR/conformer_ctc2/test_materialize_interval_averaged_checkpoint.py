#!/usr/bin/env python3

import torch

from materialize_interval_averaged_checkpoint import interval_averaged_state


def test_interval_average_identity() -> None:
    # Running averages: start=(1+3)/2=2 at batch 200 and
    # end=(1+3+5+7)/4=4 at batch 400.  The interval mean must be (5+7)/2=6.
    start = {
        "average_period": 100,
        "batch_idx_train": 200,
        "model_avg": {"weight": torch.tensor([2.0])},
    }
    end = {
        "average_period": 100,
        "batch_idx_train": 400,
        "model_avg": {"weight": torch.tensor([4.0])},
    }
    state = interval_averaged_state(start, end)
    assert torch.allclose(state["weight"], torch.tensor([6.0]))


def test_requires_forward_interval() -> None:
    checkpoint = {
        "average_period": 100,
        "batch_idx_train": 200,
        "model_avg": {"weight": torch.tensor([2.0])},
    }
    try:
        interval_averaged_state(checkpoint, checkpoint)
    except ValueError as error:
        assert "must follow" in str(error)
    else:
        raise AssertionError("Expected a non-forward interval to fail")


if __name__ == "__main__":
    test_interval_average_identity()
    test_requires_forward_interval()
    print("interval averaged checkpoint tests passed")
