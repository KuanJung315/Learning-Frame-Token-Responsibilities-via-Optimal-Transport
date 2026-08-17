#!/usr/bin/env python3
"""Tests for opt-in epoch checkpoint pruning."""

from pathlib import Path
from tempfile import TemporaryDirectory

from icefall.utils import AttributeDict
from train import _parse_retained_epochs, _prune_epoch_checkpoints


def test_parse_retained_epochs() -> None:
    assert _parse_retained_epochs("") == set()
    assert _parse_retained_epochs("20, 30") == {20, 30}
    try:
        _parse_retained_epochs("-1")
    except ValueError:
        pass
    else:
        raise AssertionError("negative retained epoch should fail")


def test_prune_retains_anchor_and_latest() -> None:
    with TemporaryDirectory() as directory:
        exp_dir = Path(directory)
        for epoch in range(1, 31):
            (exp_dir / f"epoch-{epoch}.pt").touch()
        params = AttributeDict(
            {
                "exp_dir": exp_dir,
                "keep_last_epoch_checkpoints": 2,
                "retain_epoch_checkpoints": "20",
            }
        )
        _prune_epoch_checkpoints(params)
        assert sorted(path.name for path in exp_dir.glob("epoch-*.pt")) == [
            "epoch-20.pt",
            "epoch-29.pt",
            "epoch-30.pt",
        ]


if __name__ == "__main__":
    test_parse_retained_epochs()
    test_prune_retains_anchor_and_latest()
    print("epoch checkpoint retention tests passed")
