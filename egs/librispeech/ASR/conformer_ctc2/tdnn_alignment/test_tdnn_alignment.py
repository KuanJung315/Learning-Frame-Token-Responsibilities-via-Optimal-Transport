import math

import torch

from model import TdnnLabelPriorCTC, apply_label_prior


def test_reference_topology_shape_and_size():
    model = TdnnLabelPriorCTC(
        num_features=80,
        num_classes=42,
        subsampling_factor=2,
        d_model=640,
        num_decoder_layers=0,
    )
    supervisions = {
        "num_frames": torch.tensor([96, 101]),
        "sequence_idx": torch.tensor([1, 0]),
    }
    log_probs, memory, padding_mask = model(
        torch.randn(2, 104, 80), supervisions
    )
    assert log_probs.shape == (2, 52, 42)
    assert memory.shape == (52, 2, 640)
    assert padding_mask.shape == (2, 52)
    assert (~padding_mask).sum(dim=1).tolist() == [51, 48]
    assert 4_700_000 < sum(p.numel() for p in model.parameters()) < 5_100_000


def test_label_prior_score_and_epoch_update():
    model = TdnnLabelPriorCTC(
        num_features=80,
        num_classes=3,
        subsampling_factor=2,
        d_model=8,
        num_decoder_layers=0,
    )
    scores = torch.log_softmax(torch.randn(1, 4, 3), dim=-1)
    model.accumulate_label_prior_stats(
        scores, torch.tensor([[0, 0, 4]], dtype=torch.int32)
    )
    stats = model.sync_and_update_label_prior(
        momentum=0.0, floor=math.exp(-12.0)
    )
    assert stats["frames"] == 4
    assert bool(model.label_prior_ready.item())
    adjusted = apply_label_prior(
        scores, model.label_prior, alpha=0.3, floor=math.exp(-12.0)
    )
    expected = scores - 0.3 * model.label_prior.log().view(1, 1, -1)
    torch.testing.assert_close(adjusted, expected)
