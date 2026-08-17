import torch

from objective import adamer_objectives


def test_stop_gradient_separates_model_and_beta_objectives():
    ctc_loss = torch.tensor(3.0, requires_grad=True)
    entropy = torch.tensor([1.0, 2.0], requires_grad=True)
    target = torch.tensor([1.5, 1.5])
    beta = torch.tensor(0.2, requires_grad=True)

    enctc_loss, beta_loss, _ = adamer_objectives(
        ctc_loss, entropy, target, beta
    )

    model_grads = torch.autograd.grad(
        enctc_loss, (ctc_loss, entropy, beta), allow_unused=True, retain_graph=True
    )
    assert torch.equal(model_grads[0], torch.tensor(1.0))
    assert torch.allclose(model_grads[1], torch.full((2,), -0.2))
    assert model_grads[2] is None

    beta_grads = torch.autograd.grad(
        beta_loss, (entropy, beta), allow_unused=True
    )
    assert beta_grads[0] is None
    assert torch.equal(beta_grads[1], torch.tensor(0.0))


def test_beta_gradient_direction_tracks_entropy_constraint():
    beta = torch.tensor(0.2, requires_grad=True)
    ctc_loss = torch.tensor(0.0)

    _, high_entropy_beta_loss, _ = adamer_objectives(
        ctc_loss,
        path_entropy=torch.tensor([3.0]),
        target_entropy=torch.tensor([2.0]),
        beta=beta,
    )
    high_gradient = torch.autograd.grad(high_entropy_beta_loss, beta)[0]
    assert high_gradient > 0

    _, low_entropy_beta_loss, _ = adamer_objectives(
        ctc_loss,
        path_entropy=torch.tensor([1.0]),
        target_entropy=torch.tensor([2.0]),
        beta=beta,
    )
    low_gradient = torch.autograd.grad(low_entropy_beta_loss, beta)[0]
    assert low_gradient < 0


if __name__ == "__main__":
    test_stop_gradient_separates_model_and_beta_objectives()
    test_beta_gradient_direction_tracks_entropy_constraint()
