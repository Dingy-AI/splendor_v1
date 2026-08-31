# tests/network/test_losses.py

import torch

from splendor_v1.network.losses import policy_value_loss


def test_policy_value_loss_returns_scalars():

    policy_logits = torch.tensor([
        [1.0, 2.0, 3.0]
    ])

    predicted_value = torch.tensor([
        [0.5]
    ])

    target_policy = torch.tensor([
        [0.0, 0.0, 1.0]
    ])

    target_value = torch.tensor([
        1.0
    ])

    total_loss, policy_loss, value_loss = policy_value_loss(
        policy_logits,
        predicted_value,
        target_policy,
        target_value,
    )

    assert total_loss.dim() == 0
    assert policy_loss.dim() == 0
    assert value_loss.dim() == 0


def test_total_loss_equals_policy_plus_value():

    policy_logits = torch.tensor([
        [1.0, 2.0, 3.0]
    ])

    predicted_value = torch.tensor([
        [0.25]
    ])

    target_policy = torch.tensor([
        [0.2, 0.3, 0.5]
    ])

    target_value = torch.tensor([
        1.0
    ])

    total_loss, policy_loss, value_loss = policy_value_loss(
        policy_logits,
        predicted_value,
        target_policy,
        target_value,
    )

    assert torch.allclose(
        total_loss,
        policy_loss + value_loss,
    )


def test_value_loss_is_zero_when_prediction_matches_target():

    policy_logits = torch.tensor([
        [1.0, 1.0]
    ])

    predicted_value = torch.tensor([
        [1.0]
    ])

    target_policy = torch.tensor([
        [0.5, 0.5]
    ])

    target_value = torch.tensor([
        1.0
    ])

    _, _, value_loss = policy_value_loss(
        policy_logits,
        predicted_value,
        target_policy,
        target_value,
    )

    assert torch.allclose(
        value_loss,
        torch.tensor(0.0),
    )


def test_value_loss_matches_expected_mse():

    policy_logits = torch.tensor([
        [1.0, 1.0]
    ])

    predicted_value = torch.tensor([
        [0.5]
    ])

    target_policy = torch.tensor([
        [0.5, 0.5]
    ])

    target_value = torch.tensor([
        -0.5
    ])

    _, _, value_loss = policy_value_loss(
        policy_logits,
        predicted_value,
        target_policy,
        target_value,
    )

    # (0.5 - -0.5)^2 = 1.0
    assert torch.allclose(
        value_loss,
        torch.tensor(1.0),
    )


def test_policy_loss_prefers_correct_action():

    target_policy = torch.tensor([
        [0.0, 0.0, 1.0]
    ])

    predicted_value = torch.tensor([
        [0.0]
    ])

    target_value = torch.tensor([
        0.0
    ])

    good_logits = torch.tensor([
        [0.0, 0.0, 5.0]
    ])

    bad_logits = torch.tensor([
        [5.0, 0.0, 0.0]
    ])

    _, good_policy_loss, _ = policy_value_loss(
        good_logits,
        predicted_value,
        target_policy,
        target_value,
    )

    _, bad_policy_loss, _ = policy_value_loss(
        bad_logits,
        predicted_value,
        target_policy,
        target_value,
    )

    assert good_policy_loss < bad_policy_loss


def test_policy_loss_with_uniform_logits_and_uniform_target():

    policy_logits = torch.tensor([
        [0.0, 0.0, 0.0, 0.0]
    ])

    target_policy = torch.tensor([
        [0.25, 0.25, 0.25, 0.25]
    ])

    predicted_value = torch.tensor([
        [0.0]
    ])

    target_value = torch.tensor([
        0.0
    ])

    _, policy_loss, _ = policy_value_loss(
        policy_logits,
        predicted_value,
        target_policy,
        target_value,
    )

    expected = torch.log(
        torch.tensor(4.0)
    )

    assert torch.allclose(
        policy_loss,
        expected,
        atol=1e-6,
    )


def test_loss_handles_batch():

    policy_logits = torch.tensor([
        [1.0, 2.0, 3.0],
        [3.0, 2.0, 1.0],
        [0.0, 1.0, 0.0],
    ])

    predicted_value = torch.tensor([
        [1.0],
        [-0.5],
        [0.0],
    ])

    target_policy = torch.tensor([
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.2, 0.6, 0.2],
    ])

    target_value = torch.tensor([
        1.0,
        -1.0,
        0.0,
    ])

    total_loss, policy_loss, value_loss = policy_value_loss(
        policy_logits,
        predicted_value,
        target_policy,
        target_value,
    )

    assert total_loss.dim() == 0
    assert policy_loss.dim() == 0
    assert value_loss.dim() == 0

    assert torch.isfinite(total_loss)
    assert torch.isfinite(policy_loss)
    assert torch.isfinite(value_loss)


def test_loss_supports_backward_pass():

    policy_logits = torch.randn(
        4,
        10,
        requires_grad=True,
    )

    predicted_value = torch.randn(
        4,
        1,
        requires_grad=True,
    )

    target_policy = torch.softmax(
        torch.randn(4, 10),
        dim=-1,
    )

    target_value = torch.tensor([
        1.0,
        -1.0,
        0.0,
        1.0,
    ])

    total_loss, _, _ = policy_value_loss(
        policy_logits,
        predicted_value,
        target_policy,
        target_value,
    )

    total_loss.backward()

    assert policy_logits.grad is not None
    assert predicted_value.grad is not None

    assert torch.isfinite(
        policy_logits.grad
    ).all()

    assert torch.isfinite(
        predicted_value.grad
    ).all()