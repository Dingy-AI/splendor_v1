import math

import pytest
import torch
import torch.nn.functional as F

from splendor_v1.env.core.action_constants import ACTION_SPACE_SIZE
from splendor_v1.env.env import SplendorEnv
from splendor_v1.network.model_4_legal_scorer import (
    ACTION_EMBED_DIM,
    SplendorNetwork,
    WDL_SIZE,
)


def make_model():
    model = SplendorNetwork()
    model.eval()
    return model


def test_model4_uses_128_dim_action_embeddings():
    """
    Architecture contract for the first Model 4 experiment.
    """
    assert ACTION_EMBED_DIM == 128

    model = make_model()

    assert model.action_embedding.embedding_dim == 128
    assert model.padding_action_id == ACTION_SPACE_SIZE

    # +1 row is reserved for the padding action ID.
    assert (
        model.action_embedding.num_embeddings
        == ACTION_SPACE_SIZE + 1
    )


def test_single_state_dynamic_legal_action_shapes():
    """
    MCTS-style input:
        one observation
        one 1D list of legal action IDs
    """
    model = make_model()

    observations = torch.randn(
        1,
        258,
        dtype=torch.float32,
    )

    legal_action_ids = torch.tensor(
        [3, 17, 94, 221, 527, 801, 900],
        dtype=torch.long,
    )

    with torch.inference_mode():
        legal_logits, wdl_logits = model.forward_legal(
            observations,
            legal_action_ids,
        )

    assert legal_logits.shape == (
        1,
        len(legal_action_ids),
    )

    assert wdl_logits.shape == (
        1,
        WDL_SIZE,
    )

    assert torch.isfinite(
        legal_logits
    ).all()

    assert torch.isfinite(
        wdl_logits
    ).all()


def test_model4_supports_more_than_64_legal_actions():
    """
    There must be no hidden 64-action capacity limit.

    100 legal actions should simply produce 100 logits.
    """
    model = make_model()

    observations = torch.randn(
        1,
        258,
        dtype=torch.float32,
    )

    legal_action_ids = torch.arange(
        100,
        dtype=torch.long,
    )

    with torch.inference_mode():
        legal_logits, wdl_logits = model.forward_legal(
            observations,
            legal_action_ids,
        )

    assert legal_logits.shape == (
        1,
        100,
    )

    assert wdl_logits.shape == (
        1,
        WDL_SIZE,
    )


def test_dynamic_padding_mask_removes_padding_from_policy():
    """
    Different positions in one batch may have different numbers
    of legal actions.

    Padding must receive effectively zero softmax probability and
    must not steal probability mass from real legal actions.
    """
    model = make_model()

    pad = model.padding_action_id

    observations = torch.randn(
        3,
        258,
        dtype=torch.float32,
    )

    legal_action_ids = torch.tensor(
        [
            [3, 17, 94, pad, pad, pad],
            [8, 31, 99, 120, 527, pad],
            [1, 2, 3, 4, 5, 6],
        ],
        dtype=torch.long,
    )

    # Let Model 4 infer the mask from padding IDs.
    expected_mask = (
        legal_action_ids != pad
    )

    with torch.inference_mode():
        legal_logits, _ = model(
            observations,
            legal_action_ids,
        )

        probabilities = torch.softmax(
            legal_logits,
            dim=-1,
        )

    assert legal_logits.shape == (
        3,
        6,
    )

    assert probabilities.shape == (
        3,
        6,
    )

    # Every real-action distribution must still sum to 1.
    assert torch.allclose(
        probabilities.sum(
            dim=-1
        ),
        torch.ones(
            3,
            dtype=probabilities.dtype,
        ),
        atol=1e-6,
    )

    # Padding must receive no meaningful probability.
    padded_probabilities = probabilities[
        ~expected_mask
    ]

    assert torch.all(
        padded_probabilities
        <= 1e-7
    )

    # Each row must have positive probability somewhere among
    # its real actions.
    for row in range(
        legal_action_ids.shape[0]
    ):
        real_probs = probabilities[
            row
        ][
            expected_mask[row]
        ]

        assert torch.all(
            real_probs > 0
        )

        assert torch.allclose(
            real_probs.sum(),
            torch.tensor(
                1.0,
                dtype=real_probs.dtype,
            ),
            atol=1e-6,
        )


def test_explicit_legal_mask_matches_inferred_mask():
    """
    Supplying the mask explicitly should produce the same output
    as inferring it from padding_action_id.
    """
    model = make_model()

    pad = model.padding_action_id

    observations = torch.randn(
        2,
        258,
        dtype=torch.float32,
    )

    legal_action_ids = torch.tensor(
        [
            [10, 20, 30, pad],
            [40, 50, 60, 70],
        ],
        dtype=torch.long,
    )

    legal_action_mask = (
        legal_action_ids != pad
    )

    with torch.inference_mode():
        inferred_logits, inferred_wdl = model(
            observations,
            legal_action_ids,
        )

        explicit_logits, explicit_wdl = model(
            observations,
            legal_action_ids,
            legal_action_mask,
        )

    assert torch.allclose(
        inferred_logits,
        explicit_logits,
        atol=1e-7,
    )

    assert torch.allclose(
        inferred_wdl,
        explicit_wdl,
        atol=1e-7,
    )


def test_wdl_value_conversion_remains_valid():
    """
    Model 4 keeps Model 3's WDL semantics:
        value = P(WIN) - P(LOSS)
    """
    model = make_model()

    observations = torch.randn(
        4,
        258,
        dtype=torch.float32,
    )

    legal_action_ids = torch.tensor(
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12],
        ],
        dtype=torch.long,
    )

    with torch.inference_mode():
        _, wdl_logits = model(
            observations,
            legal_action_ids,
        )

        values = model.wdl_logits_to_value(
            wdl_logits
        )

        probabilities = model.wdl_probabilities(
            wdl_logits
        )

    assert values.shape == (
        4,
        1,
    )

    assert probabilities.shape == (
        4,
        WDL_SIZE,
    )

    assert torch.allclose(
        probabilities.sum(
            dim=-1
        ),
        torch.ones(
            4,
            dtype=probabilities.dtype,
        ),
        atol=1e-6,
    )

    assert torch.all(
        values >= -1.0
    )

    assert torch.all(
        values <= 1.0
    )


def test_model4_policy_and_wdl_backward_pass():
    """
    Verify that the new action embedding, shared legal scorer,
    attention trunk, and WDL head can all participate in training.
    """
    model = SplendorNetwork()
    model.train()

    pad = model.padding_action_id

    observations = torch.randn(
        3,
        258,
        dtype=torch.float32,
    )

    legal_action_ids = torch.tensor(
        [
            [3, 17, 94, pad, pad],
            [8, 31, 99, 120, pad],
            [1, 2, 3, 4, 5],
        ],
        dtype=torch.long,
    )

    legal_mask = (
        legal_action_ids != pad
    )

    # Visit-count-style policy targets, already normalized.
    target_policy = torch.tensor(
        [
            [0.10, 0.60, 0.30, 0.00, 0.00],
            [0.20, 0.10, 0.40, 0.30, 0.00],
            [0.05, 0.10, 0.15, 0.20, 0.50],
        ],
        dtype=torch.float32,
    )

    target_wdl = torch.tensor(
        [0, 2, 2],
        dtype=torch.long,
    )

    legal_logits, wdl_logits = model(
        observations,
        legal_action_ids,
        legal_mask,
    )

    log_probs = F.log_softmax(
        legal_logits,
        dim=-1,
    )

    policy_loss = -(
        target_policy
        * log_probs
    ).sum(
        dim=-1
    ).mean()

    wdl_loss = F.cross_entropy(
        wdl_logits,
        target_wdl,
    )

    total_loss = (
        policy_loss
        + wdl_loss
    )

    assert torch.isfinite(
        total_loss
    )

    total_loss.backward()

    # New Model 4 components must receive gradients.
    assert (
        model.action_embedding.weight.grad
        is not None
    )

    assert (
        model.legal_action_scorer[0].weight.grad
        is not None
    )

    assert (
        model.legal_action_scorer[-1].weight.grad
        is not None
    )

    # WDL head still trains.
    assert (
        model.wdl_head.weight.grad
        is not None
    )

    # Shared trunk must be connected during joint training.
    assert (
        model.final_norm.weight.grad
        is not None
    )

    # padding_idx should never learn.
    padding_grad = (
        model.action_embedding.weight.grad[
            model.padding_action_id
        ]
    )

    assert torch.allclose(
        padding_grad,
        torch.zeros_like(
            padding_grad
        ),
        atol=0.0,
    )


def test_real_environment_position_forward_legal():
    """
    End-to-end model-side smoke test using an actual Splendor state.

    This verifies:
        environment observation -> Model 4
        environment legal actions -> canonical IDs -> Model 4
    """
    env = SplendorEnv()
    env.reset(
        seed=0
    )

    state = env.state

    observation = (
        env.observation_encoder.encoder(
            state
        )
    )

    legal_actions = env._legal_actions(
        state
    )

    assert legal_actions

    legal_action_ids = torch.tensor(
        [
            env.action_to_id(
                action
            )
            for action in legal_actions
        ],
        dtype=torch.long,
    )

    observation_tensor = torch.as_tensor(
        observation,
        dtype=torch.float32,
    ).unsqueeze(
        0
    )

    model = make_model()

    with torch.inference_mode():
        legal_logits, wdl_logits = model.forward_legal(
            observation_tensor,
            legal_action_ids,
        )

        probabilities = torch.softmax(
            legal_logits,
            dim=-1,
        )

        value = model.wdl_logits_to_value(
            wdl_logits
        ).item()

    assert legal_logits.shape == (
        1,
        len(legal_actions),
    )

    assert probabilities.shape == (
        1,
        len(legal_actions),
    )

    assert torch.all(
        probabilities >= 0
    )

    assert torch.allclose(
        probabilities.sum(),
        torch.tensor(
            1.0,
            dtype=probabilities.dtype,
        ),
        atol=1e-6,
    )

    assert math.isfinite(
        value
    )

    assert -1.0 <= value <= 1.0


def test_invalid_1d_actions_for_multi_state_batch_rejected():
    """
    A 1D legal-action list is only unambiguous for one state.
    """
    model = make_model()

    observations = torch.randn(
        2,
        258,
        dtype=torch.float32,
    )

    legal_action_ids = torch.tensor(
        [1, 2, 3],
        dtype=torch.long,
    )

    with pytest.raises(
        ValueError,
        match="single observation",
    ):
        model(
            observations,
            legal_action_ids,
        )


def test_padding_cannot_be_marked_as_real_action():
    model = make_model()

    pad = model.padding_action_id

    observations = torch.randn(
        1,
        258,
        dtype=torch.float32,
    )

    legal_action_ids = torch.tensor(
        [[1, 2, pad]],
        dtype=torch.long,
    )

    bad_mask = torch.tensor(
        [[True, True, True]],
        dtype=torch.bool,
    )

    with pytest.raises(
        ValueError,
        match="Padding action ID",
    ):
        model(
            observations,
            legal_action_ids,
            bad_mask,
        )
