import math

import pytest
import torch

from splendor_v1.env.env import SplendorEnv
from splendor_v1.network.model_3_wdl_output import (
    SplendorNetwork,
    WDL_LOSS,
    WDL_DRAW,
    WDL_WIN,
)
from splendor_v1.network.losses_3_wdl import (
    policy_wdl_loss,
)
from splendor_v1.mcts.neural_evaluator_v3 import (
    neural_evaluate,
)
from splendor_v1.mcts.mcts_v3 import MCTS


@pytest.fixture
def env():
    env = SplendorEnv(num_players=2)
    env.reset(seed=12345)
    return env


@pytest.fixture
def model():
    torch.manual_seed(12345)

    model = SplendorNetwork()
    model.eval()

    return model


def test_model3_forward_shapes(env, model):
    """
    Model 3 should return:

        policy_logits: [batch, ACTION_SPACE_SIZE]
        wdl_logits:    [batch, 3]
    """

    observation = env.observation_encoder.encoder(
        env.state
    )

    obs_tensor = torch.as_tensor(
        observation,
        dtype=next(model.parameters()).dtype,
        device=next(model.parameters()).device,
    ).unsqueeze(0)

    with torch.inference_mode():
        policy_logits, wdl_logits = model(
            obs_tensor
        )

    assert policy_logits.ndim == 2
    assert policy_logits.shape[0] == 1

    assert wdl_logits.shape == (1, 3)

    assert torch.isfinite(
        policy_logits
    ).all()

    assert torch.isfinite(
        wdl_logits
    ).all()


def test_wdl_probabilities_and_scalar_value(env, model):
    """
    WDL logits should convert into probabilities that sum to 1,
    and the MCTS scalar value should remain inside [-1, +1].
    """

    observation = env.observation_encoder.encoder(
        env.state
    )

    obs_tensor = torch.as_tensor(
        observation,
        dtype=next(model.parameters()).dtype,
        device=next(model.parameters()).device,
    ).unsqueeze(0)

    with torch.inference_mode():
        _, wdl_logits = model(
            obs_tensor
        )

        wdl_probs = torch.softmax(
            wdl_logits,
            dim=-1,
        )

        value = model.wdl_logits_to_value(
            wdl_logits
        )

    assert wdl_probs.shape == (1, 3)

    assert torch.allclose(
        wdl_probs.sum(dim=-1),
        torch.ones(
            1,
            dtype=wdl_probs.dtype,
            device=wdl_probs.device,
        ),
        atol=1e-6,
    )

    assert value.shape == (1, 1)
    value_number = value.item()

    assert -1.0 <= value_number <= 1.0


def test_policy_wdl_loss_backward(env, model):
    """
    Verify that the new policy + WDL loss can run a full backward pass.
    """

    model.train()

    observation = env.observation_encoder.encoder(
        env.state
    )

    obs_tensor = torch.as_tensor(
        observation,
        dtype=next(model.parameters()).dtype,
        device=next(model.parameters()).device,
    ).unsqueeze(0)

    policy_logits, wdl_logits = model(
        obs_tensor
    )

    action_space_size = (
        policy_logits.shape[-1]
    )

    legal_actions = env._legal_actions(
        env.state
    )

    legal_action_ids = [
        env.action_to_id(action)
        for action in legal_actions
    ]

    target_policy = torch.zeros(
        (1, action_space_size),
        dtype=policy_logits.dtype,
        device=policy_logits.device,
    )

    # A valid normalized policy target over legal actions.
    probability = 1.0 / len(
        legal_action_ids
    )

    target_policy[
        0,
        legal_action_ids,
    ] = probability

    # Arbitrary valid WDL target for smoke testing.
    target_wdl = torch.tensor(
        [WDL_WIN],
        dtype=torch.long,
        device=wdl_logits.device,
    )

    (
        total_loss,
        policy_loss,
        value_loss,
        policy_kl,
    ) = policy_wdl_loss(
        policy_logits,
        wdl_logits,
        target_policy,
        target_wdl,
    )

    assert total_loss.ndim == 0
    assert policy_loss.ndim == 0
    assert value_loss.ndim == 0
    assert policy_kl.ndim == 0

    assert torch.isfinite(
        total_loss
    )

    model.zero_grad(
        set_to_none=True
    )

    total_loss.backward()

    found_gradient = False

    for parameter in model.parameters():

        if parameter.grad is None:
            continue

        found_gradient = True

        assert torch.isfinite(
            parameter.grad
        ).all()

    assert found_gradient


def test_neural_evaluator_v3(env, model):
    """
    Verify that the Model 3 evaluator:

        - scores only legal actions
        - returns normalized probabilities
        - converts WDL into a scalar MCTS value
    """

    model.eval()

    legal_actions = env._legal_actions(
        env.state
    )

    policy_probs, value = (
        neural_evaluate(
            env,
            model,
            env.state,
            legal_actions=legal_actions,
        )
    )

    assert policy_probs.ndim == 1

    assert len(policy_probs) == len(
        legal_actions
    )

    assert torch.isfinite(
        policy_probs
    ).all()

    assert torch.all(
        policy_probs >= 0
    )

    assert torch.allclose(
        policy_probs.sum(),
        torch.tensor(
            1.0,
            dtype=policy_probs.dtype,
            device=policy_probs.device,
        ),
        atol=1e-6,
    )

    assert math.isfinite(
        value
    )

    assert -1.0 <= value <= 1.0


def test_terminal_value_wdl_semantics(model):
    """
    Model 3 MCTS terminal convention:

        sole winner -> +1 for winner, -1 for loser
        shared result -> 0
        no winner -> 0
    """

    mcts = MCTS(
        simulations=1,
        rollout_type="neural",
        selection_type="puct",
        model=model,
    )

    class DummyState:
        def __init__(self, winners):
            self.winners = winners

    # Sole winner.
    state = DummyState(
        winners=[0]
    )

    assert mcts.terminal_value(
        state,
        root_player=0,
    ) == 1.0

    assert mcts.terminal_value(
        state,
        root_player=1,
    ) == -1.0

    # Shared result.
    state = DummyState(
        winners=[0, 1]
    )

    assert mcts.terminal_value(
        state,
        root_player=0,
    ) == 0.0

    assert mcts.terminal_value(
        state,
        root_player=1,
    ) == 0.0

    # Winnerless draw / deadlock.
    state = DummyState(
        winners=[]
    )

    assert mcts.terminal_value(
        state,
        root_player=0,
    ) == 0.0


def test_tiny_model3_mcts_search(env, model):
    """
    Run a very small real PUCT search through the Model 3 evaluator.

    This is the important end-to-end smoke test.
    """

    model.eval()

    legal_actions = env._legal_actions(
        env.state
    )

    assert legal_actions

    mcts = MCTS(
        simulations=5,
        rollout_type="neural",
        selection_type="puct",
        model=model,
        c_puct=3.0,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.25,
    )

    action, root = mcts.search(
        env,
        env.state,
        return_root=True,
        add_root_noise=False,
        teacher_mode=False,
    )

    assert action is not None

    assert action in legal_actions

    assert root.expanded

    assert root.children

    # Every expanded child should have a valid finite network prior.
    priors = []

    for child in root.children:

        assert child.network_prior is not None

        prior = float(
            child.network_prior
        )

        assert math.isfinite(
            prior
        )

        assert prior >= 0.0

        priors.append(
            prior
        )

    assert sum(
        priors
    ) == pytest.approx(
        1.0,
        abs=1e-5,
    )

    # A fresh search performs one backup per simulation.
    assert root.visits == 5

    assert math.isfinite(
        root.value
    )
