# tests/mcts/test_neural_evaluator.py

import torch

from splendor_v1.env.env import SplendorEnv
from splendor_v1.env.core.action_constants import ACTION_SPACE_SIZE
from splendor_v1.network.model import SplendorNetwork
from splendor_v1.mcts.neural_evaluator import neural_evaluate


def test_neural_evaluate_returns_policy_and_value():

    env = SplendorEnv()
    env.reset()

    model = SplendorNetwork()

    policy_probs, value = neural_evaluate(
        env,
        model,
        env.state,
    )

    assert policy_probs.shape == (
        ACTION_SPACE_SIZE,
    )

    assert isinstance(value, float)


def test_neural_evaluate_value_is_bounded():

    env = SplendorEnv()
    env.reset()

    model = SplendorNetwork()

    _, value = neural_evaluate(
        env,
        model,
        env.state,
    )

    assert -1.0 <= value <= 1.0


def test_neural_evaluate_policy_sums_to_one():

    env = SplendorEnv()
    env.reset()

    model = SplendorNetwork()

    policy_probs, _ = neural_evaluate(
        env,
        model,
        env.state,
    )

    assert torch.allclose(
        policy_probs.sum(),
        torch.tensor(1.0),
        atol=1e-6,
    )


def test_neural_evaluate_has_no_nan():

    env = SplendorEnv()
    env.reset()

    model = SplendorNetwork()

    policy_probs, value = neural_evaluate(
        env,
        model,
        env.state,
    )

    assert not torch.isnan(
        policy_probs
    ).any()

    assert not torch.isnan(
        torch.tensor(value)
    )


def test_neural_evaluate_illegal_actions_have_zero_probability():

    env = SplendorEnv()
    env.reset()

    state = env.state

    model = SplendorNetwork()

    policy_probs, _ = neural_evaluate(
        env,
        model,
        state,
    )

    action_mask = env.action_mask(
        state
    )

    illegal_ids = action_mask == 0

    assert torch.all(
        policy_probs[illegal_ids] == 0
    )


def test_neural_evaluate_legal_actions_have_positive_probability():

    env = SplendorEnv()
    env.reset()

    state = env.state

    model = SplendorNetwork()

    policy_probs, _ = neural_evaluate(
        env,
        model,
        state,
    )

    action_mask = env.action_mask(
        state
    )

    legal_ids = action_mask == 1

    assert torch.all(
        policy_probs[legal_ids] > 0
    )