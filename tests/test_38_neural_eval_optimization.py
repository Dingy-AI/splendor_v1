import numpy as np
import torch

from splendor_v1.env.env import SplendorEnv
from splendor_v1.network.model import SplendorNetwork
from splendor_v1.mcts.neural_evaluator import (
    neural_evaluate,
    slow_neural_evaluate,
)
from splendor_v1.mcts.mcts import MCTS
from splendor_v1.mcts.node import Node

def test_neural_evaluate_matches_slow_version():

    torch.manual_seed(0)

    env = SplendorEnv()
    env.reset()

    state = env.state

    model = SplendorNetwork()

    legal_actions = env._legal_actions(
        state
    )

    slow_policy_probs, slow_value = (
        slow_neural_evaluate(
            env,
            model,
            state,
            legal_actions=legal_actions,
        )
    )

    fast_policy_probs, fast_value = (
        neural_evaluate(
            env,
            model,
            state,
            legal_actions=legal_actions,
        )
    )

    assert len(fast_policy_probs) == len(
        legal_actions
    )

    for i, action in enumerate(
        legal_actions
    ):
        action_id = env.action_to_id(
            action
        )

        assert torch.isclose(
            fast_policy_probs[i],
            slow_policy_probs[action_id],
            atol=1e-6,
        )

    assert np.isclose(
        fast_value,
        slow_value,
        atol=1e-6,
    )

def test_neural_evaluate_legal_probs_sum_to_one():

    torch.manual_seed(0)

    env = SplendorEnv()
    env.reset()

    state = env.state

    model = SplendorNetwork()

    legal_actions = env._legal_actions(
        state
    )

    policy_probs, _ = neural_evaluate(
        env,
        model,
        state,
        legal_actions=legal_actions,
    )

    assert torch.isclose(
        policy_probs.sum(),
        torch.tensor(1.0),
        atol=1e-6,
    )

def test_expand_all_with_priors_match_slow_policy():

    torch.manual_seed(0)

    env = SplendorEnv()
    env.reset()

    state = env.state

    model = SplendorNetwork()

    legal_actions = env._legal_actions(
        state
    )

    slow_policy_probs, _ = (
        slow_neural_evaluate(
            env,
            model,
            state,
            legal_actions=legal_actions,
        )
    )

    fast_policy_probs, _ = neural_evaluate(
        env,
        model,
        state,
        legal_actions=legal_actions,
    )

    for i, action in enumerate(
        legal_actions
    ):
        action_id = env.action_to_id(
            action
        )

        expected_prior = (
            slow_policy_probs[
                action_id
            ].item()
        )

        actual_prior = (
            fast_policy_probs[
                i
            ].item()
        )

        assert np.isclose(
            actual_prior,
            expected_prior,
            atol=1e-6,
        )

def test_expand_all_with_priors_matches_slow_version():

    torch.manual_seed(0)

    env = SplendorEnv()
    env.reset()

    state = env.state

    model = SplendorNetwork()

    mcts = MCTS(
        simulations=20,
        rollout_type="neural",
        selection_type="puct",
        model=model,
    )

    slow_root = Node(
        state=state.clone(),
    )

    fast_root = Node(
        state=state.clone(),
    )

    mcts.slow_expand_all_with_priors(
        env,
        slow_root,
    )

    mcts.expand_all_with_priors(
        env,
        fast_root,
    )

    assert slow_root.expanded
    assert fast_root.expanded

    assert len(slow_root.children) == len(
        fast_root.children
    )

    for slow_child, fast_child in zip(
        slow_root.children,
        fast_root.children,
    ):

        assert slow_child.action == fast_child.action

        assert np.isclose(
            slow_child.prior,
            fast_child.prior,
            atol=1e-6,
        )

def test_expand_all_with_priors_sum_to_one():

    torch.manual_seed(0)

    env = SplendorEnv()
    env.reset()

    state = env.state

    model = SplendorNetwork()

    mcts = MCTS(
        simulations=20,
        rollout_type="neural",
        selection_type="puct",
        model=model,
    )

    root = Node(
        state=state.clone(),
    )

    mcts.expand_all_with_priors(
        env,
        root,
    )

    total_prior = sum(
        child.prior
        for child in root.children
    )

    assert np.isclose(
        total_prior,
        1.0,
        atol=1e-6,
    )


def test_expand_all_with_priors_keeps_children_lazy():

    torch.manual_seed(0)

    env = SplendorEnv()
    env.reset()

    state = env.state

    model = SplendorNetwork()

    mcts = MCTS(
        simulations=20,
        rollout_type="neural",
        selection_type="puct",
        model=model,
    )

    root = Node(
        state=state.clone(),
    )

    mcts.expand_all_with_priors(
        env,
        root,
    )

    assert root.children

    for child in root.children:

        assert child.state is None
        assert child.parent is root
        assert child.action is not None