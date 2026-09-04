# tests/mcts/test_neural_mcts.py

import torch

from splendor_v1.env.env import SplendorEnv
from splendor_v1.network.model import SplendorNetwork
from splendor_v1.mcts.mcts import MCTS
from splendor_v1.env.core.action_constants import ACTION_SPACE_SIZE


def test_neural_mcts_requires_model():

    try:
        MCTS(
            simulations=5,
            rollout_type="neural",
            model=None,
        )

        assert False, (
            "Expected ValueError when "
            "neural rollout has no model"
        )

    except ValueError:
        pass


def test_neural_mcts_search_returns_action():

    env = SplendorEnv()
    env.reset()

    model = SplendorNetwork()

    mcts = MCTS(
        simulations=5,
        rollout_type="neural",
        model=model,
    )

    action = mcts.search(
        env,
        env.state,
    )

    assert action is not None


def test_neural_mcts_returns_legal_action():

    env = SplendorEnv()
    env.reset()

    state = env.state

    model = SplendorNetwork()

    mcts = MCTS(
        simulations=5,
        rollout_type="neural",
        model=model,
    )

    action = mcts.search(
        env,
        state,
    )

    legal_actions = env._legal_actions(
        state
    )

    assert action in legal_actions


def test_neural_mcts_search_updates_root_visits():

    env = SplendorEnv()
    env.reset()

    model = SplendorNetwork()

    simulations = 5

    mcts = MCTS(
        simulations=simulations,
        rollout_type="neural",
        model=model,
    )

    action, root = mcts.search(
        env,
        env.state,
        return_root=True,
    )

    assert action is not None

    assert root.visits == simulations


def test_neural_mcts_children_receive_visits():

    env = SplendorEnv()
    env.reset()

    model = SplendorNetwork()

    mcts = MCTS(
        simulations=10,
        rollout_type="neural",
        model=model,
    )

    _, root = mcts.search(
        env,
        env.state,
        return_root=True,
    )

    total_child_visits = sum(
        child.visits
        for child in root.children
    )

    assert total_child_visits > 0

def test_neural_rollout_flips_value_for_opponent(
    monkeypatch,
):

    env = SplendorEnv()
    env.reset()

    model = SplendorNetwork()

    mcts = MCTS(
        simulations=1,
        rollout_type="neural",
        model=model,
    )

    root_player = env.state.current_player

    child_state = env.state.clone()

    child_state.current_player = (
        1 - root_player
    )

    class FakeChild:
        state = child_state
        legal_actions = None        

    def fake_neural_evaluate(
        env,
        model,
        state,
        legal_actions
    ):
        policy = torch.zeros(
            ACTION_SPACE_SIZE
        )

        return policy, 0.75

    monkeypatch.setattr(
        "splendor_v1.mcts.mcts.neural_evaluate",
        fake_neural_evaluate,
    )

    value = mcts.rollout(
        env,
        FakeChild(),
        root_player,
    )

    assert value == -0.75

def test_neural_rollout_keeps_value_for_root_player(
    monkeypatch,
):

    env = SplendorEnv()
    env.reset()

    model = SplendorNetwork()

    mcts = MCTS(
        simulations=1,
        rollout_type="neural",
        model=model,
    )

    root_player = env.state.current_player

    child_state = env.state.clone()
    child_state.current_player = root_player

    class FakeChild:
        state = child_state
        legal_actions = None

    def fake_neural_evaluate(
        env,
        model,
        state,
        legal_actions
    ):
        policy = torch.zeros(
            ACTION_SPACE_SIZE
        )

        return policy, 0.75

    monkeypatch.setattr(
        "splendor_v1.mcts.mcts.neural_evaluate",
        fake_neural_evaluate,
    )

    value = mcts.rollout(
        env,
        FakeChild(),
        root_player,
    )

    assert value == 0.75