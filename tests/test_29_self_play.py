import numpy as np

from splendor_v1.env.env import SplendorEnv
from splendor_v1.mcts.mcts import MCTS
from splendor_v1.mcts.node import Node
from splendor_v1.network.model import SplendorNetwork
from splendor_v1.training.replay_buffer import ReplayBuffer
from splendor_v1.training.self_play import (
    root_visit_policy,
    play_self_play_game,
)

from splendor_v1.env.core.constants import OBSERVATION_SIZE

from splendor_v1.env.core.action_constants import ACTION_SPACE_SIZE

def test_root_visit_policy_shape():

    env = SplendorEnv()
    env.reset()

    legal_actions = env._legal_actions(
        env.state
    )

    root = Node(
        state=env.state.clone()
    )

    for action in legal_actions[:3]:

        child = Node(
            state=env.state.clone(),
            parent=root,
            action=action,
            visits=1,
        )

        root.children.append(child)

    target_policy = root_visit_policy(
        env,
        root,
    )

    assert target_policy.shape == (
        ACTION_SPACE_SIZE,
    )

def test_root_visit_policy_sums_to_one():

    env = SplendorEnv()
    env.reset()

    legal_actions = env._legal_actions(
        env.state
    )

    root = Node(
        state=env.state.clone()
    )

    visits = [5, 3, 2]

    for action, visit_count in zip(
        legal_actions[:3],
        visits,
    ):

        child = Node(
            state=env.state.clone(),
            parent=root,
            action=action,
            visits=visit_count,
        )

        root.children.append(child)

    target_policy = root_visit_policy(
        env,
        root,
    )

    assert np.isclose(
        target_policy.sum(),
        1.0,
    )

def test_root_visit_policy_matches_visit_distribution():

    env = SplendorEnv()
    env.reset()

    legal_actions = env._legal_actions(
        env.state
    )

    root = Node(
        state=env.state.clone()
    )

    visits = [6, 3, 1]

    for action, visit_count in zip(
        legal_actions[:3],
        visits,
    ):

        child = Node(
            state=env.state.clone(),
            parent=root,
            action=action,
            visits=visit_count,
        )

        root.children.append(child)

    target_policy = root_visit_policy(
        env,
        root,
    )

    action_ids = [
        env.action_to_id(action)
        for action in legal_actions[:3]
    ]

    assert np.isclose(
        target_policy[action_ids[0]],
        0.6,
    )

    assert np.isclose(
        target_policy[action_ids[1]],
        0.3,
    )

    assert np.isclose(
        target_policy[action_ids[2]],
        0.1,
    )

def test_root_visit_policy_unvisited_actions_are_zero():

    env = SplendorEnv()
    env.reset()

    legal_actions = env._legal_actions(
        env.state
    )

    root = Node(
        state=env.state.clone()
    )

    selected_action = legal_actions[0]

    root.children.append(
        Node(
            state=env.state.clone(),
            parent=root,
            action=selected_action,
            visits=10,
        )
    )

    target_policy = root_visit_policy(
        env,
        root,
    )

    selected_id = env.action_to_id(
        selected_action
    )

    assert target_policy[selected_id] == 1.0

    assert np.count_nonzero(
        target_policy
    ) == 1

def test_root_visit_policy_raises_when_no_visits():

    env = SplendorEnv()
    env.reset()

    root = Node(
        state=env.state.clone()
    )

    legal_actions = env._legal_actions(
        env.state
    )

    root.children.append(
        Node(
            state=env.state.clone(),
            parent=root,
            action=legal_actions[0],
            visits=0,
        )
    )

    try:

        root_visit_policy(
            env,
            root,
        )

        assert False

    except ValueError:
        pass

import pytest


def test_root_visit_policy_raises_when_no_visits():

    env = SplendorEnv()
    env.reset()

    root = Node(
        state=env.state.clone()
    )

    with pytest.raises(ValueError):
        root_visit_policy(
            env,
            root,
        )

def test_self_play_adds_examples_to_replay_buffer():

    env = SplendorEnv()

    model = SplendorNetwork()

    mcts = MCTS(
        simulations=5,
        rollout_type="neural",
        selection_type="puct",
        model=model,
    )

    replay_buffer = ReplayBuffer()

    assert len(replay_buffer) == 0

    play_self_play_game(
        env,
        mcts,
        replay_buffer,
    )

    assert len(replay_buffer) > 0

def test_self_play_examples_have_correct_structure():

    env = SplendorEnv()

    model = SplendorNetwork()

    mcts = MCTS(
        simulations=3,
        rollout_type="neural",
        selection_type="puct",
        model=model,
    )

    replay_buffer = ReplayBuffer()

    play_self_play_game(
        env,
        mcts,
        replay_buffer,
    )

    assert len(replay_buffer) > 0

    for (
        observation,
        target_policy,
        target_value,
    ) in replay_buffer.buffer:

        assert len(observation) == (
            OBSERVATION_SIZE
        )

        assert target_policy.shape == (
            ACTION_SPACE_SIZE,
        )

        assert np.isclose(
            target_policy.sum(),
            1.0,
        )

        assert target_value in (
            -1.0,
            0.0,
            1.0,
        )

def test_self_play_target_policies_are_valid_probabilities():

    env = SplendorEnv()

    model = SplendorNetwork()

    mcts = MCTS(
        simulations=3,
        rollout_type="neural",
        selection_type="puct",
        model=model,
    )

    replay_buffer = ReplayBuffer()

    play_self_play_game(
        env,
        mcts,
        replay_buffer,
    )

    for (
        _,
        target_policy,
        _,
    ) in replay_buffer.buffer:

        assert np.all(
            target_policy >= 0.0
        )

        assert np.all(
            target_policy <= 1.0
        )

        assert np.isclose(
            target_policy.sum(),
            1.0,
        )

def test_self_play_contains_winner_and_loser_targets():

    env = SplendorEnv()

    model = SplendorNetwork()

    mcts = MCTS(
        simulations=3,
        rollout_type="neural",
        selection_type="puct",
        model=model,
    )

    replay_buffer = ReplayBuffer()

    play_self_play_game(
        env,
        mcts,
        replay_buffer,
    )

    target_values = [
        example[2]
        for example in replay_buffer.buffer
    ]

    assert all(
        value in (-1.0, 0.0, 1.0)
        for value in target_values
    )

def test_self_play_replay_buffer_examples_are_finite():

    env = SplendorEnv()

    model = SplendorNetwork()

    mcts = MCTS(
        simulations=20,
        rollout_type="neural",
        selection_type="puct",
        model=model,
    )

    replay_buffer = ReplayBuffer()

    play_self_play_game(
        env,
        mcts,
        replay_buffer,
    )

    for (
        observation,
        target_policy,
        target_value,
    ) in replay_buffer.buffer:

        assert np.all(
            np.isfinite(observation)
        )

        assert np.all(
            np.isfinite(target_policy)
        )

        assert np.isfinite(
            target_value
        )