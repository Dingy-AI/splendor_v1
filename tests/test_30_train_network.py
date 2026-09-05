import numpy as np
import torch
import pytest


from splendor_v1.env.env import SplendorEnv
from splendor_v1.network.model import SplendorNetwork
from splendor_v1.training.replay_buffer import ReplayBuffer
from splendor_v1.training.train import train_network
from splendor_v1.env.core.constants import OBSERVATION_SIZE
from splendor_v1.env.core.action_constants import ACTION_SPACE_SIZE
from splendor_v1.training.train import run_training


from splendor_v1.mcts.node import Node
from splendor_v1.training.self_play import select_self_play_action

@pytest.fixture
def env():
    return SplendorEnv()

def make_replay_buffer(num_samples=16):
    replay_buffer = ReplayBuffer()

    for _ in range(num_samples):

        observation = np.zeros(
            OBSERVATION_SIZE,
            dtype=np.float32,
        )

        target_policy = np.zeros(
            ACTION_SPACE_SIZE,
            dtype=np.float32,
        )
        target_policy[0] = 1.0

        target_value = 1.0

        replay_buffer.add(
            (observation,
            target_policy,
            target_value)
        )

    return replay_buffer

def test_train_network_updates_parameters():

    model = SplendorNetwork(
        OBSERVATION_SIZE,
        ACTION_SPACE_SIZE,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    replay_buffer = make_replay_buffer()

    parameters_before = [
        parameter.detach().clone()
        for parameter in model.parameters()
    ]

    train_network(
        model=model,
        replay_buffer=replay_buffer,
        optimizer=optimizer,
        batch_size=8,
        training_steps=1,
    )

    parameters_after = list(
        model.parameters()
    )

    assert any(
        not torch.equal(before, after)
        for before, after in zip(
            parameters_before,
            parameters_after,
        )
    )

def test_train_network_returns_valid_losses():

    model = SplendorNetwork(
        OBSERVATION_SIZE,
        ACTION_SPACE_SIZE,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    replay_buffer = make_replay_buffer()

    stats = train_network(
        model=model,
        replay_buffer=replay_buffer,
        optimizer=optimizer,
        batch_size=8,
        training_steps=2,
    )

    assert "loss" in stats
    assert "policy_loss" in stats
    assert "value_loss" in stats

    assert np.isfinite(stats["loss"])
    assert np.isfinite(
        stats["policy_loss"]
    )
    assert np.isfinite(
        stats["value_loss"]
    )

def test_train_network_requires_enough_samples():

    model = SplendorNetwork(
        OBSERVATION_SIZE,
        ACTION_SPACE_SIZE,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    replay_buffer = make_replay_buffer(
        num_samples=4
    )

    with pytest.raises(ValueError):
        train_network(
            model=model,
            replay_buffer=replay_buffer,
            optimizer=optimizer,
            batch_size=8,
            training_steps=1,
        )

def test_run_training_completes_one_iteration():

    env = SplendorEnv()

    model = SplendorNetwork(
        OBSERVATION_SIZE,
        ACTION_SPACE_SIZE,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    replay_buffer = ReplayBuffer()

    history = run_training(
        env=env,
        model=model,
        optimizer=optimizer,
        replay_buffer=replay_buffer,
        num_iterations=1,
        self_play_games_per_iteration=1,
        simulations=2,
        batch_size=1,
        training_steps=1,
    )

    assert len(history) == 1

def test_run_training_returns_valid_loss_stats():

    env = SplendorEnv()

    model = SplendorNetwork(
        OBSERVATION_SIZE,
        ACTION_SPACE_SIZE,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    replay_buffer = ReplayBuffer()

    history = run_training(
        env=env,
        model=model,
        optimizer=optimizer,
        replay_buffer=replay_buffer,
        num_iterations=1,
        self_play_games_per_iteration=1,
        simulations=2,
        batch_size=1,
        training_steps=1,
    )

    stats = history[0]

    assert "loss" in stats
    assert "policy_loss" in stats
    assert "value_loss" in stats

    assert np.isfinite(stats["loss"])
    assert np.isfinite(stats["policy_loss"])
    assert np.isfinite(stats["value_loss"])

def test_run_training_adds_self_play_data():

    env = SplendorEnv()

    model = SplendorNetwork(
        OBSERVATION_SIZE,
        ACTION_SPACE_SIZE,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    replay_buffer = ReplayBuffer()

    assert len(replay_buffer) == 0

    run_training(
        env=env,
        model=model,
        optimizer=optimizer,
        replay_buffer=replay_buffer,
        num_iterations=1,
        self_play_games_per_iteration=1,
        simulations=2,
        batch_size=1,
        training_steps=1,
    )

    assert len(replay_buffer) > 0

def make_root_with_visits(
    state,
    actions,
    visits,
):
    root = Node(state=state)

    for action, visit_count in zip(
        actions,
        visits,
    ):
        child = Node(
            state=state.clone(),
            parent=root,
            action=action,
            visits=visit_count,
        )

        root.children.append(child)

    return root

def test_self_play_action_temperature_zero_picks_most_visited(
    env,
):
    env.reset()

    actions = env._legal_actions(
        env.state
    )[:3]

    root = make_root_with_visits(
        env.state,
        actions,
        [2, 10, 5],
    )

    selected_action = (
        select_self_play_action(
            root,
            temperature=0,
        )
    )

    assert selected_action == actions[1]

def test_self_play_action_is_from_root_children(env):
    env.reset()

    actions = env._legal_actions(
        env.state
    )[:3]

    root = make_root_with_visits(
        env.state,
        actions,
        [6, 3, 1],
    )

    for _ in range(100):

        selected_action = (
            select_self_play_action(
                root,
                temperature=1.0,
            )
        )

        assert selected_action in actions

def test_self_play_action_sampling_follows_visit_distribution(
    env,
):
    env.reset()

    actions = env._legal_actions(
        env.state
    )[:2]

    root = make_root_with_visits(
        env.state,
        actions,
        [9, 1],
    )

    counts = {
        actions[0]: 0,
        actions[1]: 0,
    }

    for _ in range(1000):

        selected_action = (
            select_self_play_action(
                root,
                temperature=1.0,
            )
        )

        counts[selected_action] += 1

    assert counts[actions[0]] > counts[actions[1]]


def test_self_play_action_raises_with_no_children(
    env,
):
    env.reset()

    root = Node(
        state=env.state.clone()
    )

    with pytest.raises(ValueError):
        select_self_play_action(
            root,
            temperature=1.0,
        )

def test_self_play_action_raises_for_negative_temperature(
    env,
):
    env.reset()

    actions = env._legal_actions(
        env.state
    )[:1]

    root = make_root_with_visits(
        env.state,
        actions,
        [10],
    )

    with pytest.raises(ValueError):
        select_self_play_action(
            root,
            temperature=-1.0,
        )