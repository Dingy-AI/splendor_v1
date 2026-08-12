import pytest
from splendor_v1.env.env import SplendorEnv
from splendor_v1.env.core.enums import GemColor, NodeType, ActionType, GemColor
from splendor_v1.env.core.card import Card
from splendor_v1.env.core.actions import Action
from splendor_v1.env.core.action_constants import ACTION_SPACE_SIZE, DISCARD_START, NOBLE_START, DISCARD_COLORS
from copy import deepcopy
from splendor_v1.mcts.node import Node

from splendor_v1.mcts.mcts import MCTS
import math 

import random
@pytest.fixture
def env():
    return SplendorEnv()

@pytest.fixture
def mcts():
    return MCTS(simulations=10)

def test_search_returns_legal_action(env, mcts):

    env.reset()

    legal_actions = env._legal_actions(env.state)

    action = mcts.search(
        env,
        env.state
    )

    assert action is not None
    assert action in legal_actions

def test_search_does_not_mutate_environment(env, mcts):

    env.reset()

    original_state = env.state.clone()

    mcts.search(
        env,
        env.state
    )

    assert env.state == original_state

def test_search_expands_root(env, mcts):

    env.reset()

    action, root = mcts.search(
        env,
        env.state,
        return_root=True
    )

    assert len(root.children) > 0


def test_search_runs_requested_number_of_simulations(env, mcts):

    env.reset()
    action, root = mcts.search(
        env,
        env.state,
        return_root=True
    )

    assert root.visits == mcts.simulations

def test_search_expands_root(env, mcts):

    env.reset()

    action, root = mcts.search(
        env,
        env.state,
        return_root=True
    )

    assert len(root.children) > 0

def test_search_root_children_are_legal_actions(env, mcts):
    env.reset()

    legal_actions = env._legal_actions(env.state)


    action, root = mcts.search(
        env,
        env.state,
        return_root=True
    )

    for child in root.children:
        assert child.action in legal_actions


def test_search_returns_most_visited_child_action(env, mcts):
    env.reset()

    action, root = mcts.search(
        env,
        env.state,
        return_root=True
    )

    best_child = max(
        root.children,
        key=lambda child: child.visits
    )

    assert action == best_child.action