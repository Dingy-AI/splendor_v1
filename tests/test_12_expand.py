import pytest
from splendor_v1.env.env import SplendorEnv
from splendor_v1.env.core.enums import GemColor, NodeType, ActionType, GemColor
from splendor_v1.env.core.card import Card
from splendor_v1.env.core.actions import Action
from splendor_v1.env.core.action_constants import ACTION_SPACE_SIZE, DISCARD_START, NOBLE_START, DISCARD_COLORS
from copy import deepcopy
from splendor_v1.mcts.node import Node

from splendor_v1.mcts.mcts import MCTS

import random
@pytest.fixture
def env():
    return SplendorEnv()

@pytest.fixture
def mcts(env):
    return MCTS(env)

def test_expand_creates_child(env, mcts):
    env.reset()

    
    root = Node(
        state=env.state.clone(),
        untried_actions=env._legal_actions(env.state)
    )
    original_state = root.state.clone()

    original_untried = list(root.untried_actions)
    child = mcts.expand(env, root)

    assert child is not None
    assert len(root.children) == 1

    assert root.children[0] == child

    assert child.parent == root

    assert child.action in original_untried
    assert root.state == original_state

def test_expand_child_state_matches_action(env, mcts):
    env.reset()
    root = Node(
        state=env.state.clone(),
        untried_actions=env._legal_actions(env.state)
    )
    original_state = root.state.clone()
    mcts.expand(env, root)

    assert root.state == original_state 

def test_expand_child_state_matches_action(env, mcts):
    env.reset()
    root = Node(
        state=env.state.clone(),
        untried_actions=env._legal_actions(env.state)
    )
    child = mcts.expand(env, root)

    check_env = env.clone()
    check_env.state = root.state.clone()

    check_env.step(child.action)

    assert child.state == check_env.state


def test_expand_removes_untried_action(env, mcts):

    env.reset()
    root = Node(
        state=env.state.clone(),
        untried_actions=env._legal_actions(env.state)
    )

    before = len(root.untried_actions)

    mcts.expand(env, root)

    assert len(root.untried_actions) == before - 1

def test_expand_uses_unique_actions(env, mcts):
    env.reset()
    root = Node(
        state=env.state.clone(),
        untried_actions=env._legal_actions(env.state)
    )

    child1 = mcts.expand(env, root)
    child2 = mcts.expand(env, root)

    assert child1.action != child2.action
    assert len(root.children) == 2

def test_expanded_child_starts_unvisited(env, mcts):
    env.reset()
    root = Node(
        state=env.state.clone(),
        untried_actions=env._legal_actions(env.state)
    )

    child = mcts.expand(env, root)

    assert child.visits == 0
    assert child.value == 0

def test_child_state_is_independent(env, mcts):

    env.reset()


    root = Node(
        state=env.state.clone(),
        untried_actions=env._legal_actions(env.state)
    )

    child = mcts.expand(env, root)

    assert child.state is not root.state
def test_expand_returns_none_when_fully_expanded(env, mcts):

    env.reset()


    root = Node(
        state=env.state.clone(),
        untried_actions=env._legal_actions(env.state)
    )

    while root.untried_actions:
        mcts.expand(env, root)

    assert mcts.expand(env, root) is None