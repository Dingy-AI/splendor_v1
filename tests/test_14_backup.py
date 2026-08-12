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
    return MCTS()

def test_backup_updates_single_node(env, mcts):

    node = Node(state=None)

    mcts.backup(node, 1.0)

    assert node.visits == 1
    assert node.value == 1.0

def test_backup_propagates_to_root(env, mcts):

    root = Node(state=None)

    child = Node(
        state=None,
        parent=root
    )

    grandchild = Node(
        state=None,
        parent=child
    )

    mcts.backup(grandchild, 1.0)

    assert grandchild.visits == 1
    assert grandchild.value == 1.0

    assert child.visits == 1
    assert child.value == 1.0

    assert root.visits == 1
    assert root.value == 1.0

def test_backup_zero_value_still_updates_visits(mcts):

    root = Node(state=None)

    child = Node(
        state=None,
        parent=root
    )

    mcts.backup(child, 0.0)

    assert child.visits == 1
    assert child.value == 0.0

    assert root.visits == 1
    assert root.value == 0.0

def test_backup_negative_value(mcts):

    root = Node(state=None)

    child = Node(
        state=None,
        parent=root
    )

    mcts.backup(child, -1.0)

    assert child.visits == 1
    assert child.value == -1.0

    assert root.visits == 1
    assert root.value == -1.0

def test_backup_accumulates_multiple_results(mcts):

    root = Node(state=None)

    child = Node(
        state=None,
        parent=root
    )

    mcts.backup(child, 1.0)
    mcts.backup(child, 0.0)
    mcts.backup(child, -1.0)

    assert child.visits == 3
    assert child.value == 0.0

    assert root.visits == 3
    assert root.value == 0.0


def test_backup_does_not_update_siblings(mcts):

    root = Node(state=None)

    child_a = Node(state=None, parent=root)
    child_b = Node(state=None, parent=root)

    root.children = [child_a, child_b]

    mcts.backup(child_a, 1.0)

    assert child_a.visits == 1
    assert child_a.value == 1.0

    assert child_b.visits == 0
    assert child_b.value == 0.0

    assert root.visits == 1
    assert root.value == 1.0