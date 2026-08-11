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
def mcts(env):
    return MCTS(env)


def test_ucb_unvisited_child_is_infinite(env, mcts):

    parent = Node(state=None)
    parent.visits = 10

    child = Node(state=None, parent=parent)
    child.visits = 0

    score = mcts.ucb_score(parent, child)

    assert score == float("inf")

def test_ucb_prefers_higher_average_value_when_visits_equal(env, mcts):

    parent = Node(state=None)
    parent.visits = 20

    child_a = Node(state=None, parent=parent)
    child_a.visits = 10
    child_a.value_sum = 8.0   # average = 0.8

    child_b = Node(state=None, parent=parent)
    child_b.visits = 10
    child_b.value_sum = 4.0   # average = 0.4

    score_a = mcts.ucb_score(parent, child_a)
    score_b = mcts.ucb_score(parent, child_b)

    assert score_a > score_b

def test_ucb_prefers_less_visited_child_when_values_equal(env, mcts):

    parent = Node(state=None)
    parent.visits = 100

    child_a = Node(state=None, parent=parent)
    child_a.visits = 50
    child_a.value_sum = 25.0   # average = 0.5

    child_b = Node(state=None, parent=parent)
    child_b.visits = 5
    child_b.value_sum = 2.5    # average = 0.5

    score_a = mcts.ucb_score(parent, child_a)
    score_b = mcts.ucb_score(parent, child_b)

    assert score_b > score_a


def test_ucb_exact_value(env, mcts):

    parent = Node(state=None)
    parent.visits = 100

    child = Node(state=None, parent=parent)
    child.visits = 10
    child.value_sum = 6.0

    c = math.sqrt(2)

    expected = (
        6.0 / 10
        + c * math.sqrt(math.log(100) / 10)
    )

    score = mcts.ucb_score(
        parent,
        child,
        exploration_constant=c
    )

    assert math.isclose(score, expected)

def test_select_returns_root_if_root_has_untried_actions(env, mcts):

    root = Node(state=None)
    root.untried_actions = ["action"]

    selected = mcts.select(root)

    assert selected is root

def test_select_returns_dead_end_node(env, mcts):

    root = Node(state=None)
    root.untried_actions = []
    root.children = []

    selected = mcts.select(root)

    assert selected is root

def test_select_chooses_highest_ucb_child(env, mcts):

    root = Node(state=None)
    root.visits = 100
    root.untried_actions = []

    child_a = Node(state=None, parent=root)
    child_a.visits = 50
    child_a.value_sum = 10
    child_a.untried_actions = ["expand_me"]

    child_b = Node(state=None, parent=root)
    child_b.visits = 50
    child_b.value_sum = 40
    child_b.untried_actions = ["expand_me"]

    root.children = [child_a, child_b]

    selected = mcts.select(root)

    assert selected is child_b

def test_select_walks_multiple_levels(mcts):

    root = Node(state=None)
    root.visits = 100
    root.untried_actions = []

    child = Node(state=None, parent=root)
    child.visits = 50
    child.value_sum = 30
    child.untried_actions = []

    grandchild = Node(state=None, parent=child)
    grandchild.visits = 10
    grandchild.value_sum = 8
    grandchild.untried_actions = ["new_action"]

    root.children = [child]
    child.children = [grandchild]

    selected = mcts.select(root)

    assert selected is grandchild