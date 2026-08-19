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
from splendor_v1.mcts.rollout import heuristic_action,heuristic_rollout, heuristic_value

@pytest.fixture
def env():
    return SplendorEnv()

def test_heuristic_action_prefers_buy():

    actions = [
        Action(action_type=ActionType.TAKE_GEMS),
        Action(action_type=ActionType.RESERVE_VISIBLE),
        Action(action_type=ActionType.BUY_VISIBLE),
    ]

    action = heuristic_action(
        state=None,
        actions=actions,
    )

    assert action.action_type == ActionType.BUY_VISIBLE

def test_heuristic_action_prefers_noble_when_no_buy():

    actions = [
        Action(action_type=ActionType.TAKE_GEMS),
        Action(action_type=ActionType.TAKE_NOBLE),
    ]

    action = heuristic_action(
        state=None,
        actions=actions,
    )

    assert action.action_type == ActionType.TAKE_NOBLE

def test_heuristic_action_uses_discard_when_required():

    actions = [
        Action(action_type=ActionType.DISCARD_GEMS),
    ]

    action = heuristic_action(
        state=None,
        actions=actions,
    )

    assert action.action_type == ActionType.DISCARD_GEMS

def test_heuristic_value_positive_when_root_ahead(env):

    env.reset()

    env.state.players[0].points = 10
    env.state.players[1].points = 5

    value = heuristic_value(
        env.state,
        root_player=0,
    )

    assert value > 0


def test_heuristic_value_negative_when_root_behind(env):
    env.reset()

    env.state.players[0].points = 4
    env.state.players[1].points = 9

    value = heuristic_value(
        env.state,
        root_player=0,
    )

    assert value < 0


def test_heuristic_value_is_clamped(env):
    env.reset()

    env.state.players[0].points = 100
    env.state.players[1].points = 0

    value = heuristic_value(
        env.state,
        root_player=0,
    )

    assert -1.0 <= value <= 1.0

def test_heuristic_rollout_does_not_mutate_child_state(env):

    env.reset()

    child = Node(
        state=env.state.clone()
    )

    original = child.state.clone()

    heuristic_rollout(
        env,
        child,
        root_player=child.state.current_player,
        max_steps=10,
    )

    assert child.state == original

def test_heuristic_rollout_returns_valid_value(env):
    env.reset()

    child = Node(
        state=env.state.clone()
    )

    value = heuristic_rollout(
        env,
        child,
        root_player=child.state.current_player,
        max_steps=10,
    )

    assert -1.0 <= value <= 1.0

import random
import pytest


@pytest.mark.parametrize("seed", range(5))
def test_heuristic_rollout_many_seeds(seed, env):

    random.seed(seed)

    env.reset()

    child = Node(
        state=env.state.clone()
    )

    value = heuristic_rollout(
        env,
        child,
        root_player=child.state.current_player,
        max_steps=20,
    )

    assert -1.0 <= value <= 1.0