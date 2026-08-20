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
from splendor_v1.mcts.rollout import heuristic_action,heuristic_rollout_v2, heuristic_value

@pytest.fixture
def env():
    return SplendorEnv()



def test_heuristic_rollout_returns_valid_value(env):

    env.reset()

    child = Node(
        state=env.state.clone()
    )

    value = heuristic_rollout_v2(
        env,
        child,
        root_player=child.state.current_player,
        max_steps=10,
    )

    assert -1.0 <= value <= 1.0

def test_heuristic_rollout_does_not_mutate_child_state(env):

    env.reset()

    child = Node(
        state=env.state.clone()
    )

    original = child.state.clone()

    heuristic_rollout_v2(
        env,
        child,
        root_player=child.state.current_player,
        max_steps=10,
    )

    assert child.state == original

def test_heuristic_rollout_many_seeds(env):

    for seed in range(20):

        random.seed(seed)
        env.reset()

        child = Node(
            state=env.state.clone()
        )

        value = heuristic_rollout_v2(
            env,
            child,
            root_player=child.state.current_player,
            max_steps=20,
        )

        assert -1.0 <= value <= 1.0

def test_heuristic_rollout_uses_fallback_value(env):

    env.reset()

    child = Node(
        state=env.state.clone()
    )

    value = heuristic_rollout_v2(
        env,
        child,
        root_player=child.state.current_player,
        max_steps=1,
    )

    assert -1.0 <= value <= 1.0