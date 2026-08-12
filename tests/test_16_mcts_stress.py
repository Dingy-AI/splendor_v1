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

@pytest.mark.parametrize("seed", range(20))
def test_search_multiple_states(env, seed):

    random.seed(seed)

    env.reset()

    # Progress game slightly so you're not only testing initial state
    for _ in range(5):

        actions = env._legal_actions(env.state)

        if not actions:
            break

        env.step(
            random.choice(actions)
        )

    state_before = env.state.clone()

    mcts = MCTS(simulations=10)

    action = mcts.search(
        env,
        env.state
    )

    if env._legal_actions(env.state):
        assert action in env._legal_actions(env.state)

    assert env.state == state_before