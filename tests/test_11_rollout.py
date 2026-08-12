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
def mcts():
    return MCTS()

@pytest.fixture
def node(env):
    return Node(state = env.state)
    



def test_random_rollout_does_not_mutate(env, mcts, node):

    random.seed(1)

    env.reset()
    node.state = env.state
    root_player = env.state.current_player

    original = env.clone()
    try:

        reward = mcts.random_rollout(env, node, root_player)

    except Exception as e:
        pytest.fail(
                f"Exception during random mutate.\n"
                f"Exception: {e}"
        )


    assert env.state == original.state

def test_random_rollout_reward_range(env, mcts, node):
    env.reset()
    node.state = env.state
    root_player = env.state.current_player

    for _ in range(10):
        reward = mcts.random_rollout(env, node, root_player)

        assert reward in [-1,0,1]

def test_rollout_finishes_game(env, mcts, node):
    env.reset()
    node.state = env.state
    root_player = env.state.current_player

    final_state = mcts.random_rollout(env, node, root_player, return_state=True)

    assert final_state.game_over == True

def test_random_rollout_many_games(env, mcts, node):
    # did not have an issue with 100
    for seed in range(10):
        random.seed(seed)

        env.reset()
        root_player = env.state.current_player

        node.state = env.state

        reward = mcts.random_rollout(env, node, root_player)

        assert reward is not None