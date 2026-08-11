import pytest
from splendor_v1.env.env import SplendorEnv
from splendor_v1.env.core.enums import GemColor, NodeType, ActionType, GemColor
from splendor_v1.env.core.card import Card
from splendor_v1.env.core.actions import Action
from splendor_v1.env.core.action_constants import ACTION_SPACE_SIZE, DISCARD_START, NOBLE_START, DISCARD_COLORS
from copy import deepcopy


from splendor_v1.mcts.mcts import MCTS

import random
@pytest.fixture
def env():
    return SplendorEnv()

@pytest.fixture
def mcts(env):
    return MCTS(env)



def test_random_rollout_does_not_mutate(env, mcts):

    random.seed(1)

    env.reset()

    original = env.clone()
    try:

        reward = mcts.random_rollout(env)

    except Exception as e:
        pytest.fail(
                f"Exception during random mutate.\n"
                f"Exception: {e}"
        )


    assert env.state == original.state
