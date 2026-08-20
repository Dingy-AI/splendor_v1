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


from splendor_v1.env.core.constants import OBSERVATION_SIZE

@pytest.fixture
def env():
    return SplendorEnv()

def test_observation_size(env):

    env.reset()

    obs = env.observation_encoder.encoder(
        env.state
    )

    assert len(obs) == OBSERVATION_SIZE