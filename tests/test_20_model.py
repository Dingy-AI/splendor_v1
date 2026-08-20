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
import torch

from splendor_v1.env.core.constants import OBSERVATION_SIZE
from splendor_v1.network.model import SplendorNetwork
from splendor_v1.env.core.action_constants import ACTION_SPACE_SIZE

@pytest.fixture
def env():
    return SplendorEnv()


def test_forward_batch():

    model = SplendorNetwork()

    batch_size = 32

    x = torch.randn(
        batch_size,
        OBSERVATION_SIZE
    )

    policy_logits, values = model(x)

    assert policy_logits.shape == (
        batch_size,
        ACTION_SPACE_SIZE
    )

    assert values.shape == (
        batch_size,
        1
    )
