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


@pytest.fixture
def model():
    return SplendorNetwork()


def test_forward_batch(model):

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

def test_forward_single_observation(model):

    x = torch.randn(
        1,
        OBSERVATION_SIZE
    )

    policy_logits, values = model(x)

    assert policy_logits.shape == (
        1,
        ACTION_SPACE_SIZE
    )

    assert values.shape == (1, 1)

def test_value_is_bounded(model):
    x = torch.randn(
        32,
        OBSERVATION_SIZE
    )

    _, values = model(x)

    assert torch.all(values >= -1.0)
    assert torch.all(values <= 1.0)

def test_forward_has_no_nan(model):

    x = torch.randn(
        32,
        OBSERVATION_SIZE
    )

    policy_logits, values = model(x)

    assert not torch.isnan(policy_logits).any()
    assert not torch.isnan(values).any()


def test_forward_real_observation(env, model):

    env.reset()

    obs = env.observation_encoder.encoder(
        env.state
    )

    x = torch.tensor(
        obs,
        dtype=torch.float32
    ).unsqueeze(0)

    policy_logits, value = model(x)

    assert x.shape == (1, OBSERVATION_SIZE)
    assert policy_logits.shape == (1, ACTION_SPACE_SIZE)
    assert value.shape == (1, 1)