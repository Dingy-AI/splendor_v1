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
import numpy as np


from splendor_v1.env.core.constants import OBSERVATION_SIZE
from splendor_v1.network.model import SplendorNetwork
from splendor_v1.env.core.action_constants import ACTION_SPACE_SIZE

from splendor_v1.network.policy import apply_action_mask, get_policy_probs

@pytest.fixture
def env():
    return SplendorEnv()


@pytest.fixture
def model():
    return SplendorNetwork()

def test_apply_action_mask_sets_illegal_logits_to_negative_infinity():

    policy_logits = torch.tensor([
        [1.0, 2.0, 3.0, 4.0]
    ])

    action_mask = np.array(
        [1, 0, 1, 0],
        dtype=np.int8,
    )

    masked_logits = apply_action_mask(
        policy_logits,
        action_mask,
    )

    assert masked_logits[0, 0] == 1.0
    assert torch.isneginf(masked_logits[0, 1])

    assert masked_logits[0, 2] == 3.0
    assert torch.isneginf(masked_logits[0, 3])

def test_apply_action_mask_preserves_shape():

    policy_logits = torch.randn(
        1,
        ACTION_SPACE_SIZE,
    )

    action_mask = np.ones(
        ACTION_SPACE_SIZE,
        dtype=np.int8,
    )

    masked_logits = apply_action_mask(
        policy_logits,
        action_mask,
    )

    assert masked_logits.shape == (
        1,
        ACTION_SPACE_SIZE,
    )

def test_apply_action_mask_raises_if_no_legal_actions():

    policy_logits = torch.randn(
        1,
        ACTION_SPACE_SIZE,
    )

    action_mask = np.zeros(
        ACTION_SPACE_SIZE,
        dtype=np.int8,
    )

    with pytest.raises(ValueError):
        apply_action_mask(
            policy_logits,
            action_mask,
        )


def test_get_policy_probs_illegal_actions_are_zero():

    policy_logits = torch.tensor([
        [1.0, 2.0, 3.0, 4.0]
    ])

    action_mask = np.array(
        [1, 0, 1, 0],
        dtype=np.int8,
    )

    policy_probs = get_policy_probs(
        policy_logits,
        action_mask,
    )

    assert policy_probs[0, 1] == 0.0
    assert policy_probs[0, 3] == 0.0

    assert policy_probs[0, 0] > 0.0
    assert policy_probs[0, 2] > 0.0

def test_get_policy_probs_sum_to_one():

    policy_logits = torch.randn(
        1,
        ACTION_SPACE_SIZE,
    )

    action_mask = np.zeros(
        ACTION_SPACE_SIZE,
        dtype=np.int8,
    )

    # Arbitrarily make first 10 actions legal
    action_mask[:10] = 1

    policy_probs = get_policy_probs(
        policy_logits,
        action_mask,
    )

    total = policy_probs.sum(dim=-1)

    assert torch.allclose(
        total,
        torch.tensor([1.0]),
        atol=1e-6,
    )


def test_get_policy_probs_has_no_nan():

    policy_logits = torch.randn(
        1,
        ACTION_SPACE_SIZE,
    )

    action_mask = np.ones(
        ACTION_SPACE_SIZE,
        dtype=np.int8,
    )

    policy_probs = get_policy_probs(
        policy_logits,
        action_mask,
    )

    assert not torch.isnan(
        policy_probs
    ).any()

def test_get_policy_probs_preserves_shape():

    policy_logits = torch.randn(
        1,
        ACTION_SPACE_SIZE,
    )

    action_mask = np.ones(
        ACTION_SPACE_SIZE,
        dtype=np.int8,
    )

    policy_probs = get_policy_probs(
        policy_logits,
        action_mask,
    )

    assert policy_probs.shape == (
        1,
        ACTION_SPACE_SIZE,
    )

def test_full_policy_flow_with_real_environment(env, model):

    # -------------------------
    # Environment
    # -------------------------
    env.reset()

    state = env.state

    # -------------------------
    # Real observation
    # -------------------------
    obs = env.observation_encoder.encoder(
        state
    )

    assert len(obs) == OBSERVATION_SIZE

    obs_tensor = torch.as_tensor(
        obs,
        dtype=torch.float32,
    ).unsqueeze(0)

    assert obs_tensor.shape == (
        1,
        OBSERVATION_SIZE,
    )

    # -------------------------
    # Neural network
    # -------------------------

    policy_logits, value = model(
        obs_tensor
    )

    assert policy_logits.shape == (
        1,
        ACTION_SPACE_SIZE,
    )

    assert value.shape == (1, 1)

    # -------------------------
    # Real environment mask
    # -------------------------
    action_mask = env.action_mask(
        state
    )

    assert action_mask.shape == (
        ACTION_SPACE_SIZE,
    )

    assert action_mask.sum() > 0

    # -------------------------
    # Mask + softmax
    # -------------------------
    policy_probs = get_policy_probs(
        policy_logits,
        action_mask,
    )

    # Correct shape
    assert policy_probs.shape == (
        1,
        ACTION_SPACE_SIZE,
    )

    # Sum to 1
    assert torch.allclose(
        policy_probs.sum(dim=-1),
        torch.tensor([1.0]),
        atol=1e-6,
    )

    # No NaNs
    assert not torch.isnan(
        policy_probs
    ).any()

    # Illegal actions have exactly 0 probability
    illegal_ids = action_mask == 0

    assert torch.all(
        policy_probs[0][illegal_ids] == 0
    )

    # Legal actions have positive probability
    legal_ids = action_mask == 1

    assert torch.all(
        policy_probs[0][legal_ids] > 0
    )