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

def test_player_encoding_size_with_reserved_card():

    env = SplendorEnv()
    env.reset()

    player = env.state.players[0]

    # Give player a real card however your tests
    # normally construct/reserve one.

    features = env.observation_encoder._encode_single_player(
        player
    )

    assert len(features) == 45

def test_observation_size_across_many_states():

    for seed in range(5):

        random.seed(seed)

        env = SplendorEnv()
        env.reset()

        state = env.state

        for _ in range(200):

            obs = env.observation_encoder.encoder(
                state
            )

            assert len(obs) == OBSERVATION_SIZE, (
                f"Seed {seed}: "
                f"got {len(obs)} features"
            )

            actions = env._legal_actions(state)

            if not actions:
                break

            action = random.choice(actions)

            _, _, terminated, truncated, _ = env.step(
                action,
                state=state,
            )

            if terminated or truncated:
                break