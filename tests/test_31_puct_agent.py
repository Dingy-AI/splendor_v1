import pytest
import torch

from splendor_v1.agents.neural_puct_agent import (
    NeuralPUCTAgent,
)
from splendor_v1.env.env import SplendorEnv
from splendor_v1.env.core.constants import OBSERVATION_SIZE
from splendor_v1.env.core.action_constants import ACTION_SPACE_SIZE
from splendor_v1.network.model import (
    SplendorNetwork,
)


@pytest.fixture
def env():
    env = SplendorEnv()
    env.reset()
    return env


@pytest.fixture
def model():
    return SplendorNetwork(
        OBSERVATION_SIZE,
        ACTION_SPACE_SIZE,
    )


def test_neural_puct_agent_returns_legal_action(
    env,
    model,
):
    agent = NeuralPUCTAgent(
        model=model,
        simulations=2,
    )

    action = agent.select_action(
        env,
        env.state,
    )

    legal_actions = env._legal_actions(
        env.state
    )

    assert action in legal_actions


def test_neural_puct_agent_uses_model(
    model,
):
    agent = NeuralPUCTAgent(
        model=model,
        simulations=2,
    )

    assert agent.model is model
    assert agent.mcts.model is model


def test_neural_puct_agent_uses_puct(
    model,
):
    agent = NeuralPUCTAgent(
        model=model,
        simulations=2,
    )

    assert agent.mcts.selection_type == "puct"
    assert agent.mcts.rollout_type == "neural"


def test_neural_puct_agent_respects_simulations(
    model,
):
    agent = NeuralPUCTAgent(
        model=model,
        simulations=7,
    )

    assert agent.mcts.simulations == 7