import pytest
from splendor_v1.env.env import SplendorEnv
from splendor_v1.env.core.enums import GemColor, NodeType, ActionType, GemColor
from splendor_v1.env.core.card import Card

@pytest.fixture
def env():
    return SplendorEnv()

def test_action_to_id_round_trip(env):

    env.reset()
    actions = env._legal_actions(env.state)

    for action in actions:


        action_id = env.action_to_id(action)
        recovered_action = env.id_to_action(action_id)
        assert action == recovered_action

def test_id_to_action_round_trip(env):
    env.reset()
    
    action = env.id_to_action(5)

    recovered = env.action_to_id(action)

    assert 5 == recovered 

def test_invalid_action_id(env):
    env.reset()
    with pytest.raises(ValueError):
        env.id_to_action(-1)

