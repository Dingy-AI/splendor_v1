import pytest
from splendor_v1.env.env import SplendorEnv
from splendor_v1.env.core.enums import GemColor, NodeType, ActionType, GemColor
from splendor_v1.env.core.card import Card

@pytest.fixture
def env():
    return SplendorEnv()

def test_action_to_id(env):

    env.reset()
    actions = env._legal_actions(env.state)

    for action in actions:


        action_id = env.action_to_id(action)
        recovered_action = env.id_to_action(action_id)
        print("action id:", action_id)

        print("action:", action)
        print("recovered action:", recovered_action)
        assert action == recovered_action