import pytest
from splendor_v1.env.env import SplendorEnv
from splendor_v1.env.core.enums import GemColor, NodeType, ActionType, GemColor
from splendor_v1.env.core.card import Card
from splendor_v1.env.core.actions import Action
from splendor_v1.env.core.action_constants import ACTION_SPACE_SIZE, DISCARD_START, NOBLE_START

@pytest.fixture
def env():
    return SplendorEnv()

def test_take_one_gem(env):
    
    env.reset()

    action = Action(

        action_type=ActionType.TAKE_GEMS,
        gem_colors=(GemColor.WHITE,)
    )

    obs, reward, terminated, truncated, info= env.step(action)


    assert env.state.bank[GemColor.WHITE] == 3
    assert env.state.players[0].gems[GemColor.WHITE] == 1

