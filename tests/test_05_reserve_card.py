import pytest
from splendor_v1.env.env import SplendorEnv
from splendor_v1.env.core.enums import GemColor, NodeType, ActionType, GemColor
from splendor_v1.env.core.card import Card
from splendor_v1.env.core.actions import Action
from splendor_v1.env.core.action_constants import ACTION_SPACE_SIZE, DISCARD_START, NOBLE_START
from copy import deepcopy
@pytest.fixture
def env():
    return SplendorEnv()

def test_reserve_card(env):

    env.reset()
    reserved_card_copy = deepcopy(env.state.visible_cards[0][0])
    action = Action(

        action_type=ActionType.RESERVE_VISIBLE,
        tier=0,
        slot=0,
    )

    env.step(action)

    print(reserved_card_copy)
    print(env.state.visible_cards[0][0])

