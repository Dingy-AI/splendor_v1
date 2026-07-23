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

def test_buy_visible_card(env):
    env.reset()

    env.state.bank = {
        GemColor.WHITE: 3,
        GemColor.BLUE: 3,
        GemColor.GREEN: 3,
        GemColor.RED: 3,
        GemColor.BLACK: 4,
        GemColor.GOLD: 5
    }


    env.state.visible_cards[0][0] = Card(
        id=500,
        tier=0,
        points=2,
        bonus_color=GemColor.RED,
        cost={ 
            GemColor.WHITE: 1,
            GemColor.BLUE: 1,
            GemColor.RED: 1,
            GemColor.GREEN: 1,
            GemColor.BLACK:0
        }
    )

    env.state.players[0].gems ={ 
            GemColor.WHITE: 1,
            GemColor.BLUE: 1,
            GemColor.RED: 1,
            GemColor.GREEN: 1,
            GemColor.BLACK: 0,
            GemColor.GOLD: 0
    }


    action = Action(
        action_type = ActionType.BUY_VISIBLE,
        tier=0,
        slot=0,
        payment_id=0,
        gold_payment=(0,0,0,0,0)
    )

    env.step(action)


    assert sum(env.state.players[0].gems.values()) == 0
    assert env.state.players[0].bonuses[GemColor.RED] == 1
    assert env.state.players[0].points == 2

    assert sum(env.state.bank.values()) == 25
    assert env.state.visible_cards[0][0].id != 500
    #TODO running into an issue with buying a card. 
    # you have to also 'choose' which gems you want to get rid depending on if you have gold or not
    
    assert env.state.players[0].purchased_cards[0].id == 500