import pytest
from splendor_v1.env.env import SplendorEnv
from splendor_v1.env.core.enums import GemColor, NodeType, ActionType, GemColor
from splendor_v1.env.core.card import Card

@pytest.fixture
def env():
    return SplendorEnv()

def test_legal_actions_count(env):
    env.reset()

    env.state.bank = {
        GemColor.WHITE: 3,
        GemColor.BLUE: 3,
        GemColor.GREEN: 3,
        GemColor.RED: 3,
        GemColor.BLACK: 4,
        GemColor.GOLD: 4
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
            GemColor.GOLD: 1
    }

    actions = env._legal_actions(env.state)

    print(len(actions))


    take_gems = 0
    discard_gems = 0
    actions_list = []
    for action in actions:

        if action.action_type == ActionType.TAKE_GEMS:
            take_gems += 1
        if action.action_type == ActionType.DISCARD_GEMS:
            discard_gems += 1

        if action.action_type == ActionType.BUY_VISIBLE and action.tier == 0 and action.slot == 0:
            actions_list.append(action)


    assert take_gems == 26
    assert discard_gems == 0

    print(actions_list)
    print(len(actions_list))