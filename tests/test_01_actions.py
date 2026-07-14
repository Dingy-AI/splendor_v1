import pytest
from splendor_v1.env.env import SplendorEnv
from splendor_v1.env.core.enums import GemColor, NodeType, ActionType, CardColor
from splendor_v1.env.core.player import Player
from splendor_v1.env.core.card import Card

@pytest.fixture
def env():
    return SplendorEnv()

def test_first_turn_actions(env):
    state = env.reset()
    actions = env._legal_actions(env.state)
    buy_visible = 0
    buy_reserved = 0
    reserve_visible = 0
    reserve_top_deck = 0
    take_gems = 0
    discard_gems = 0
    take_noble = 0
    
    for action in actions:
        if action.action_type == ActionType.RESERVE_VISIBLE:
            reserve_visible += 1
        elif action.action_type == ActionType.RESERVE_TOP_DECK:
            reserve_top_deck += 1
        elif action.action_type == ActionType.BUY_RESERVED:
            buy_reserved += 1
        elif action.action_type == ActionType.BUY_VISIBLE:
            buy_visible += 1
        elif action.action_type == ActionType.TAKE_GEMS:
            take_gems += 1
        elif action.action_type == ActionType.DISCARD_GEMS:
            discard_gems += 1
        elif action.action_type == ActionType.TAKE_NOBLE:
            take_noble += 1

    assert buy_reserved == 0
    assert buy_visible == 0
    assert reserve_visible == 12
    assert reserve_top_deck == 3
    assert take_gems == 30
    assert discard_gems == 0
    assert take_noble == 0    

    # print(actions)
    # print(len(actions))

def test_no_white_gem_action(env):
    env.reset()

    # print(env.state.bank)
    env.state.bank[GemColor.WHITE] = 0
    actions = env._legal_actions(env.state)
    # print(env.state.bank)

    # print((actions))
    # print(len(actions))

    non_white_actions = 0
    for action in actions:
        if action.action_type == ActionType.TAKE_GEMS:
            if GemColor.WHITE not in action.gem_colors:
                non_white_actions += 1
            
    assert non_white_actions == 18

def test_no_gold_action(env):
    env.reset()

    env.state.bank[GemColor.GOLD] = 0
    actions = env._legal_actions(env.state)
    assert len(actions) == 45
#overflow state
#noble claim state
#player qualifies for two nobles 

def test_reserve_limit_reached(env):
    env.reset()
    env.state.players[0].reserved_cards = [
            Card(10000, 2, 1, CardColor.WHITE, {CardColor.WHITE: 5}), 
            Card(10000, 2, 1, CardColor.WHITE, {CardColor.WHITE: 5}),
            Card(10000, 2, 1, CardColor.WHITE, {CardColor.WHITE: 5})]

    actions = env._legal_actions(env.state)

    reserved_actions = 0
    for action in actions:
        if action.action_type == ActionType.RESERVE_TOP_DECK:
            reserved_actions += 1
        elif action.action_type == ActionType.RESERVE_VISIBLE:
            reserved_actions += 1
            
    assert reserved_actions == 0 
