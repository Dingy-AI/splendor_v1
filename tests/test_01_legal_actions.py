import pytest
from splendor_v1.env.env import SplendorEnv
from splendor_v1.env.core.enums import GemColor, NodeType, ActionType, GemColor
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
            Card(10000, 2, 1, GemColor.WHITE, {GemColor.WHITE: 5}), 
            Card(10000, 2, 1, GemColor.WHITE, {GemColor.WHITE: 5}),
            Card(10000, 2, 1, GemColor.WHITE, {GemColor.WHITE: 5})]

    actions = env._legal_actions(env.state)

    reserved_actions = 0
    for action in actions:
        if action.action_type == ActionType.RESERVE_TOP_DECK:
            reserved_actions += 1
        elif action.action_type == ActionType.RESERVE_VISIBLE:
            reserved_actions += 1
            
    assert reserved_actions == 0 

def test_buy_card(env):
    env.reset()

    env.state.players[0].gems = {GemColor.WHITE:5, GemColor.RED:5, GemColor.BLACK: 5, GemColor.GREEN:5}

    actions = env._legal_actions(env.state)
    num_buy_visible = 0
    for action in actions:
        if action.action_type == ActionType.BUY_VISIBLE:
            num_buy_visible += 1

    assert num_buy_visible > 0
    
    test_card = Card(1,1,5,GemColor.RED, cost={GemColor.WHITE: 1, GemColor.RED: 1, GemColor.GREEN:1, GemColor.BLACK: 1})


    is_affordable = env._can_afford(env.state.players[0], test_card)

    assert is_affordable == True

def test_buy_reserve_card(env):
    env.reset()
    env.state.players[0].gems = {GemColor.WHITE:5, GemColor.RED:5, GemColor.BLACK: 5, GemColor.GREEN:5}
    test_card = Card(1,1,5,GemColor.RED, cost={GemColor.WHITE: 1, GemColor.RED: 1, GemColor.GREEN:1, GemColor.BLACK: 1})
    env.state.players[0].reserved_cards = [test_card]
    actions = env._legal_actions(env.state)

    num_buy_reserve = 0
    for action in actions:
        if action.action_type == ActionType.BUY_RESERVED:
            num_buy_reserve += 1
    assert num_buy_reserve > 0

def test_reserve_limit(env):
    env.reset()
    test_card = Card(1,1,5,GemColor.RED, cost={GemColor.WHITE: 1, GemColor.RED: 1, GemColor.GREEN:1, GemColor.BLACK: 1})
    env.state.players[0].reserved_cards=[test_card, test_card, test_card]

    actions = env._legal_actions(env.state)

    num_reserve_cards = 0 
    for action in actions:
        if action.action_type == ActionType.BUY_RESERVED:
            num_reserve_cards += 1

    assert num_reserve_cards == 0


def test_gold_pile_empty(env):
    env.reset()

    env.state.bank[GemColor.GOLD] = 0

    actions = env._legal_actions(env.state)


    num_reserve_actions = 0
    for action in actions: 
        if action.action_type == ActionType.RESERVE_TOP_DECK or action.action_type == ActionType.RESERVE_VISIBLE:
            num_reserve_actions += 1

    assert num_reserve_actions == 15

def test_overflow_state(env):
    env.reset()

    env.state.node_type = NodeType.OVERFLOW_DISCARD
    env.state.players[0].gems = {GemColor.WHITE:5, GemColor.RED:5, GemColor.BLACK: 5, GemColor.GREEN:5}

    actions = env._legal_actions(env.state)

    assert len(actions) == 4

def test_noble_claim_state_false(env):
    env.reset()

    env.state.node_type = NodeType.NOBLE_CLAIM

    env.state.players[0].gems = {GemColor.WHITE:5, GemColor.RED:5, GemColor.BLACK: 5, GemColor.GREEN:5}

    actions = env._legal_actions(env.state)
    assert len(actions) == 0

def test_noble_claim_state_true(env):
    env.reset()

    env.state.node_type = NodeType.NOBLE_CLAIM

    env.state.players[0].bonuses = {GemColor.WHITE:5, GemColor.RED:5, GemColor.BLACK: 5, GemColor.GREEN:5, GemColor.BLUE: 5}

    actions = env._legal_actions(env.state)
    
    assert len(actions) > 1

def test_no_gem_in_bank(env):
    env.reset()

    env.state.bank[GemColor.WHITE] = 0
    env.state.bank[GemColor.BLUE] = 0
    env.state.bank[GemColor.GREEN] = 0
    env.state.bank[GemColor.RED] = 0
    env.state.bank[GemColor.BLACK] = 0
    env.state.bank[GemColor.GOLD] = 0
    

    actions = env._legal_actions(env.state)
    
    num_take_gems = 0
    num_reserve = 0
    for action in actions:
        if action.action_type == ActionType.TAKE_GEMS:
            num_take_gems += 1

        elif action.action_type == ActionType.RESERVE_TOP_DECK or action.action_type == ActionType.RESERVE_VISIBLE:
            num_reserve += 1

    assert num_take_gems == 0 
    assert num_reserve == 15

def test_two_of_a_kind_rule(env):
    env.reset()

    env.state.bank[GemColor.WHITE] = 3
    env.state.bank[GemColor.BLUE] = 3
    env.state.bank[GemColor.GREEN] = 3
    env.state.bank[GemColor.RED] = 3
    env.state.bank[GemColor.BLACK] = 3
    env.state.bank[GemColor.GOLD] = 3
    

    actions = env._legal_actions(env.state)
    
    num_take_gems = 0
    num_reserve = 0
    for action in actions:
        if action.action_type == ActionType.TAKE_GEMS:
            num_take_gems += 1

        elif action.action_type == ActionType.RESERVE_TOP_DECK or action.action_type == ActionType.RESERVE_VISIBLE:
            num_reserve += 1

    assert num_take_gems == 25
    assert num_reserve == 15
