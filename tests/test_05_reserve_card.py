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
    reserved_card_copy = deepcopy(env.state.visible_cards[1][0])
    action = Action(

        action_type=ActionType.RESERVE_VISIBLE,
        tier=1,
        slot=0,
    )

    obs, reward, terminated, truncated, info = env.step(action)

    assert reward == 0 
    assert (reserved_card_copy != env.state.visible_cards[1][0])

    assert len(env.state.players[0].reserved_cards) == 1
    assert env.state.players[0].gems[GemColor.GOLD] == 1
    assert env.state.players[0].reserved_cards[0] == reserved_card_copy

    assert env.state.current_player == 1
    assert env.state.players[0].gems[GemColor.GOLD] == 1
    assert sum(env.state.players[0].gems.values()) == 1


def test_reserve_card_top_deck(env):
    env.reset()
    reserved_card_copy = deepcopy(env.state.decks[1][-1])
    action = Action(

        action_type=ActionType.RESERVE_TOP_DECK,
        tier=1
    )

    env.step(action)

    assert (reserved_card_copy != env.state.visible_cards[1][0])

    assert env.state.players[0].reserved_cards[0] == reserved_card_copy
    
    assert env.state.current_player == 1
    assert env.state.players[0].gems[GemColor.GOLD] == 1
    assert sum(env.state.players[0].gems.values()) == 1

def test_no_gold_available(env):

    env.reset()
    reserved_card_copy = deepcopy(env.state.visible_cards[1][0])

    env.state.bank[GemColor.GOLD] = 0
    action = Action(

        action_type=ActionType.RESERVE_VISIBLE,
        tier=1,
        slot=0,
    )
    env.step(action)

    assert env.state.players[0].reserved_cards[0] == reserved_card_copy
    assert env.state.players[0].gems[GemColor.GOLD] == 0

def test_reserve_card_limit(env):
    #TODO #Might not be necessary to implement
    pass

def test_empty_deck(env):
    #TODO Need to implement this in the future as a check. Currently it is fine
    pass

def test_gold_overflow(env):
    env.reset()
    action = Action(

        action_type=ActionType.RESERVE_VISIBLE,
        tier=1,
        slot=0,
    )

    env.state.players[0].gems = {
        GemColor.WHITE: 2,
        GemColor.BLUE: 2,
        GemColor.GREEN: 2,
        GemColor.RED: 2,
        GemColor.BLACK: 2,
        GemColor.GOLD: 0
    }

    env.step(action)

    assert env.state.node_type == NodeType.OVERFLOW_DISCARD
    assert env.state.current_player == 0
    assert sum(env.state.players[0].gems.values()) == 11