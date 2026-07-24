import pytest
from splendor_v1.env.env import SplendorEnv
from splendor_v1.env.core.enums import GemColor, NodeType, ActionType, GemColor
from splendor_v1.env.core.card import Card
from splendor_v1.env.core.actions import Action
from splendor_v1.env.core.action_constants import ACTION_SPACE_SIZE, DISCARD_START, NOBLE_START

@pytest.fixture
def env():
    return SplendorEnv()

def test_action_mask_size(env):

    env.reset()

    mask = env.action_mask(env.state)

    assert len(mask) == ACTION_SPACE_SIZE

def test_action_mask_default(env):
    env.reset()

    mask = env.action_mask(env.state)

    # Default clean game should have 45 choices
    assert sum(mask) == 45

def test_action_mask_illegal(env):
    env.reset()
    

    buy_action = Action(
            action_type=ActionType.BUY_VISIBLE,
            tier = 3,
            slot = 3,
        payment_id=0,
        gold_payment=(0,0,0,0,0)
    )

    action_id = env.action_to_id(buy_action)
    mask = env.action_mask(env.state)

    assert mask[action_id] == False 

def test_action_with_gems(env):
    env.reset()


    

    env.state.players[0].gems = {
        GemColor.WHITE: 5,
        GemColor.BLUE: 5,
        GemColor.GREEN: 5,
        GemColor.RED: 5,
        GemColor.BLACK: 5,
        GemColor.GOLD: 0
    }
    
    mask = env.action_mask(env.state)
    assert sum(mask) != 45

def test_overflow_discard(env):
    env.reset()

    env.state.node_type = NodeType.OVERFLOW_DISCARD

    env.state.players[0].gems = {
        GemColor.WHITE: 5,
        GemColor.BLUE: 5,
        GemColor.GREEN: 5,
        GemColor.RED: 5,
        GemColor.BLACK: 5,
        GemColor.GOLD: 0
    }
    mask = env.action_mask(env.state)

    mask_counter = 0
    for mask_action in mask:
        if mask_action != 0:
            break

        mask_counter += 1

    assert mask_counter == DISCARD_START


def test_noble_claim(env):
    env.reset()

    env.state.node_type = NodeType.NOBLE_CLAIM

    env.state.players[0].bonuses = {
        GemColor.WHITE: 5,
        GemColor.BLUE: 5,
        GemColor.GREEN: 5,
        GemColor.RED: 5,
        GemColor.BLACK: 5,
        GemColor.GOLD: 0
    }

    mask = env.action_mask(env.state)
    mask_counter = 0
    for mask_action in mask:
        if mask_action != 0:
            break

        mask_counter += 1

    assert mask_counter == NOBLE_START


# TODO: Need to do test what happens when the player is not the main player
# In theory it should still work but idk :)
