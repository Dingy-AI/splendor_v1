import pytest
from splendor_v1.env.env import SplendorEnv
from splendor_v1.env.core.enums import GemColor, NodeType, ActionType, GemColor
from splendor_v1.env.core.card import Card
from splendor_v1.env.core.actions import Action
from splendor_v1.env.core.action_constants import ACTION_SPACE_SIZE

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
            slot = 3
    )

    action_id = env.action_to_id(buy_action)
    mask = env.action_mask(env.state)

    assert mask[action_id] == False 
        
    
        

# TODO: Need to do test what happens when the player is not the main player
# In theory it should still work but idk :)
