import pytest
from splendor_v1.env.env import SplendorEnv
from splendor_v1.env.core.enums import GemColor, NodeType, ActionType, GemColor
from splendor_v1.env.core.card import Card
from splendor_v1.env.core.action_constants import ACTION_SPACE_SIZE, BUY_T1_START, BUY_T2_START, BUY_T3_START,BUY_RESERVED_START, ACTION_END

from copy import deepcopy
@pytest.fixture
def env():
    return SplendorEnv()

def test_action_to_id_round_trip(env):

    env.reset()
    actions = env._legal_actions(env.state)

    for action in actions:


        action_id = env.action_to_id(action)
        recovered_action = env.id_to_action(action_id)
        assert action == recovered_action
        






def test_id_to_action_round_trip(env):
    env.reset()
    
    action = env.id_to_action(5)
    recovered = env.action_to_id(action)


    assert 5 == recovered 

def test_invalid_action_id(env):
    env.reset()
    with pytest.raises(ValueError):
        env.id_to_action(-1)

#tested take gems and reserve
# TODO: have not tested buy, pick noble, discard, 
# save this part of testing for later

def test_id_to_action_round_trip_all(env):
    env.reset()



    for i in range(ACTION_SPACE_SIZE):
        action_copy = deepcopy(env.id_to_action(i))
        recovered_id = env.action_to_id(action_copy)
        assert recovered_id == i

def test_id_to_action_to_id_round_trip_all(env):
    env.reset()
    for i in range(ACTION_SPACE_SIZE):
        action_copy = deepcopy(env.id_to_action(i))
        recovered_id = env.action_to_id(action_copy)

        assert recovered_id == i


def test_action_buy(env):
    env.reset()

    action = env.id_to_action(BUY_T1_START)

    assert action.action_type == ActionType.BUY_VISIBLE
    assert action.tier == 1
    assert action.payment_id == 0

    
    action = env.id_to_action(BUY_T1_START+1)
    assert action.action_type == ActionType.BUY_VISIBLE
    assert action.tier == 1
    assert action.payment_id == 1
    
    action = env.id_to_action(BUY_T2_START)
    assert action.action_type == ActionType.BUY_VISIBLE
    assert action.tier == 2
    assert action.payment_id == 0
    

    action = env.id_to_action(BUY_T3_START)
    assert action.action_type == ActionType.BUY_VISIBLE
    assert action.tier == 3
    assert action.payment_id == 0

def test_action_buy_reserved(env):
    env.reset()

    action = env.id_to_action(BUY_RESERVED_START)

    assert action.action_type == ActionType.BUY_RESERVED
    assert action.tier == None
    assert action.payment_id == 0
    assert action.reserved_index == 0

    action = env.id_to_action(BUY_RESERVED_START+222)

    assert action.action_type == ActionType.BUY_RESERVED
    assert action.tier == None
    assert action.payment_id == 0
    assert action.reserved_index == 2

def test_check_action_size_equal(env):
    assert ACTION_SPACE_SIZE == ACTION_END