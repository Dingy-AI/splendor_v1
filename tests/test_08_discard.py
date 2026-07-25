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

def test_discard(env):
    env.reset()

    env.state.players[0].gems = {
        GemColor.WHITE: 3,
        GemColor.BLACK: 0,
        GemColor.GREEN: 3,
        GemColor.BLUE: 0,
        GemColor.RED: 0,
        GemColor.GOLD: 3
    }


    action = Action (

        action_type=ActionType.TAKE_GEMS,
        gem_colors=(GemColor.GREEN, GemColor.BLUE, GemColor.RED)
    )

    env.step(action)

    assert env.state.node_type == NodeType.OVERFLOW_DISCARD
    assert env.state.current_player == 0
    assert sum(env.state.players[0].gems.values()) == 12

    actions = env._legal_actions(env.state)


    num_discard_gems = 0

    for action in actions:
        assert action.action_type == ActionType.DISCARD_GEMS
        num_discard_gems += 1

    assert num_discard_gems == 5

    env.step(actions[3])

    assert env.state.node_type == NodeType.OVERFLOW_DISCARD
    assert env.state.current_player == 0

    
    assert sum(env.state.players[0].gems.values()) == 11
    actions = env._legal_actions(env.state)

    num_discard_gems = 0

    for action in actions:
        assert action.action_type == ActionType.DISCARD_GEMS
        num_discard_gems += 1

    assert num_discard_gems == 4


    env.step(actions[0])

    assert env.state.node_type == NodeType.MAIN_DECISION
    assert env.state.current_player == 1

def test_discard_noble(env):
    env.reset()

    env.state.players[0].gems = {
        GemColor.WHITE: 3,
        GemColor.BLACK: 0,
        GemColor.GREEN: 3,
        GemColor.BLUE: 0,
        GemColor.RED: 0,
        GemColor.GOLD: 3
    }

    env.state.players[0].bonuses = {
        GemColor.WHITE: 4,
        GemColor.BLACK: 4,
        GemColor.GREEN: 4,
        GemColor.BLUE: 4,
        GemColor.RED: 4,
    }

    action = Action (

        action_type=ActionType.TAKE_GEMS,
        gem_colors=(GemColor.GREEN, GemColor.BLUE, GemColor.RED)
    )

    env.step(action)

    assert env.state.node_type == NodeType.OVERFLOW_DISCARD
    assert env.state.current_player == 0

    actions = env._legal_actions(env.state)

    env.step(actions[0])
    assert env.state.node_type == NodeType.OVERFLOW_DISCARD
    assert env.state.current_player == 0

    env.step(actions[0])

    assert env.state.node_type == NodeType.NOBLE_CLAIM
    assert env.state.current_player == 0

    actions = env._legal_actions(env.state)

    env.step(actions[0])

    assert env.state.node_type == NodeType.MAIN_DECISION
    assert env.state.current_player == 1


    actions = env._legal_actions(env.state)

    env.step(actions[0])
    assert env.state.node_type == NodeType.MAIN_DECISION
    assert env.state.current_player == 0

    #reserve action i think
    env.step(actions[0])


    assert env.state.node_type == NodeType.OVERFLOW_DISCARD
    assert env.state.current_player == 0

    actions = env._legal_actions(env.state)
    env.step(actions[0])

    assert env.state.node_type == NodeType.NOBLE_CLAIM
    assert env.state.current_player == 0

    actions = env._legal_actions(env.state)
    env.step(actions[0])

    
    assert env.state.node_type == NodeType.MAIN_DECISION
    assert env.state.current_player == 1

# TODO need a regular discard test case 
# TODO need a test case where you bought a card and activates 2-3 noble claim
# then you need take gems action which causes you to exceed the discard limit
# creates a scenario where you need to discard and also