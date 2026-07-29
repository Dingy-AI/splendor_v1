import pytest
from splendor_v1.env.core.constants import VICTORY_REQUIREMENT
from splendor_v1.env.env import SplendorEnv
from splendor_v1.env.core.enums import GemColor, NodeType, ActionType, GemColor
from splendor_v1.env.core.card import Card
from splendor_v1.env.core.actions import Action
from splendor_v1.env.core.action_constants import ACTION_SPACE_SIZE, DISCARD_START, NOBLE_START
from copy import deepcopy
@pytest.fixture
def env():
    return SplendorEnv()


def test_reset_noble(env):
    env.reset()

    assert len(env.state.nobles) == 3
    assert all(noble is not None for noble in env.state.nobles)


def test_noble_claim(env):
    env.reset()

    env.state.players[0].bonuses = {

        GemColor.WHITE: 4,
        GemColor.BLUE: 4,
        GemColor.GREEN: 4,
        GemColor.RED:4,
        GemColor.BLACK:4
    }   

    actions = env._legal_actions(env.state)

    num_noble_action = 0 
    for action in actions:
        if action.action_type == ActionType.TAKE_NOBLE:
            num_noble_action += 1
    assert num_noble_action == 0

    action = Action (

        action_type=ActionType.TAKE_GEMS,
        gem_colors=(GemColor.GREEN, GemColor.BLUE, GemColor.RED)
    )

    assert env.state.current_player == 0

    assert env.state.node_type == NodeType.MAIN_DECISION

    # 1st player does a take gem action
    # this triggers the noble claim due to 1st player qualifying for it
    env.step(action)

    assert env.state.node_type == NodeType.NOBLE_CLAIM

    actions = env._legal_actions(env.state)

    # there should be 3 noble actions here 
    assert len(actions) == 3
    for action in actions:
        assert action.action_type == ActionType.TAKE_NOBLE
        action_id = env.action_to_id(action)
        assert action_id >= NOBLE_START

    # current player should still be 1st player to do the noble action
    assert env.state.current_player == 0
    env.step(actions[0])

    # after taking noble action it should now be back to main decision
    # should be player 2's turn 
    assert env.state.node_type == NodeType.MAIN_DECISION
    assert env.state.current_player == 1

    assert env.state.players[0].nobles != None

    action = Action (

        action_type=ActionType.TAKE_GEMS,
        gem_colors=(GemColor.GREEN, GemColor.BLUE, GemColor.RED)
    )


    # after player 2 it should go back to first player 
    env.step(action)

    assert env.state.current_player == 0
    assert env.state.node_type == NodeType.MAIN_DECISION


    # should still be first player since he is in noble claim again
    env.step(action)

    assert env.state.current_player == 0
    assert env.state.node_type == NodeType.NOBLE_CLAIM

def test_noble_end_game_trigger(env):
    env.reset()


    env.state.players[0].bonuses = {

        GemColor.WHITE: 4,
        GemColor.BLUE: 4,
        GemColor.GREEN: 4,
        GemColor.RED:4,
        GemColor.BLACK:4
    }   

    env.state.players[0].points = VICTORY_REQUIREMENT-1

    action = Action (

        action_type=ActionType.TAKE_GEMS,
        gem_colors=(GemColor.GREEN, GemColor.BLUE, GemColor.RED)
    )

    env.step(action)

    actions = env._legal_actions(env.state)

    obs, reward, terminated, truncated, info = env.step(actions[0])

    assert terminated == False 
    assert env.state.end_triggered == True
    assert env.state.current_player == 1
    assert env.state.game_over == False 
    obs, reward, terminated, truncated, info = env.step(action)

    assert terminated == True
    assert env.state.game_over == True
    assert env.state.winners[0] == 0

#TODO need to do a discard/noble qualification at the same time