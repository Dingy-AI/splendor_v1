import pytest
from splendor_v1.env.env import SplendorEnv
from splendor_v1.env.core.enums import GemColor, NodeType, ActionType, GemColor
from splendor_v1.env.core.card import Card
from splendor_v1.env.core.actions import Action
from splendor_v1.env.core.action_constants import ACTION_SPACE_SIZE, DISCARD_START, NOBLE_START

@pytest.fixture
def env():
    return SplendorEnv()

def test_take_one_gem(env):
    
    env.reset()

    action = Action(

        action_type=ActionType.TAKE_GEMS,
        gem_colors=(GemColor.WHITE,)
    )

    obs, reward, terminated, truncated, info= env.step(action)


    assert env.state.bank[GemColor.WHITE] == 3
    assert env.state.players[0].gems[GemColor.WHITE] == 1

def test_take_two_different_color(env):
    env.reset()

    action = Action(

        action_type=ActionType.TAKE_GEMS,
        gem_colors=(GemColor.WHITE, GemColor.GREEN)
    )


    obs, reward, terminated, truncated, info= env.step(action)


    assert env.state.bank[GemColor.WHITE] == 3
    assert env.state.players[0].gems[GemColor.WHITE] == 1
    assert env.state.bank[GemColor.GREEN] == 3
    assert env.state.players[0].gems[GemColor.GREEN] == 1


def test_take_two_same_color(env):
    env.reset()

    action = Action(

        action_type=ActionType.TAKE_GEMS,
        gem_colors=(GemColor.WHITE, GemColor.WHITE)
    )


    obs, reward, terminated, truncated, info= env.step(action)


    assert env.state.bank[GemColor.WHITE] == 2
    assert env.state.players[0].gems[GemColor.WHITE] == 2

def test_take_three_different_color(env):
    env.reset()

    action = Action (

        action_type=ActionType.TAKE_GEMS,
        gem_colors=(GemColor.GREEN, GemColor.BLUE, GemColor.RED)
    )

    
    obs, reward, terminated, truncated, info= env.step(action)


    assert env.state.bank[GemColor.WHITE] == 4
    assert env.state.players[0].gems[GemColor.WHITE] == 0
    assert env.state.bank[GemColor.BLACK] == 4
    assert env.state.players[0].gems[GemColor.BLACK] == 0

    assert env.state.bank[GemColor.BLUE] == 3
    assert env.state.players[0].gems[GemColor.BLUE] == 1
    assert env.state.bank[GemColor.GREEN] == 3
    assert env.state.players[0].gems[GemColor.GREEN] == 1
    assert env.state.bank[GemColor.RED] == 3
    assert env.state.players[0].gems[GemColor.RED] == 1

def test_turn_player_transition(env):
    env.reset()

    action = Action (

        action_type=ActionType.TAKE_GEMS,
        gem_colors=(GemColor.GREEN, GemColor.BLUE, GemColor.RED)
    )

    
    obs, reward, terminated, truncated, info= env.step(action)

    assert env.state.node_type == NodeType.MAIN_DECISION
    assert env.state.current_player == 1

def test_check_overflow(env):

    env.reset()


    env.state.players[0].gems = {
        GemColor.WHITE: 2,
        GemColor.BLUE: 2,
        GemColor.GREEN: 2,
        GemColor.RED: 2,
        GemColor.BLACK: 2,
        GemColor.GOLD: 0
    }
    
    action = Action (

        action_type=ActionType.TAKE_GEMS,
        gem_colors=(GemColor.GREEN, GemColor.BLUE, GemColor.RED)
    )

    obs, reward, terminated, truncated, info= env.step(action)

    assert env.state.node_type == NodeType.OVERFLOW_DISCARD
    assert env.state.current_player == 0
    assert sum(env.state.players[0].gems.values()) > 10