import pytest
from splendor_v1.env.env import SplendorEnv
from splendor_v1.env.core.enums import GemColor, NodeType, ActionType, GemColor
from splendor_v1.env.core.card import Card
from splendor_v1.env.core.actions import Action
from splendor_v1.env.core.action_constants import ACTION_SPACE_SIZE, DISCARD_START, NOBLE_START

@pytest.fixture
def env():
    return SplendorEnv()

def get_take_gem_actions(env):
    return [
        action
        for action in env._legal_take_gems(
            env.state
        )
        if action.action_type
        == ActionType.TAKE_GEMS
    ]


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

def test_take_gems_reward(env):
    env.reset()
    action = Action (

        action_type=ActionType.TAKE_GEMS,
        gem_colors=(GemColor.GREEN, GemColor.BLUE, GemColor.RED)
    )

    obs, reward, terminated, truncated, info = env.step(action)
    assert reward == 0


#TODO need to create a test case just for observation and really hammer that one out

def test_take_gems_with_three_or_more_colors_has_no_single_or_two_different(
    env,
):
    env.reset()

    env.state.bank = {
        GemColor.WHITE: 1,
        GemColor.BLUE: 1,
        GemColor.GREEN: 1,
        GemColor.RED: 0,
        GemColor.BLACK: 0,
        GemColor.GOLD: 5,
    }

    actions = get_take_gem_actions(env)

    gem_choices = [
        action.gem_colors
        for action in actions
    ]

    assert (
        GemColor.WHITE,
        GemColor.BLUE,
        GemColor.GREEN,
    ) in gem_choices

    assert not any(
        len(colors) == 1
        for colors in gem_choices
    )

    assert not any(
        (
            len(colors) == 2
            and colors[0] != colors[1]
        )
        for colors in gem_choices
    )


def test_take_gems_with_two_available_colors_allows_two_different(
    env,
):
    env.reset()

    env.state.bank = {
        GemColor.WHITE: 1,
        GemColor.BLUE: 1,
        GemColor.GREEN: 0,
        GemColor.RED: 0,
        GemColor.BLACK: 0,
        GemColor.GOLD: 5,
    }

    actions = get_take_gem_actions(env)

    gem_choices = [
        action.gem_colors
        for action in actions
    ]

    assert (
        GemColor.WHITE,
        GemColor.BLUE,
    ) in gem_choices

    assert not any(
        len(colors) == 1
        for colors in gem_choices
    )


def test_take_gems_with_one_available_color_allows_single(
    env,
):
    env.reset()

    env.state.bank = {
        GemColor.WHITE: 1,
        GemColor.BLUE: 0,
        GemColor.GREEN: 0,
        GemColor.RED: 0,
        GemColor.BLACK: 0,
        GemColor.GOLD: 5,
    }

    actions = get_take_gem_actions(env)

    gem_choices = [
        action.gem_colors
        for action in actions
    ]

    assert (
        GemColor.WHITE,
    ) in gem_choices

    assert len(gem_choices) == 1

def test_take_two_same_color_still_allowed_when_bank_has_four(
    env,
):
    env.reset()

    env.state.bank = {
        GemColor.WHITE: 4,
        GemColor.BLUE: 1,
        GemColor.GREEN: 1,
        GemColor.RED: 0,
        GemColor.BLACK: 0,
        GemColor.GOLD: 5,
    }

    actions = get_take_gem_actions(env)

    gem_choices = [
        action.gem_colors
        for action in actions
    ]

    assert (
        GemColor.WHITE,
        GemColor.WHITE,
    ) in gem_choices

def test_legal_take_gems_action_count_with_mixed_bank(
    env,
):
    env.reset()

    env.state.bank = {
        GemColor.WHITE: 4,
        GemColor.BLUE: 3,
        GemColor.GREEN: 1,
        GemColor.RED: 0,
        GemColor.BLACK: 0,
        GemColor.GOLD: 5,
    }

    actions = env._legal_take_gems(
        env.state
    )

    gem_choices = {
        action.gem_colors
        for action in actions
    }

    assert len(actions) == 2

    assert gem_choices == {
        (
            GemColor.WHITE,
            GemColor.BLUE,
            GemColor.GREEN,
        ),
        (
            GemColor.WHITE,
            GemColor.WHITE,
        ),
    }