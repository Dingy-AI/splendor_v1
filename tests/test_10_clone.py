import pytest
from splendor_v1.env.env import SplendorEnv
from splendor_v1.env.core.enums import GemColor, NodeType, ActionType, GemColor
from splendor_v1.env.core.card import Card
from splendor_v1.env.core.actions import Action
from splendor_v1.env.core.action_constants import ACTION_SPACE_SIZE, DISCARD_START, NOBLE_START, DISCARD_COLORS
from copy import deepcopy


import random
@pytest.fixture
def env():
    return SplendorEnv()


def test_clone_independence():

    env = SplendorEnv()
    env.reset()

    clone = env.clone()

    # States should start equal
    assert env.state == clone.state

    # Check top-level objects are different
    assert env is not clone
    assert env.state is not clone.state

    # Players should not share references
    for original_player, cloned_player in zip(
        env.state.players,
        clone.state.players
    ):
        assert original_player is not cloned_player

        # Mutable dictionaries
        assert original_player.gems is not cloned_player.gems
        assert original_player.bonuses is not cloned_player.bonuses

        # Mutable lists
        assert (
            original_player.purchased_cards
            is not cloned_player.purchased_cards
        )

        assert (
            original_player.reserved_cards
            is not cloned_player.reserved_cards
        )

    # Bank should not share references
    assert env.state.bank is not clone.state.bank

    # Visible cards should not share list references
    for tier in env.state.visible_cards:

        assert (
            env.state.visible_cards[tier]
            is not clone.state.visible_cards[tier]
        )

def test_clone_step_does_not_modify_original():

    env = SplendorEnv()
    env.reset()

    clone = env.clone()


    # Take any legal action
    action = clone._legal_actions(env.state)[0]

    clone.step(action)


    # Clone should now be different
    assert env.state != clone.state

def test_clone_independence():
    env = SplendorEnv()
    env.reset()

    clone_game_state = env.state.clone()

    clone_game_state.game_over = True

    assert env.state != clone_game_state

