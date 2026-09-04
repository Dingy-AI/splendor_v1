import pytest

from splendor_v1.env.env import SplendorEnv
from splendor_v1.env.core.card import Card
from splendor_v1.env.core.player import Player
from splendor_v1.env.core.enums import GemColor, ActionType, CardType
from splendor_v1.env.core.constants import COLOR_ORDER
from splendor_v1.env.core.actions import Action


from splendor_v1.env.core.cost_lookup_table_v3 import (
    T1_PAYMENT_LOOKUP,
    T2_PAYMENT_LOOKUP,
    T3_PAYMENT_LOOKUP,
)

# ================================================================
# Test cards
# ================================================================

T1_SINGLE_COLOR = Card(
    id=0,
    tier=1,
    points=0,
    bonus_color=GemColor.WHITE,
    cost={
        GemColor.WHITE: 0,
        GemColor.BLUE: 3,
        GemColor.GREEN: 0,
        GemColor.RED: 0,
        GemColor.BLACK: 0,
    },
)


T1_TWO_COLOR = Card(
    id=15,
    tier=1,
    points=0,
    bonus_color=GemColor.WHITE,
    cost={
        GemColor.WHITE: 0,
        GemColor.BLUE: 2,
        GemColor.GREEN: 0,
        GemColor.RED: 0,
        GemColor.BLACK: 2,
    },
)


T2_THREE_COLOR = Card(
    id=50,
    tier=2,
    points=1,
    bonus_color=GemColor.WHITE,
    cost={
        GemColor.WHITE: 0,
        GemColor.BLUE: 0,
        GemColor.GREEN: 3,
        GemColor.RED: 2,
        GemColor.BLACK: 2,
    },
)


T2_HIGH_BLACK = Card(
    id=56,
    tier=2,
    points=2,
    bonus_color=GemColor.BLUE,
    cost={
        GemColor.WHITE: 2,
        GemColor.BLUE: 0,
        GemColor.GREEN: 0,
        GemColor.RED: 1,
        GemColor.BLACK: 4,
    },
)


T3_TWO_COLOR = Card(
    id=75,
    tier=3,
    points=5,
    bonus_color=GemColor.WHITE,
    cost={
        GemColor.WHITE: 3,
        GemColor.BLUE: 0,
        GemColor.GREEN: 0,
        GemColor.RED: 0,
        GemColor.BLACK: 7,
    },
)


T3_THREE_COLOR = Card(
    id=80,
    tier=3,
    points=4,
    bonus_color=GemColor.WHITE,
    cost={
        GemColor.WHITE: 3,
        GemColor.BLUE: 0,
        GemColor.GREEN: 0,
        GemColor.RED: 3,
        GemColor.BLACK: 6,
    },
)


ALL_TEST_CARDS = [
    T1_SINGLE_COLOR,
    T1_TWO_COLOR,
    T2_THREE_COLOR,
    T2_HIGH_BLACK,
    T3_TWO_COLOR,
    T3_THREE_COLOR,
]


# ================================================================
# Helpers
# ================================================================

def make_player(
    gems=None,
    bonuses=None,
):
    player = Player(
        id=0,
        gems={
            color: 0
            for color in GemColor
        },
        bonuses={
            color: 0
            for color in COLOR_ORDER
        },
        reserved_cards=[],
        purchased_cards=[],
        nobles=[],
        points=0,
    )

    if gems:
        player.gems.update(gems)

    if bonuses:
        player.bonuses.update(bonuses)

    return player


def get_payment_lookup(card):
    if card.tier == 1:
        return T1_PAYMENT_LOOKUP

    if card.tier == 2:
        return T2_PAYMENT_LOOKUP

    if card.tier == 3:
        return T3_PAYMENT_LOOKUP

    raise ValueError(
        f"Unknown card tier: {card.tier}"
    )


def old_valid_payments(
    env,
    player,
    card,
):
    """
    Reference implementation.

    This uses the current exhaustive payment-table system.

    The returned tuple represents how much GOLD substitutes
    for each actual color in COLOR_ORDER.
    """

    payment_lookup = get_payment_lookup(card)

    color_mapping = env.get_color_mapping(card)

    valid_payments = []

    for canonical_payment in payment_lookup:

        actual_gold_payment = env.map_payment_to_card(
            canonical_payment,
            color_mapping,
        )

        if env._can_pay(
            player,
            card,
            actual_gold_payment,
        ):
            valid_payments.append(
                actual_gold_payment
            )

    return valid_payments


# ================================================================
# Tier 1
# ================================================================

def test_t1_single_color_no_gold_needed():

    env = SplendorEnv()

    player = make_player(
        gems={
            GemColor.BLUE: 3,
        }
    )

    payments = set(
        env._generate_valid_payments(
            player,
            T1_SINGLE_COLOR,
        )
    )

    assert payments == {
        (0, 0, 0, 0, 0),
    }


def test_t1_single_color_one_gold_available():

    env = SplendorEnv()

    player = make_player(
        gems={
            GemColor.BLUE: 3,
            GemColor.GOLD: 1,
        }
    )

    payments = set(
        env._generate_valid_payments(
            player,
            T1_SINGLE_COLOR,
        )
    )

    assert payments == {
        # Pay all 3 blue normally
        (0, 0, 0, 0, 0),

        # Replace 1 blue with gold
        (0, 1, 0, 0, 0),
    }


def test_t1_single_color_two_gold_available():

    env = SplendorEnv()

    player = make_player(
        gems={
            GemColor.BLUE: 3,
            GemColor.GOLD: 2,
        }
    )

    payments = set(
        env._generate_valid_payments(
            player,
            T1_SINGLE_COLOR,
        )
    )

    assert payments == {
        (0, 0, 0, 0, 0),
        (0, 1, 0, 0, 0),
        (0, 2, 0, 0, 0),
    }


def test_t1_single_color_gold_is_required():

    env = SplendorEnv()

    player = make_player(
        gems={
            GemColor.BLUE: 2,
            GemColor.GOLD: 1,
        }
    )

    payments = set(
        env._generate_valid_payments(
            player,
            T1_SINGLE_COLOR,
        )
    )

    assert payments == {
        (0, 1, 0, 0, 0),
    }


def test_t1_two_color_gold_can_replace_either_color():

    env = SplendorEnv()

    player = make_player(
        gems={
            GemColor.BLUE: 2,
            GemColor.BLACK: 2,
            GemColor.GOLD: 1,
        }
    )

    payments = set(
        env._generate_valid_payments(
            player,
            T1_TWO_COLOR,
        )
    )

    assert payments == {
        # No gold
        (0, 0, 0, 0, 0),

        # Gold replaces one blue
        (0, 1, 0, 0, 0),

        # Gold replaces one black
        (0, 0, 0, 0, 1),
    }


def test_t1_bonus_reduces_required_cost():

    env = SplendorEnv()

    player = make_player(
        gems={
            GemColor.BLUE: 2,
        },
        bonuses={
            GemColor.BLUE: 1,
        },
    )

    payments = set(
        env._generate_valid_payments(
            player,
            T1_SINGLE_COLOR,
        )
    )

    # Card costs 3 blue.
    # Bonus removes 1.
    # Player pays remaining 2 with blue gems.
    assert payments == {
        (0, 0, 0, 0, 0),
    }


# ================================================================
# Tier 2
# ================================================================

def test_t2_three_color_no_gold_needed():

    env = SplendorEnv()

    player = make_player(
        gems={
            GemColor.GREEN: 3,
            GemColor.RED: 2,
            GemColor.BLACK: 2,
        }
    )

    payments = set(
        env._generate_valid_payments(
            player,
            T2_THREE_COLOR,
        )
    )

    assert payments == {
        (0, 0, 0, 0, 0),
    }


def test_t2_three_color_one_gold_available():

    env = SplendorEnv()

    player = make_player(
        gems={
            GemColor.GREEN: 3,
            GemColor.RED: 2,
            GemColor.BLACK: 2,
            GemColor.GOLD: 1,
        }
    )

    payments = set(
        env._generate_valid_payments(
            player,
            T2_THREE_COLOR,
        )
    )

    assert payments == {
        # No gold
        (0, 0, 0, 0, 0),

        # Replace 1 green
        (0, 0, 1, 0, 0),

        # Replace 1 red
        (0, 0, 0, 1, 0),

        # Replace 1 black
        (0, 0, 0, 0, 1),
    }


def test_t2_three_color_gold_required_for_green():

    env = SplendorEnv()

    player = make_player(
        gems={
            GemColor.GREEN: 2,
            GemColor.RED: 2,
            GemColor.BLACK: 2,
            GemColor.GOLD: 1,
        }
    )

    payments = set(
        env._generate_valid_payments(
            player,
            T2_THREE_COLOR,
        )
    )

    assert payments == {
        (0, 0, 1, 0, 0),
    }


def test_t2_high_black_gold_required():

    env = SplendorEnv()

    player = make_player(
        gems={
            GemColor.WHITE: 2,
            GemColor.RED: 1,
            GemColor.BLACK: 3,
            GemColor.GOLD: 1,
        }
    )

    payments = set(
        env._generate_valid_payments(
            player,
            T2_HIGH_BLACK,
        )
    )

    # Black cost is 4, but player only has 3 black.
    assert payments == {
        (0, 0, 0, 0, 1),
    }


def test_t2_bonuses_and_gold():

    env = SplendorEnv()

    player = make_player(
        gems={
            GemColor.GREEN: 1,
            GemColor.RED: 1,
            GemColor.BLACK: 1,
            GemColor.GOLD: 1,
        },
        bonuses={
            GemColor.GREEN: 1,
            GemColor.RED: 1,
            GemColor.BLACK: 1,
        },
    )

    # Original cost:
    # GREEN 3, RED 2, BLACK 2
    #
    # After bonuses:
    # GREEN 2, RED 1, BLACK 1
    #
    # Player is short exactly one GREEN.

    payments = set(
        env._generate_valid_payments(
            player,
            T2_THREE_COLOR,
        )
    )

    assert payments == {
        (0, 0, 1, 0, 0),
    }


# ================================================================
# Tier 3
# ================================================================

def test_t3_two_color_no_gold_needed():

    env = SplendorEnv()

    player = make_player(
        gems={
            GemColor.WHITE: 3,
            GemColor.BLACK: 7,
        }
    )

    payments = set(
        env._generate_valid_payments(
            player,
            T3_TWO_COLOR,
        )
    )

    assert payments == {
        (0, 0, 0, 0, 0),
    }


def test_t3_two_color_two_gold_available():

    env = SplendorEnv()

    player = make_player(
        gems={
            GemColor.WHITE: 3,
            GemColor.BLACK: 7,
            GemColor.GOLD: 2,
        }
    )

    payments = set(
        env._generate_valid_payments(
            player,
            T3_TWO_COLOR,
        )
    )

    assert payments == {
        # 0 gold
        (0, 0, 0, 0, 0),

        # 1 gold
        (1, 0, 0, 0, 0),
        (0, 0, 0, 0, 1),

        # 2 gold
        (2, 0, 0, 0, 0),
        (1, 0, 0, 0, 1),
        (0, 0, 0, 0, 2),
    }


def test_t3_two_color_gold_is_required():

    env = SplendorEnv()

    player = make_player(
        gems={
            GemColor.WHITE: 3,
            GemColor.BLACK: 5,
            GemColor.GOLD: 2,
        }
    )

    payments = set(
        env._generate_valid_payments(
            player,
            T3_TWO_COLOR,
        )
    )

    # Need 7 black.
    # Player only owns 5.
    # Both gold must substitute for black.

    assert payments == {
        (0, 0, 0, 0, 2),
    }


def test_t3_three_color_with_bonuses_and_gold():

    env = SplendorEnv()

    player = make_player(
        gems={
            GemColor.WHITE: 2,
            GemColor.RED: 2,
            GemColor.BLACK: 4,
            GemColor.GOLD: 1,
        },
        bonuses={
            GemColor.WHITE: 1,
            GemColor.RED: 1,
            GemColor.BLACK: 1,
        },
    )

    # Card:
    # WHITE 3
    # RED   3
    # BLACK 6
    #
    # After bonuses:
    # WHITE 2
    # RED   2
    # BLACK 5
    #
    # Player has:
    # WHITE 2
    # RED   2
    # BLACK 4
    #
    # Therefore exactly one gold must substitute for BLACK.

    payments = set(
        env._generate_valid_payments(
            player,
            T3_THREE_COLOR,
        )
    )

    assert payments == {
        (0, 0, 0, 0, 1),
    }


def test_t3_three_color_multiple_gold_distributions():

    env = SplendorEnv()

    player = make_player(
        gems={
            GemColor.WHITE: 3,
            GemColor.RED: 3,
            GemColor.BLACK: 6,
            GemColor.GOLD: 2,
        }
    )

    payments = set(
        env._generate_valid_payments(
            player,
            T3_THREE_COLOR,
        )
    )

    assert payments == {
        # No gold
        (0, 0, 0, 0, 0),

        # One gold
        (1, 0, 0, 0, 0),
        (0, 0, 0, 1, 0),
        (0, 0, 0, 0, 1),

        # Two gold on one color
        (2, 0, 0, 0, 0),
        (0, 0, 0, 2, 0),
        (0, 0, 0, 0, 2),

        # One gold on two colors
        (1, 0, 0, 1, 0),
        (1, 0, 0, 0, 1),
        (0, 0, 0, 1, 1),
    }


# ================================================================
# Unaffordable cards
# ================================================================

@pytest.mark.parametrize(
    "card",
    ALL_TEST_CARDS,
)
def test_no_gold_and_not_enough_colored_gems_returns_no_payments(
    card,
):

    env = SplendorEnv()

    player = make_player()

    payments = env._generate_valid_payments(
        player,
        card,
    )

    assert payments == []


@pytest.mark.parametrize(
    "card",
    ALL_TEST_CARDS,
)
def test_not_enough_gold_returns_no_payments(
    card,
):

    env = SplendorEnv()

    player = make_player(
        gems={
            GemColor.GOLD: 1,
        }
    )

    payments = env._generate_valid_payments(
        player,
        card,
    )

    assert payments == []


# ================================================================
# Critical regression tests
# ================================================================

@pytest.mark.parametrize(
    "card",
    ALL_TEST_CARDS,
)
def test_generated_payments_match_old_system_without_gold(
    card,
):

    env = SplendorEnv()

    player = make_player(
        gems={
            GemColor.WHITE: 7,
            GemColor.BLUE: 7,
            GemColor.GREEN: 7,
            GemColor.RED: 7,
            GemColor.BLACK: 7,
            GemColor.GOLD: 0,
        }
    )

    old_payments = set(
        old_valid_payments(
            env,
            player,
            card,
        )
    )

    new_payments = set(
        env._generate_valid_payments(
            player,
            card,
        )
    )

    assert new_payments == old_payments


@pytest.mark.parametrize(
    "card",
    ALL_TEST_CARDS,
)
def test_generated_payments_match_old_system_with_gold(
    card,
):

    env = SplendorEnv()

    player = make_player(
        gems={
            GemColor.WHITE: 4,
            GemColor.BLUE: 4,
            GemColor.GREEN: 4,
            GemColor.RED: 4,
            GemColor.BLACK: 4,
            GemColor.GOLD: 2,
        }
    )

    old_payments = set(
        old_valid_payments(
            env,
            player,
            card,
        )
    )

    new_payments = set(
        env._generate_valid_payments(
            player,
            card,
        )
    )

    assert new_payments == old_payments


@pytest.mark.parametrize(
    "card",
    ALL_TEST_CARDS,
)
def test_generated_payments_match_old_system_with_bonuses(
    card,
):

    env = SplendorEnv()

    player = make_player(
        gems={
            GemColor.WHITE: 2,
            GemColor.BLUE: 2,
            GemColor.GREEN: 2,
            GemColor.RED: 2,
            GemColor.BLACK: 2,
            GemColor.GOLD: 2,
        },
        bonuses={
            GemColor.WHITE: 1,
            GemColor.BLUE: 1,
            GemColor.GREEN: 1,
            GemColor.RED: 1,
            GemColor.BLACK: 1,
        },
    )

    old_payments = set(
        old_valid_payments(
            env,
            player,
            card,
        )
    )

    new_payments = set(
        env._generate_valid_payments(
            player,
            card,
        )
    )

    assert new_payments == old_payments


@pytest.mark.parametrize(
    "card",
    ALL_TEST_CARDS,
)
def test_generated_payments_match_old_system_when_gold_is_required(
    card,
):

    env = SplendorEnv()

    player = make_player(
        gems={
            GemColor.WHITE: 1,
            GemColor.BLUE: 1,
            GemColor.GREEN: 1,
            GemColor.RED: 1,
            GemColor.BLACK: 1,
            GemColor.GOLD: 5,
        }
    )

    old_payments = set(
        old_valid_payments(
            env,
            player,
            card,
        )
    )

    new_payments = set(
        env._generate_valid_payments(
            player,
            card,
        )
    )

    assert new_payments == old_payments


def set_player_resources(
    player,
    gems=None,
    bonuses=None,
    white=0,
    blue=0,
    green=0,
    red=0,
    black=0,
    gold=0,
    white_bonus=0,
    blue_bonus=0,
    green_bonus=0,
    red_bonus=0,
    black_bonus=0,
):

    player.gems = {
        GemColor.WHITE: white,
        GemColor.BLUE: blue,
        GemColor.GREEN: green,
        GemColor.RED: red,
        GemColor.BLACK: black,
        GemColor.GOLD: gold,
    }

    player.bonuses = {
        GemColor.WHITE: white_bonus,
        GemColor.BLUE: blue_bonus,
        GemColor.GREEN: green_bonus,
        GemColor.RED: red_bonus,
        GemColor.BLACK: black_bonus,
    }

    if gems:
        player.gems.update(gems)

    if bonuses:
        player.bonuses.update(bonuses)


@pytest.mark.parametrize(
    "card",
    ALL_TEST_CARDS,
)
def test_valid_payments_no_resources(card):

    env = SplendorEnv()
    env.reset()

    player = env.state.players[0]

    set_player_resources(player)

    slow_result = env.slow_generate_valid_payments(
        player,
        card,
    )

    fast_result = env._generate_valid_payments(
        player,
        card,
    )

    assert fast_result == slow_result


@pytest.mark.parametrize(
    "card",
    ALL_TEST_CARDS,
)
def test_valid_payments_full_colored_gems(card):

    env = SplendorEnv()
    env.reset()

    player = env.state.players[0]

    set_player_resources(
        player,
        gems={
            GemColor.WHITE: 7,
            GemColor.BLUE: 7,
            GemColor.GREEN: 7,
            GemColor.RED: 7,
            GemColor.BLACK: 7,
            GemColor.GOLD: 0,
        },
    )

    slow_result = env.slow_generate_valid_payments(
        player,
        card,
    )

    fast_result = env._generate_valid_payments(
        player,
        card,
    )

    assert fast_result == slow_result


@pytest.mark.parametrize(
    "card",
    ALL_TEST_CARDS,
)
def test_valid_payments_with_gold(card):

    env = SplendorEnv()
    env.reset()

    player = env.state.players[0]

    set_player_resources(
        player,
        gems={
            GemColor.WHITE: 1,
            GemColor.BLUE: 1,
            GemColor.GREEN: 1,
            GemColor.RED: 1,
            GemColor.BLACK: 1,
            GemColor.GOLD: 3,
        },
    )

    slow_result = env.slow_generate_valid_payments(
        player,
        card,
    )

    fast_result = env._generate_valid_payments(
        player,
        card,
    )

    assert fast_result == slow_result


@pytest.mark.parametrize(
    "card",
    ALL_TEST_CARDS,
)
def test_valid_payments_with_bonuses(card):

    env = SplendorEnv()
    env.reset()

    player = env.state.players[0]

    set_player_resources(
        player,
        gems={
            GemColor.WHITE: 1,
            GemColor.BLUE: 1,
            GemColor.GREEN: 1,
            GemColor.RED: 1,
            GemColor.BLACK: 1,
            GemColor.GOLD: 2,
        },
        bonuses={
            GemColor.WHITE: 1,
            GemColor.BLUE: 1,
            GemColor.GREEN: 1,
            GemColor.RED: 1,
            GemColor.BLACK: 1,
        },
    )

    slow_result = env.slow_generate_valid_payments(
        player,
        card,
    )

    fast_result = env._generate_valid_payments(
        player,
        card,
    )

    assert fast_result == slow_result


@pytest.mark.parametrize(
    "card",
    ALL_TEST_CARDS,
)
def test_valid_payments_large_gold_supply(card):

    env = SplendorEnv()
    env.reset()

    player = env.state.players[0]

    set_player_resources(
        player,
        gems={
            GemColor.WHITE: 0,
            GemColor.BLUE: 0,
            GemColor.GREEN: 0,
            GemColor.RED: 0,
            GemColor.BLACK: 0,
            GemColor.GOLD: 10,
        },
    )

    slow_result = env.slow_generate_valid_payments(
        player,
        card,
    )

    fast_result = env._generate_valid_payments(
        player,
        card,
    )

    assert fast_result == slow_result





@pytest.mark.parametrize(
    "card",
    ALL_TEST_CARDS,
)
def test_valid_payments_mixed_midgame_state(card):

    env = SplendorEnv()
    env.reset()

    player = env.state.players[0]

    set_player_resources(
        player,
        gems={
            GemColor.WHITE: 2,
            GemColor.BLUE: 0,
            GemColor.GREEN: 3,
            GemColor.RED: 1,
            GemColor.BLACK: 2,
            GemColor.GOLD: 2,
        },
        bonuses={
            GemColor.WHITE: 1,
            GemColor.BLUE: 2,
            GemColor.GREEN: 0,
            GemColor.RED: 1,
            GemColor.BLACK: 0,
        },
    )

    slow_result = env.slow_generate_valid_payments(
        player,
        card,
    )

    fast_result = env._generate_valid_payments(
        player,
        card,
    )

    assert fast_result == slow_result

# ================================================================
# _generate_valid_payments optimization tests
# ================================================================

def assert_payment_generators_match(
    env,
    player,
    card,
):

    slow_result = (
        env.slow_2_generate_valid_payments(
            player,
            card,
        )
    )

    fast_result = (
        env._generate_valid_payments(
            player,
            card,
        )
    )

    assert fast_result == slow_result


# ================================================================
# Cannot afford
# ================================================================

def test_generate_valid_payments_cannot_afford():

    env = SplendorEnv()
    env.reset()

    player = env.state.players[0]

    # T1_SINGLE_COLOR costs:
    #
    # BLUE = 3
    #
    # Player only has 1 blue and 1 gold.
    # Therefore:
    #
    # minimum gold required = 2
    # gold available = 1

    set_player_resources(
        player,
        blue=1,
        gold=1,
    )

    assert_payment_generators_match(
        env,
        player,
        T1_SINGLE_COLOR,
    )

    result = env._generate_valid_payments(
        player,
        T1_SINGLE_COLOR,
    )

    assert result == []


# ================================================================
# Zero gold
# ================================================================

def test_generate_valid_payments_zero_gold():

    env = SplendorEnv()
    env.reset()

    player = env.state.players[0]

    # T1_TWO_COLOR costs:
    #
    # BLUE  = 2
    # BLACK = 2
    #
    # Player can pay everything using colored gems.
    # No gold exists, so there can only be one gold allocation.

    set_player_resources(
        player,
        blue=2,
        black=2,
        gold=0,
    )

    assert_payment_generators_match(
        env,
        player,
        T1_TWO_COLOR,
    )

    result = env._generate_valid_payments(
        player,
        T1_TWO_COLOR,
    )

    assert result == [
        (0, 0, 0, 0, 0)
    ]


# ================================================================
# All gold is mandatory
# extra_gold == 0
# ================================================================

def test_generate_valid_payments_all_gold_is_mandatory():

    env = SplendorEnv()
    env.reset()

    player = env.state.players[0]

    # T1_SINGLE_COLOR:
    #
    # BLUE cost = 3
    #
    # Player:
    #   BLUE = 2
    #   GOLD = 1
    #
    # Therefore:
    #
    # minimum_gold = (0, 1, 0, 0, 0)
    # minimum_total = 1
    # gold_available = 1
    # extra_gold = 0

    set_player_resources(
        player,
        blue=2,
        gold=1,
    )

    assert_payment_generators_match(
        env,
        player,
        T1_SINGLE_COLOR,
    )

    result = env._generate_valid_payments(
        player,
        T1_SINGLE_COLOR,
    )

    assert result == [
        (0, 1, 0, 0, 0)
    ]


# ================================================================
# One spare gold
# extra_gold == 1
# ================================================================

def test_generate_valid_payments_one_spare_gold():

    env = SplendorEnv()
    env.reset()

    player = env.state.players[0]

    # T1_TWO_COLOR:
    #
    # BLUE  = 2
    # BLACK = 2
    #
    # Player:
    #   BLUE  = 1
    #   BLACK = 2
    #   GOLD  = 2
    #
    # Mandatory:
    #   1 gold must be spent as BLUE.
    #
    # Therefore:
    #
    # minimum_total = 1
    # gold_available = 2
    # extra_gold = 1

    set_player_resources(
        player,
        blue=1,
        black=2,
        gold=2,
    )

    assert_payment_generators_match(
        env,
        player,
        T1_TWO_COLOR,
    )

    result = env._generate_valid_payments(
        player,
        T1_TWO_COLOR,
    )

    assert result == [
        # Mandatory gold only.
        (0, 1, 0, 0, 0),

        # Spend optional gold as BLACK.
        (0, 1, 0, 0, 1),

        # Spend optional gold as BLUE.
        (0, 2, 0, 0, 0),
    ]


# ================================================================
# Two spare gold
# extra_gold == 2
# ================================================================

def test_generate_valid_payments_two_spare_gold():

    env = SplendorEnv()
    env.reset()

    player = env.state.players[0]

    # T2_THREE_COLOR:
    #
    # GREEN = 3
    # RED   = 2
    # BLACK = 2
    #
    # Player:
    #   GREEN = 2
    #   RED   = 2
    #   BLACK = 2
    #   GOLD  = 3
    #
    # Mandatory:
    #   GREEN needs 1 gold.
    #
    # Therefore:
    #
    # minimum_total = 1
    # gold_available = 3
    # extra_gold = 2

    set_player_resources(
        player,
        green=2,
        red=2,
        black=2,
        gold=3,
    )

    assert_payment_generators_match(
        env,
        player,
        T2_THREE_COLOR,
    )


# ================================================================
# Three spare gold
# product() fallback
# ================================================================

def test_generate_valid_payments_three_spare_gold():

    env = SplendorEnv()
    env.reset()

    player = env.state.players[0]

    # T2_THREE_COLOR:
    #
    # Player already owns enough colored gems
    # to pay the entire card.
    #
    # minimum_total = 0
    # gold_available = 3
    # extra_gold = 3
    #
    # This should hit the product() fallback.

    set_player_resources(
        player,
        green=3,
        red=2,
        black=2,
        gold=3,
    )

    assert_payment_generators_match(
        env,
        player,
        T2_THREE_COLOR,
    )


# ================================================================
# High-cost Tier 3 card
# ================================================================

def test_generate_valid_payments_t3_card():

    env = SplendorEnv()
    env.reset()

    player = env.state.players[0]

    # T3_TWO_COLOR:
    #
    # WHITE = 3
    # BLACK = 7
    #
    # Use a state where mandatory and optional
    # gold substitution are both possible.

    set_player_resources(
        player,
        white=2,
        black=5,
        gold=4,
    )

    assert_payment_generators_match(
        env,
        player,
        T3_TWO_COLOR,
    )


# ================================================================
# Bonuses
# ================================================================

def test_generate_valid_payments_with_bonuses():

    env = SplendorEnv()
    env.reset()

    player = env.state.players[0]

    # T2_HIGH_BLACK costs:
    #
    # WHITE = 2
    # RED   = 1
    # BLACK = 4
    #
    # Bonuses change the effective requirements.

    set_player_resources(
        player,
        white=1,
        red=1,
        black=1,
        gold=3,
        white_bonus=1,
        black_bonus=2,
    )

    assert_payment_generators_match(
        env,
        player,
        T2_HIGH_BLACK,
    )


# ================================================================
# Compare all supplied cards
# ================================================================

@pytest.mark.parametrize(
    "card",
    ALL_TEST_CARDS,
)
def test_generate_valid_payments_matches_slow_2_for_all_cards(
    card,
):

    env = SplendorEnv()
    env.reset()

    player = env.state.players[0]

    # Representative resource-heavy midgame state.

    set_player_resources(
        player,
        white=2,
        blue=3,
        green=2,
        red=1,
        black=3,
        gold=3,
        white_bonus=1,
        blue_bonus=1,
        green_bonus=2,
        red_bonus=1,
        black_bonus=0,
    )

    assert_payment_generators_match(
        env,
        player,
        card,
    )