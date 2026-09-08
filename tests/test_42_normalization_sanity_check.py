import numpy as np
import pytest
from splendor_v1.env.env import SplendorEnv
from splendor_v1.env.core.constants import GemColor
from splendor_v1.env.core.card import Card
from splendor_v1.env.core.noble import Noble

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





@pytest.mark.parametrize(
    "card, expected",
    [
        (
            T1_SINGLE_COLOR,
            [
                # Costs
                0.0, 3 / 7, 0.0, 0.0, 0.0,

                # Bonus color: WHITE
                1.0, 0.0, 0.0, 0.0, 0.0,

                # Points
                0.0,
            ],
        ),
        (
            T1_TWO_COLOR,
            [
                0.0, 2 / 7, 0.0, 0.0, 2 / 7,
                1.0, 0.0, 0.0, 0.0, 0.0,
                0.0,
            ],
        ),
        (
            T2_THREE_COLOR,
            [
                0.0, 0.0, 3 / 7, 2 / 7, 2 / 7,
                1.0, 0.0, 0.0, 0.0, 0.0,
                1 / 5,
            ],
        ),
        (
            T2_HIGH_BLACK,
            [
                2 / 7, 0.0, 0.0, 1 / 7, 4 / 7,

                # Bonus color: BLUE
                0.0, 1.0, 0.0, 0.0, 0.0,

                2 / 5,
            ],
        ),
        (
            T3_TWO_COLOR,
            [
                3 / 7, 0.0, 0.0, 0.0, 1.0,
                1.0, 0.0, 0.0, 0.0, 0.0,
                1.0,
            ],
        ),
        (
            T3_THREE_COLOR,
            [
                3 / 7, 0.0, 0.0, 3 / 7, 6 / 7,
                1.0, 0.0, 0.0, 0.0, 0.0,
                4 / 5,
            ],
        ),
    ]
)
def test_card_encoding_is_normalized(
    card,
    expected,
):
    env = SplendorEnv()

    encoded = (
        env.observation_encoder
        .slow_encode_card(card)
    )

    assert len(encoded) == 11

    np.testing.assert_allclose(
        encoded,
        expected,
        rtol=1e-6,
        atol=1e-6,
    )


def test_single_player_encoding_is_normalized():

    env = SplendorEnv()
    env.reset(seed=0)

    encoder = env.observation_encoder
    player = env.state.players[0]

    # Player gems
    player.gems[GemColor.WHITE] = 13
    player.gems[GemColor.BLUE] = 6
    player.gems[GemColor.GREEN] = 0
    player.gems[GemColor.RED] = 3
    player.gems[GemColor.BLACK] = 1
    player.gems[GemColor.GOLD] = 2

    # Permanent bonuses
    player.bonuses[GemColor.WHITE] = 5
    player.bonuses[GemColor.BLUE] = 2
    player.bonuses[GemColor.GREEN] = 0
    player.bonuses[GemColor.RED] = 6
    player.bonuses[GemColor.BLACK] = 1

    # One reserved card
    player.reserved_cards = [
        T3_TWO_COLOR
    ]

    player.points = 15

    # _encode_single_player depends on this
    encoder.player_gem_norm = 13

    encoded = encoder._encode_single_player(
        player
    )

    expected = [

        # --------------------
        # Gems: /13
        # --------------------
        1.0,
        6 / 13,
        0.0,
        3 / 13,
        1 / 13,
        2 / 13,

        # --------------------
        # Bonuses: /5
        # --------------------
        1.0,
        2 / 5,
        0.0,
        6 / 5,
        1 / 5,

        # --------------------
        # Reserved card 1
        # T3_TWO_COLOR
        # --------------------
        3 / 7,
        0.0,
        0.0,
        0.0,
        1.0,

        1.0,
        0.0,
        0.0,
        0.0,
        0.0,

        1.0,

        # --------------------
        # Empty reserve 2
        # --------------------
        *([0.0] * 11),

        # Empty reserve 3
        *([0.0] * 11),

        # --------------------
        # Points: /20
        # --------------------
        15 / 20,
    ]

    assert len(encoded) == 45
    assert len(expected) == 45

    np.testing.assert_allclose(
        encoded,
        expected,
        rtol=1e-6,
        atol=1e-6,
    )

def test_bank_encoding_is_normalized():

    env = SplendorEnv()
    env.reset(seed=0)

    bank = {
        GemColor.WHITE: 4,
        GemColor.BLUE: 2,
        GemColor.GREEN: 1,
        GemColor.RED: 0,
        GemColor.BLACK: 3,
        GemColor.GOLD: 5,
    }

    encoded = (
        env.observation_encoder
        ._encode_bank(bank)
    )

    expected = [
        1.0,   # WHITE 4 / 4
        0.5,   # BLUE  2 / 4
        0.25,  # GREEN 1 / 4
        0.0,   # RED   0 / 4
        0.75,  # BLACK 3 / 4
        1.0,   # GOLD  5 / 5
    ]

    assert len(encoded) == 6

    np.testing.assert_allclose(
        encoded,
        expected,
        rtol=1e-6,
        atol=1e-6,
    )

def test_encode_decks_is_normalized():

    env = SplendorEnv()
    env.reset(seed=0)

    encoded = (
        env.observation_encoder
        ._encode_decks(env.state.decks)
    )

    expected = [
        1.0,  # 36 / 36
        1.0,  # 26 / 26
        1.0,  # 16 / 16
    ]

    assert len(encoded) == 3

    np.testing.assert_allclose(
        encoded,
        expected,
        rtol=1e-6,
        atol=1e-6,
    )

def test_encode_decks_partial_sizes():

    env = SplendorEnv()
    env.reset(seed=0)

    env.state.decks[1] = env.state.decks[1][:18]
    env.state.decks[2] = env.state.decks[2][:13]
    env.state.decks[3] = env.state.decks[3][:8]

    encoded = (
        env.observation_encoder
        ._encode_decks(env.state.decks)
    )

    expected = [
        0.5,  # 18 / 36
        0.5,  # 13 / 26
        0.5,  # 8 / 16
    ]

    np.testing.assert_allclose(
        encoded,
        expected,
        rtol=1e-6,
        atol=1e-6,
    )

def test_encode_nobles_is_normalized():

    env = SplendorEnv()
    env.reset(seed=0)

    noble_1 = Noble(
        id=0,
        Name="King Test 1",
        points=3,
        requirement={
            GemColor.WHITE: 4,
            GemColor.BLUE: 0,
            GemColor.GREEN: 4,
            GemColor.RED: 0,
            GemColor.BLACK: 4,
        },
    )

    noble_2 = Noble(
        id=1,
        Name='Queen Test 2',
        points=3,
        requirement={
            GemColor.WHITE: 3,
            GemColor.BLUE: 3,
            GemColor.GREEN: 0,
            GemColor.RED: 3,
            GemColor.BLACK: 0,
        },
    )

    nobles = [
        noble_1,
        noble_2,
        None,
    ]

    encoded = (
        env.observation_encoder
        ._encode_nobles(nobles)
    )

    expected = [
        # Noble 1
        1.0,   # WHITE 4 / 4
        0.0,   # BLUE
        1.0,   # GREEN 4 / 4
        0.0,   # RED
        1.0,   # BLACK 4 / 4
        1.0,   # points 3 / 3

        # Noble 2
        0.75,  # WHITE 3 / 4
        0.75,  # BLUE 3 / 4
        0.0,   # GREEN
        0.75,  # RED 3 / 4
        0.0,   # BLACK
        1.0,   # points 3 / 3

        # Empty noble slot
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ]

    assert len(encoded) == 18

    np.testing.assert_allclose(
        encoded,
        expected,
        rtol=1e-6,
        atol=1e-6,
    )