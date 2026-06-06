from core.card import Card
from core.enums import GemColor


BASE_TIER_1 = [
    Card(
        id=0,
        tier=1,
        points=0,
        bonus_color=GemColor.BLACK,
        cost={
            GemColor.WHITE: 1,
            GemColor.BLUE: 1,
            GemColor.GREEN: 1,
            GemColor.RED: 1,
            GemColor.BLACK: 0,
        }
    ),
    Card(
        id=1,
        tier=1,
        points=0,
        bonus_color=GemColor.BLACK,
        cost={
            GemColor.WHITE: 0,
            GemColor.BLUE: 0,
            GemColor.GREEN: 2,
            GemColor.RED: 1,
            GemColor.BLACK: 0,
        }
    ),
    Card(
        id=2,
        tier=1,
        points=0,
        bonus_color=GemColor.BLACK,
        cost={
            GemColor.WHITE: 2,
            GemColor.BLUE: 0,
            GemColor.GREEN: 2,
            GemColor.RED: 0,
            GemColor.BLACK: 0,
        }
    ),
    Card(
        id=3,
        tier=1,
        points=0,
        bonus_color=GemColor.BLACK,
        cost={
            GemColor.WHITE: 0,
            GemColor.BLUE: 0,
            GemColor.GREEN: 1,
            GemColor.RED: 3,
            GemColor.BLACK: 1,
        }
    ),
    Card(
        id=4,
        tier=1,
        points=0,
        bonus_color=GemColor.BLACK,
        cost={
            GemColor.WHITE: 0,
            GemColor.BLUE: 0,
            GemColor.GREEN: 3,
            GemColor.RED: 0,
            GemColor.BLACK: 0,
        }
    ),
    Card(
        id=5,
        tier=1,
        points=0,
        bonus_color=GemColor.BLACK,
        cost={
            GemColor.WHITE: 1,
            GemColor.BLUE: 2,
            GemColor.GREEN: 1,
            GemColor.RED: 1,
            GemColor.BLACK: 0,
        }
    ),
    Card(
        id=6,
        tier=1,
        points=0,
        bonus_color=GemColor.BLACK,
        cost={
            GemColor.WHITE: 2,
            GemColor.BLUE: 2,
            GemColor.GREEN: 0,
            GemColor.RED: 1,
            GemColor.BLACK: 0,
        }
    ),
    Card(
        id=7,
        tier=1,
        points=1,
        bonus_color=GemColor.BLACK,
        cost={
            GemColor.WHITE: 0,
            GemColor.BLUE: 4,
            GemColor.GREEN: 0,
            GemColor.RED: 0,
            GemColor.BLACK: 0,
        }
    ),
    Card(
        id=8,
        tier=1,
        points=0,
        bonus_color=GemColor.BLUE,
        cost={
            GemColor.WHITE: 1,
            GemColor.BLUE: 0,
            GemColor.GREEN: 0,
            GemColor.RED: 0,
            GemColor.BLACK: 2,
        }
    ),
    Card(
        id=9,
        tier=1,
        points=0,
        bonus_color=GemColor.BLUE,
        cost={
            GemColor.WHITE: 1,
            GemColor.BLUE: 0,
            GemColor.GREEN: 1,
            GemColor.RED: 2,
            GemColor.BLACK: 1,
        }
    ),
    Card(
        id=10,
        tier=1,
        points=0,
        bonus_color=GemColor.BLUE,
        cost={
            GemColor.WHITE: 1,
            GemColor.BLUE: 0,
            GemColor.GREEN: 1,
            GemColor.RED: 1,
            GemColor.BLACK: 1,
        }
    ),
    Card(
        id=11,
        tier=1,
        points=0,
        bonus_color=GemColor.BLUE,
        cost={
            GemColor.WHITE: 0,
            GemColor.BLUE: 1,
            GemColor.GREEN: 3,
            GemColor.RED: 1,
            GemColor.BLACK: 0,
        }
    ),
    Card(
        id=12,
        tier=1,
        points=0,
        bonus_color=GemColor.BLUE,
        cost={
            GemColor.WHITE: 0,
            GemColor.BLUE: 0,
            GemColor.GREEN: 0,
            GemColor.RED: 0,
            GemColor.BLACK: 3,
        }
    ),
    Card(
        id=13,
        tier=1,
        points=0,
        bonus_color=GemColor.BLUE,
        cost={
            GemColor.WHITE: 1,
            GemColor.BLUE: 0,
            GemColor.GREEN: 2,
            GemColor.RED: 2,
            GemColor.BLACK: 0,
        }
    ),
    Card(
        id=14,
        tier=1,
        points=0,
        bonus_color=GemColor.BLUE,
        cost={
            GemColor.WHITE: 0,
            GemColor.BLUE: 0,
            GemColor.GREEN: 2,
            GemColor.RED: 0,
            GemColor.BLACK: 2,
        }
    ),
    Card(
        id=15,
        tier=1,
        points=1,
        bonus_color=GemColor.BLUE,
        cost={
            GemColor.WHITE: 0,
            GemColor.BLUE: 0,
            GemColor.GREEN: 0,
            GemColor.RED: 4,
            GemColor.BLACK: 0,
        }
    ),
    Card(
        id=16,
        tier=1,
        points=0,
        bonus_color=GemColor.GREEN,
        cost={
            GemColor.WHITE: 2,
            GemColor.BLUE: 1,
            GemColor.GREEN: 0,
            GemColor.RED: 0,
            GemColor.BLACK: 0,
        }
    ),
    Card(
        id=17,
        tier=1,
        points=0,
        bonus_color=GemColor.GREEN,
        cost={
            GemColor.WHITE: 0,
            GemColor.BLUE: 2,
            GemColor.GREEN: 0,
            GemColor.RED: 2,
            GemColor.BLACK: 0,
        }
    ),
    Card(
        id=18,
        tier=1,
        points=0,
        bonus_color=GemColor.GREEN,
        cost={
            GemColor.WHITE: 1,
            GemColor.BLUE: 3,
            GemColor.GREEN: 1,
            GemColor.RED: 0,
            GemColor.BLACK: 0,
        }
    ),
    Card(
        id=19,
        tier=1,
        points=0,
        bonus_color=GemColor.GREEN,
        cost={
            GemColor.WHITE: 1,
            GemColor.BLUE: 1,
            GemColor.GREEN: 0,
            GemColor.RED: 1,
            GemColor.BLACK: 1,
        }
    ),
    Card(
        id=20,
        tier=1,
        points=0,
        bonus_color=GemColor.GREEN,
        cost={
            GemColor.WHITE: 1,
            GemColor.BLUE: 1,
            GemColor.GREEN: 0,
            GemColor.RED: 1,
            GemColor.BLACK: 2,
        }
    ),
    Card(
        id=21,
        tier=1,
        points=0,
        bonus_color=GemColor.GREEN,
        cost={
            GemColor.WHITE: 0,
            GemColor.BLUE: 1,
            GemColor.GREEN: 0,
            GemColor.RED: 2,
            GemColor.BLACK: 2,
        }
    ),
    Card(
        id=22,
        tier=1,
        points=0,
        bonus_color=GemColor.GREEN,
        cost={
            GemColor.WHITE: 0,
            GemColor.BLUE: 0,
            GemColor.GREEN: 0,
            GemColor.RED: 3,
            GemColor.BLACK: 0,
        }
    ),
    Card(
        id=23,
        tier=1,
        points=1,
        bonus_color=GemColor.GREEN,
        cost={
            GemColor.WHITE: 0,
            GemColor.BLUE: 0,
            GemColor.GREEN: 0,
            GemColor.RED: 0,
            GemColor.BLACK: 4,
        }
    ),
    Card(
        id=24,
        tier=1,
        points=0,
        bonus_color=GemColor.RED,
        cost={
            GemColor.WHITE: 3,
            GemColor.BLUE: 0,
            GemColor.GREEN: 0,
            GemColor.RED: 0,
            GemColor.BLACK: 0,
        }
    ),
    Card(
        id=25,
        tier=1,
        points=0,
        bonus_color=GemColor.RED,
        cost={
            GemColor.WHITE: 1,
            GemColor.BLUE: 0,
            GemColor.GREEN: 0,
            GemColor.RED: 1,
            GemColor.BLACK: 3,
        }
    ),
    Card(
        id=26,
        tier=1,
        points=0,
        bonus_color=GemColor.RED,
        cost={
            GemColor.WHITE: 0,
            GemColor.BLUE: 2,
            GemColor.GREEN: 1,
            GemColor.RED: 0,
            GemColor.BLACK: 0,
        }
    ),
    Card(
        id=27,
        tier=1,
        points=0,
        bonus_color=GemColor.RED,
        cost={
            GemColor.WHITE: 2,
            GemColor.BLUE: 0,
            GemColor.GREEN: 1,
            GemColor.RED: 0,
            GemColor.BLACK: 2,
        }
    ),
    Card(
        id=28,
        tier=1,
        points=0,
        bonus_color=GemColor.RED,
        cost={
            GemColor.WHITE: 2,
            GemColor.BLUE: 1,
            GemColor.GREEN: 1,
            GemColor.RED: 0,
            GemColor.BLACK: 1,
        }
    ),
    Card(
        id=29,
        tier=1,
        points=0,
        bonus_color=GemColor.RED,
        cost={
            GemColor.WHITE: 1,
            GemColor.BLUE: 1,
            GemColor.GREEN: 1,
            GemColor.RED: 0,
            GemColor.BLACK: 1,
        }
    ),
    Card(
        id=30,
        tier=1,
        points=0,
        bonus_color=GemColor.RED,
        cost={
            GemColor.WHITE: 2,
            GemColor.BLUE: 0,
            GemColor.GREEN: 0,
            GemColor.RED: 2,
            GemColor.BLACK: 0,
        }
    ),
    Card(
        id=31,
        tier=1,
        points=1,
        bonus_color=GemColor.RED,
        cost={
            GemColor.WHITE: 4,
            GemColor.BLUE: 0,
            GemColor.GREEN: 0,
            GemColor.RED: 0,
            GemColor.BLACK: 0,
        }
    ),
    Card(
        id=32,
        tier=1,
        points=0,
        bonus_color=GemColor.WHITE,
        cost={
            GemColor.WHITE: 0,
            GemColor.BLUE: 2,
            GemColor.GREEN: 2,
            GemColor.RED: 0,
            GemColor.BLACK: 1,
        }
    ),
    Card(
        id=33,
        tier=1,
        points=0,
        bonus_color=GemColor.WHITE,
        cost={
            GemColor.WHITE: 0,
            GemColor.BLUE: 0,
            GemColor.GREEN: 0,
            GemColor.RED: 2,
            GemColor.BLACK: 1,
        }
    ),
    Card(
        id=34,
        tier=1,
        points=0,
        bonus_color=GemColor.WHITE,
        cost={
            GemColor.WHITE: 0,
            GemColor.BLUE: 1,
            GemColor.GREEN: 1,
            GemColor.RED: 1,
            GemColor.BLACK: 1,
        }
    ),
    Card(
        id=35,
        tier=1,
        points=0,
        bonus_color=GemColor.WHITE,
        cost={
            GemColor.WHITE: 0,
            GemColor.BLUE: 3,
            GemColor.GREEN: 0,
            GemColor.RED: 0,
            GemColor.BLACK: 0,
        }
    ),
    Card(
        id=36,
        tier=1,
        points=0,
        bonus_color=GemColor.WHITE,
        cost={
            GemColor.WHITE: 0,
            GemColor.BLUE: 2,
            GemColor.GREEN: 2,
            GemColor.RED: 0,
            GemColor.BLACK: 0,
        }
    ),
    Card(
        id=37,
        tier=1,
        points=0,
        bonus_color=GemColor.WHITE,
        cost={
            GemColor.WHITE: 0,
            GemColor.BLUE: 1,
            GemColor.GREEN: 2,
            GemColor.RED: 1,
            GemColor.BLACK: 1,
        }
    ),
    Card(
        id=38,
        tier=1,
        points=0,
        bonus_color=GemColor.WHITE,
        cost={
            GemColor.WHITE: 3,
            GemColor.BLUE: 1,
            GemColor.GREEN: 0,
            GemColor.RED: 0,
            GemColor.BLACK: 1,
        }
    ),
    Card(
        id=39,
        tier=1,
        points=1,
        bonus_color=GemColor.WHITE,
        cost={
            GemColor.WHITE: 0,
            GemColor.BLUE: 0,
            GemColor.GREEN: 4,
            GemColor.RED: 0,
            GemColor.BLACK: 0,
        }
    )

]

BASE_TIER_2 = []
BASE_TIER_3 = []

NOBLES = []