from core.card import Card
from core.enums import GemColor


BASE_TIER_1 = [
    Card(
        id=0,
        tier=1,
        points=0,
        bonus_color=GemColor.GREEN,
        cost={
            GemColor.WHITE: 0,
            GemColor.BLUE: 1,
            GemColor.GREEN: 2,
            GemColor.RED: 0,
            GemColor.BLACK: 0,
        }
    )

]

BASE_TIER_2 = []
BASE_TIER_3 = []

NOBLES = []