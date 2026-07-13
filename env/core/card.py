from dataclasses import dataclass
from splendor_v1.env.core.enums import CardColor

@dataclass(frozen=True)
class Card:
    id: int
    tier: int
    points: int
    bonus_color: CardColor
    cost: dict[CardColor, int]