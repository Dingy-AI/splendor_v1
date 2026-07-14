from dataclasses import dataclass
from splendor_v1.env.core.enums import GemColor

@dataclass(frozen=True)
class Card:
    id: int
    tier: int
    points: int
    bonus_color: GemColor
    cost: dict[GemColor, int]