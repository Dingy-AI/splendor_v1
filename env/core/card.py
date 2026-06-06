from dataclasses import dataclass
from core.enums import GemColor

@dataclass(frozen=True)
class Card:
    id: int
    tier: int
    points: int
    bonus_color: GemColor
    cost: dict[GemColor, int]