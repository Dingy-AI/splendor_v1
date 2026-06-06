from dataclasses import dataclass
from enums import GemColor

@dataclass
class Card:
    id: int
    tier: int
    points: int
    bonus_color: int
    cost: dict[GemColor, int]