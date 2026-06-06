from dataclasses import dataclass
from core.enums import GemColor
from core.card import Card

@dataclass(frozen=True)
class Player:
    gems: dict[GemColor, int]
    bonuses: dict[GemColor, int]
    reserved_cards: list[Card]
    purchased_cards: list[Card]
    points: int = 0