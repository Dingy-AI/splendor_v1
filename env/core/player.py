from dataclasses import dataclass
from enums import GemColor
from card import Card

@dataclass
class Player:
    gems: dict[GemColor, int]
    bonuses: dict[GemColor, int]
    reserved_cards: list[Card]
    purchased_cards: list[Card]
    points: int = 0