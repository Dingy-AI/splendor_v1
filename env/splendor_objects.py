from dataclasses import dataclass

from enum import Enum

MAX_GEMS = 10
MAX_RESERVES = 3

@dataclass
class GemColor(Enum):
    WHITE = 0
    BLUE = 1
    GREEN = 2
    RED = 3
    BROWN = 4
    GOLD = 5

@dataclass
class Card:
    id: int
    tier: int
    points: int
    bonus_color: int
    cost: dict[GemColor, int]

@dataclass
class Noble:
    id: int
    points: int
    requirement: dict[GemColor,int,]

@dataclass
class Player:
    gems: dict[GemColor, int]
    bonuses: dict[GemColor, int]
    reserved_cards: list[Card]
    purchased_cards: list[Card]