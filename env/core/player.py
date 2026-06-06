from dataclasses import dataclass
from env.core.enums import GemColor
from env.core.card import Card
from env.core.noble import Noble

@dataclass(frozen=True)
class Player:
    gems: dict[GemColor, int]
    bonuses: dict[GemColor, int]
    reserved_cards: list[Card]
    purchased_cards: list[Card]
    noble: list[Noble]
    points: int = 0