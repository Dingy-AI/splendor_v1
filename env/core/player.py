from dataclasses import dataclass
from env.core.enums import GemColor, CardColor
from env.core.card import Card
from env.core.noble import Noble

@dataclass(frozen=True)
class Player:
    id: int
    gems: dict[GemColor, int]
    bonuses: dict[CardColor, int]
    reserved_cards: list[Card]
    purchased_cards: list[Card]
    noble: list[Noble]
    points: int = 0