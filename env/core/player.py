from dataclasses import dataclass, field
from splendor_v1.env.core.enums import GemColor
from splendor_v1.env.core.card import Card
from splendor_v1.env.core.noble import Noble

@dataclass
class Player:
    id: int
    gems: dict[GemColor, int] 
    bonuses: dict[GemColor, int] 
    reserved_cards: list[Card]
    purchased_cards: list[Card]
    noble: list[Noble]

    points: int = 0


    # gems: dict = field(default_factory=dict)
    # bonuses: dict = field(default_factory=dict)
    # reserved_cards: list = field(default_factory=list)
    # purchased_cards: list = field(default_factory=list)
    # noble: list = field(default_factory=list)

