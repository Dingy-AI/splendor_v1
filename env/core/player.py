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
    nobles: list[Noble]

    points: int = 0


    def clone(self):
        return Player(
            id=self.id,

            gems=self.gems.copy(),

            bonuses=self.bonuses.copy(),

            reserved_cards=self.reserved_cards.copy(),

            purchased_cards=self.purchased_cards.copy(),

            nobles=self.nobles.copy(),

            points=self.points,
        )

    # gems: dict = field(default_factory=dict)
    # bonuses: dict = field(default_factory=dict)
    # reserved_cards: list = field(default_factory=list)
    # purchased_cards: list = field(default_factory=list)
    # noble: list = field(default_factory=list)

