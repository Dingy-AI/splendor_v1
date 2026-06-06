from dataclasses import dataclass
from core.enums import NodeType
from core.player import Player
from core.noble import Noble
from core.card import Card

@dataclass(frozen=True)
class GameState:
    node_type: NodeType
    players: list[Player]
    bank: dict
    nobles: list[Noble]
    visible_cards: dict[int, list[Card]]
    decks: list[Card]
    current_player: int
    turn_number: int
    game_over: bool = False