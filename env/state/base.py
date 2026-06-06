from dataclasses import dataclass
from enums import NodeType
from player import Player
from noble import Noble
from card import Card

@dataclass
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