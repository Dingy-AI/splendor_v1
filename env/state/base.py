from dataclasses import dataclass
from env.core.enums import NodeType, GemColor
from env.core.player import Player
from env.core.noble import Noble
from env.core.card import Card

@dataclass
class GameState:
    node_type: NodeType
    players: list[Player]
    bank: dict[GemColor, int]
    nobles: list[Noble]
    visible_cards: dict[int, list[Card]]
    decks: dict[int, list[Card]]
    current_player: int
    turn_number: int
    winners: list[int]
    game_over: bool = False
    end_triggered: bool = False