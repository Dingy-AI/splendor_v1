from dataclasses import dataclass
from splendor_v1.env.core.enums import NodeType, GemColor
from splendor_v1.env.core.player import Player
from splendor_v1.env.core.noble import Noble
from splendor_v1.env.core.card import Card

from copy import deepcopy
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
    noble_taken: bool = False

    def clone(self):
        return deepcopy(self)