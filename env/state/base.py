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

    # def clone(self):
    #     return deepcopy(self)

    def clone(self):
        return GameState(
            node_type=self.node_type,

            players=[
                player.clone()
                for player in self.players
            ],

            bank=self.bank.copy(),

            # Copy the containers, but share Card/Noble objects.
            nobles=self.nobles.copy(),

            visible_cards={
                tier: cards.copy()
                for tier, cards in self.visible_cards.items()
            },

            decks={
                tier: cards.copy()
                for tier, cards in self.decks.items()
            },

            current_player=self.current_player,
            turn_number=self.turn_number,

            winners=self.winners.copy(),

            game_over=self.game_over,
            end_triggered=self.end_triggered,
            noble_taken=self.noble_taken,
        )