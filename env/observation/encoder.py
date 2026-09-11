from splendor_v1.env.state.base import GameState
import numpy as np
from splendor_v1.env.core.player import Player
from splendor_v1.env.core.enums import GemColor, NodeType
from splendor_v1.env.core.card import Card
from splendor_v1.env.core.noble import Noble

#TODO NEED TO DO A COMPLETE REWRITE OF THE ENCODER

FOUR_PLAYER_GEM_NORM = 7
from splendor_v1.env.core.constants import CARD_COST_MAX, CARD_POINT_MAX, COLOR_ORDER, GEM_SCALE_NORM, BONUS_NORM, MAX_RESERVES, CARD_COST_SCALE_NORM, CARD_POINTS_NORM, NOBLE_POINT_MAX, NOBLE_REQUIREMENT_MAX, PLAYER_GEM_BONUS_MAX, PLAYER_POINT_MAX, POINT_SCALE_NORM, MAX_PLAYER_COUNT, TIER_1_DECK_SIZE, TIER_2_DECK_SIZE, TIER_3_DECK_SIZE, TWO_PLAYER_BANK_GEM_MAX, TWO_PLAYER_BANK_GOLD_MAX, TWO_PLAYER_GEM_NORM, THREE_PLAYER_GEM_NORM, MAX_DECK_SIZE_NORM, NOBLE_SCALE_NORM

class ObservationEncoder:
    def __init__(self):
        self._card_encoding_cache = {}
        self._empty_card_encoding = [0.0] * 11
        self._gem_colors = tuple(GemColor)
        self._bank_normalizers = tuple(
            (
                color,
                TWO_PLAYER_BANK_GOLD_MAX
                if color == GemColor.GOLD
                else TWO_PLAYER_BANK_GEM_MAX,
            )
            for color in self._gem_colors
        )
        self._reserved_card_encoding_cache = {}
        self._empty_reserved_card_encoding = [0.0] * 12
        self._hidden_reserved_card_encoding = [0.0] * 11 + [1.0]
        self._reserve_padding = tuple(
            [0.0] * (12 * count) for count in range(MAX_RESERVES + 1)
        )
        

    def encoder(self, state: GameState):
        self.num_players = len(state.players)

        if self.num_players == 2:
            self.player_gem_norm = TWO_PLAYER_GEM_NORM
        elif self.num_players == 3:
            self.player_gem_norm = THREE_PLAYER_GEM_NORM
        elif self.num_players == 4:
            self.player_gem_norm = FOUR_PLAYER_GEM_NORM
        else:
            raise ValueError(f"Unsupported player count: {self.num_players}")

        features = self._encode_players(state.players, state.current_player)
        features.extend(self._encode_bank(state.bank))
        features.extend(self._encode_decks(state.decks))
        features.extend(self._encode_nobles(state.nobles))
        features.extend(self._encode_board(state.visible_cards))
        features.extend(self._encode_node_type(state.node_type))

        # Each observation owns its array; later calls cannot overwrite replay data.
        return np.array(features, dtype=np.float32)

    def _encode_node_type(
        self,
        node_type,
    ):
        feature = [
            0.0,
            0.0,
            0.0,
        ]

        if (
            node_type
            == NodeType.MAIN_DECISION
        ):
            feature[0] = 1.0

        elif (
            node_type
            == NodeType.OVERFLOW_DISCARD
        ):
            feature[1] = 1.0

        elif (
            node_type
            == NodeType.NOBLE_CLAIM
        ):
            feature[2] = 1.0

        else:
            raise ValueError(
                f"Unknown node type: "
                f"{node_type}"
            )

        return feature

        
    def _encode_players(self, players, current_player):
        feature = self._encode_single_player(
            players[current_player], is_current_player=True
        )
        feature.extend(
            self._encode_single_player(
                players[1 - current_player], is_current_player=False
            )
        )
        return feature

    def slow_encode_single_player(self, player:Player):
        feature_gems = []
        feature_bonus = []
        feature_reserved_cards = []
        for color in GemColor:
            if (color in player.gems):
                feature_gems.append(player.gems[color] / GEM_SCALE_NORM)
            if color != GemColor.GOLD and color in player.bonuses:
                feature_bonus.append(player.bonuses[color] / BONUS_NORM)

        if len(player.reserved_cards) == 0:
            feature_reserved_cards = [0] * 33 #TODO might need to change this 
        else:
            # for i in len(player.reserved_cards):
            for reserved_index, card in enumerate(player.reserved_cards):
                # need to refactor this into card encoding in the future
                feature_bonus_color = [0] * 5
                for color in GemColor:
                    if color == GemColor.GOLD:
                        continue
                    # feature_reserved_cards.append(card.cost[color] / CARD_COST_SCALE_NORM)
                    feature_reserved_cards.append(card.cost[color])

                    if card.bonus_color == color:
                        feature_bonus_color[color.value] = 1

                feature_reserved_cards.extend(feature_bonus_color)
                # feature_reserved_cards.append(card.points / CARD_POINTS_NORM)
                feature_reserved_cards.append(card.points)

            if len(player.reserved_cards) < MAX_RESERVES:
                feature_reserved_cards.extend([0] * 11 * (MAX_RESERVES - len(player.reserved_cards)))

        feature = feature_gems + feature_bonus + feature_reserved_cards + [player.points / POINT_SCALE_NORM]
        return feature


    def _encode_single_player(self, player: Player, is_current_player: bool):
        gems = player.gems
        bonuses = player.bonuses
        gem_norm = self.player_gem_norm
        feature = [gems[color] / gem_norm for color in self._gem_colors]
        feature.extend(
            [bonuses[color] / PLAYER_GEM_BONUS_MAX for color in COLOR_ORDER]
        )

        reserved_cards = player.reserved_cards
        hidden_flags = player.reserved_card_hidden
        if len(reserved_cards) != len(hidden_flags):
            raise ValueError(
                "reserved_cards and reserved_card_hidden must have the same length."
            )

        for card, is_hidden in zip(reserved_cards, hidden_flags):
            feature.extend(
                self._encode_reserved_card(card, is_hidden, is_current_player)
            )

        missing_reserves = MAX_RESERVES - len(reserved_cards)
        if missing_reserves > 0:
            feature.extend(self._reserve_padding[missing_reserves])

        feature.append(player.points / PLAYER_POINT_MAX)
        return feature

    def _encode_reserved_card(self, card, is_hidden, is_current_player):
        if not is_current_player and is_hidden:
            return self._hidden_reserved_card_encoding

        if card is None:
            return self._empty_reserved_card_encoding

        cached = self._reserved_card_encoding_cache.get(card.id)
        if cached is not None:
            return cached

        encoded = self._encode_card(card) + [0.0]
        self._reserved_card_encoding_cache[card.id] = encoded
        return encoded
    def _encode_bank(self, bank):
        return [bank[color] / norm for color, norm in self._bank_normalizers]
    
    def _encode_decks(self, decks):

        feature = [
            len(decks[1]) / TIER_1_DECK_SIZE,
            len(decks[2]) / TIER_2_DECK_SIZE,
            len(decks[3]) / TIER_3_DECK_SIZE,
        ]

        return feature

    def _encode_nobles(self, nobles:list[Noble]):
        feature = []

        for noble in nobles:
            if noble == None:
                feature.extend([0] * 6)
                continue
            # Requirements
            for color in COLOR_ORDER:
                feature.append(
                    noble.requirement[color]
                    / NOBLE_REQUIREMENT_MAX
                )


            # Victory points
            feature.append(
                noble.points
                / NOBLE_POINT_MAX
            )
        return feature
    
    def _encode_board(
        self,
        visible_cards: dict[int, list[Card]],
    ):

        feature = []

        for tier in (1, 2, 3):

            cards = visible_cards[tier]

            for card in cards:
                feature.extend(
                    self._encode_card(card)
                )

            missing_cards = 4 - len(cards)

            feature.extend(
                [0] * (
                    missing_cards * 11
                )
            )

        return feature

    # We can't delete this since we are using it if encode card finds it missing
    def slow_encode_card(self, card):
        vec = []
        if card != None:
            # 3. cost vector
            for color in GemColor:

                if color == GemColor.GOLD:
                    continue

                vec.append(card.cost.get(color, 0)
                           / CARD_COST_MAX
                )
            # 2. bonus color one-hot
            for color in GemColor:
                if color == GemColor.GOLD:
                    continue
                vec.append(1.0 if card.bonus_color == color else 0.0)

            # 1. points
            vec.append(card.points / CARD_POINT_MAX)
        return vec

    def _encode_card(self, card):

        if card is None:
            return self._empty_card_encoding

        cached_encoding = self._card_encoding_cache.get(
            card.id
        )

        if cached_encoding is not None:
            return cached_encoding

        encoded_card = self.slow_encode_card(card)

        self._card_encoding_cache[
            card.id
        ] = encoded_card

        return encoded_card



# Sections:
# Current player gems (6)
# Current player bonuses (5)
# Current player points (1)
# Current player reserves (3*features +1 for visible)


# Other player gems (6*3)
# Other player bonuses (5*3)
# Other player points (3)

# other player reserve (3 (num reserves) * 3 (players)*(features + 1 for visible))
# Next player etc... (1) 


# Bank (6 incl gold) (6)
# Visible cards (12 cards × features) (features = 11 = 5 colors + 5 bonuses + 1 points)
# Nobles (max 5 × features)
# Meta (turn, node type)

