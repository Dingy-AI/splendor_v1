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
        

    def encoder(self, state:GameState):


        self.num_players = len(state.players)

        if  self.num_players  == 2:
            self.player_gem_norm = TWO_PLAYER_GEM_NORM
        elif  self.num_players  == 3:
            self.player_gem_norm = THREE_PLAYER_GEM_NORM
        elif  self.num_players  == 4:
            self.player_gem_norm = FOUR_PLAYER_GEM_NORM
        else:
            raise ValueError(
                f"Unsupported player count: { self.num_players }"
            )

        features = []

        features = self._encode_players(state.players,state.current_player)

        # print(len(features))
        features = features + self._encode_bank(state.bank)
        # print(len(features))

        features = features + self._encode_decks(state.decks)
        # print(len(features))

        features = features + self._encode_nobles(state.nobles)
        # print(len(features))

        features = features + self._encode_board(state.visible_cards)
        # print(len(features))
        features += self._encode_node_type(
            state.node_type
        )


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

        
    def _encode_players(
        self,
        players,
        current_player,
    ):
        feature = []

        # Current player
        feature.extend(
            self._encode_single_player(
                players[current_player],
                is_current_player=True,
            )
        )

        # Opponent
        opponent = (
            1 - current_player
        )

        feature.extend(
            self._encode_single_player(
                players[opponent],
                is_current_player=False,
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


    def _encode_single_player(
        self,
        player: Player,
        is_current_player: bool,
    ):

        feature = []

        # -------------------------
        # Gems
        # -------------------------

        for color in GemColor:
            feature.append(
                player.gems[color]
                / self.player_gem_norm
            )

        # -------------------------
        # Permanent bonuses
        # -------------------------

        for color in COLOR_ORDER:
            feature.append(
                player.bonuses[color]
                / PLAYER_GEM_BONUS_MAX
            )

        # -------------------------
        # Reserved cards
        # -------------------------
        if (
            len(player.reserved_cards)
            != len(player.reserved_card_hidden)
        ):
            raise ValueError(
                "reserved_cards and "
                "reserved_card_hidden must have "
                "the same length."
            )

        for card, is_hidden in zip(
            player.reserved_cards,
            player.reserved_card_hidden,
        ):
            feature.extend(
                self._encode_reserved_card(
                    card=card,
                    is_hidden=is_hidden,
                    is_current_player=(
                        is_current_player
                    ),
                )
            )

        # -------------------------
        # Empty reserve slots
        # -------------------------

        missing_reserves = (
            MAX_RESERVES
            - len(player.reserved_cards)
        )

        if missing_reserves > 0:
            feature.extend(
                [0.0]
                * (
                    12
                    * missing_reserves
                )
            )

        # -------------------------
        # Points
        # -------------------------

        feature.append(
            player.points
            / PLAYER_POINT_MAX
        )

        return feature

    def _encode_reserved_card(
        self,
        card,
        is_hidden,
        is_current_player,
    ):
        # You always know your own reserved card,
        # even if you drew it from the top deck.
        if is_current_player:
            return (
                list(self._encode_card(card))
                + [0.0]
            )

        # Opponent's face-down reserved card.
        if is_hidden:
            return (
                [0.0] * 11
                + [1.0]
            )

        # Opponent publicly reserved this card.
        return (
            list(self._encode_card(card))
            + [0.0]
        )
    def _encode_bank(self, bank):

        feature = []

        for color in GemColor:

            if color == GemColor.GOLD:
                feature.append(
                    bank[color]
                    / TWO_PLAYER_BANK_GOLD_MAX
                )
            else:
                feature.append(
                    bank[color]
                    / TWO_PLAYER_BANK_GEM_MAX
                )

        return feature
    
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

