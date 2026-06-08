from state.base import GameState
import numpy as np
from core.player import Player
from core.enums import GemColor, CardColor
from core.card import Card

from core.constants import GEM_SCALE_NORM, BONUS_NORM, MAX_RESERVES, CARD_COST_SCALE_NORM, CARD_POINTS_NORM, POINT_SCALE_NORM

class ObservationEncoder:
    def __init__(self):
        pass

    def encoder(self, state:GameState):

        features = []
        #need different functions to encode different things


        #player encoder
        #noble encoder
        #deck encoder 
        #bank encoder 
        #board encoder
        #meta encoder? -> not sure if I really need this maybe just what the node type is?
            #looks like I will just need to say what node it is
            # also probably just say how many turns it has been

        self._encode_player(state.players,state.current_player)


        return np.array(features, dtype=np.float32)


        #need to convert things everything into a number :)
        # we have player features 
        # we have our player feature 
        # we have opponent player feature 
        # ideally we want to always have next player at the same place
        # gems -> 6
        # cards -> we don't need our cards, we just need the bonuses -> 5?
        # reserve cards -> points, bonus_one_hot(5), cost(5), if card is unknown we use torch.randn(11)?
        # 11 * 3

        #(6+5+ (11*3)) -> 44 

        # do this for players in current turn order
        # 44*4 
        #board state
        # each card is 11 features there are 12 cards = 132 features
        # deck size - normalize this -> remaining_cards/max_cards 3
        # noble -> each noble has requirement(5), points (1) -> 6*5
        # need to know bank state + 6

        # need to track each opponents score so +1 instead of 44 -> 45
        #players 45 * 4 
        #everything else 132 + 3 + 30 + 6 
        # 354 input features

        #work on normalization and encoder next time :3
         
    
    # we need a normalization strategy for our encoding? :O
    # we have to do turn order in the feature 
    # need to know deck size as well 
    # deck encoder? 

    def _encode_players(self, players:list[Player], current_player:int):
        
        feature = []
        players = players[current_player:] + players[:current_player]

        for player in players:
            feature = feature + self._encode_single_player(player)

        return feature

    def _encode_single_player(self, player:Player):
        feature_gems = []
        feature_bonus = []
        feature_reserved_cards = []
        for color in GemColor:
            feature_gems.append(player.gems[color] / GEM_SCALE_NORM)
            if color != GemColor.GOLD:
                feature_bonus.append(player.bonuses[color] / BONUS_NORM)

        if len(player.reserved_cards) == 0:
            feature_reserved_cards = [0] * 33
        else:
            for i in len(player.reserved_cards):
                # need to refactor this into card encoding in the future
                feature_reserved_cards.append(player.reserved_cards[i].points)
                feature_bonus_color = [0] * 5
                for color in CardColor:

                    feature_reserved_cards.append(player.reserved_cards[i].cost[color] / CARD_COST_SCALE_NORM)
                    if player.reserved_cards[i].bonus_color == color:
                        feature_bonus_color[color.value] = 1

                feature_reserved_cards.append(feature_bonus_color)
                feature_reserved_cards.append([player.reserved_cards[i].points] / CARD_POINTS_NORM)
            if len(player.reserved_cards) < MAX_RESERVES:
                feature_reserved_cards.append([0] * 11 * (MAX_RESERVES - len(player.reserved_cards)))

        feature = feature_gems + feature_bonus + feature_reserved_cards + [player.points / POINT_SCALE_NORM]
        return feature
    

    def _encode_card(self, card):
        vec = []

        # 1. points
        vec.append(card.points)

        # 2. bonus color one-hot
        for color in self.gem_colors:
            vec.append(1.0 if card.bonus_color == color else 0.0)

        # 3. cost vector
        for color in self.gem_colors:
            vec.append(card.cost.get(color, 0))

        return vec



# Sections:
# Current player gems (5)
# Current player bonuses (5)
# Current player points (1)
# Next player etc...
# Bank (6 incl gold) (6)
# Visible cards (12 cards × features)
# Nobles (max 5 × features)
# Meta (turn, node type)