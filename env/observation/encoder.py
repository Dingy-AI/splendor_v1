from state.base import GameState

class ObservationEncoder:
    def __init__(self):
        pass

    def encoder(self, state:GameState):

        features = []
        #need different functions to encode different things
        state.node_type

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
        pass 
    
    # we need a normalization strategy for our encoding? :O
    # we have to do turn order in the feature 
    # need to know deck size as well 
    # deck encoder? 

    def _encode_player(self, player):
        return None

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