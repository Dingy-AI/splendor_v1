from splendor_v1.env.core.enums import GemColor


MAX_GEMS = 10
MAX_RESERVES = 3
VICTORY_REQUIREMENT = 15
MAX_PLAYER_COUNT = 2
MAX_NOBLE_COUNT = 3

GEM_SCALE_NORM = 10.0 #normalize the number of gems of each type in the hand

BONUS_NORM = 7.0 #normalize the number of cards of each type


POINT_SCALE_NORM = 15.0
NOBLE_SCALE_NORM = 4.0
CARD_COST_SCALE_NORM = 7.0 # the cost of a card for a certain gem does not exceed 7

MAX_DECK_SIZE_NORM = [0,40,30,20]

TIER1_NORM = 40
TIER2_NORM = 30
TIER3_NORM = 20

CARD_POINTS_NORM = 3.0 #normalizing the amount of points on the card itself

TWO_PLAYER_GEM_NORM = 4
THREE_PLAYER_GEM_NORM = 5
FOUR_PLAYER_GEM_NORM = 7

COLOR_ORDER = [

    GemColor.WHITE,
    GemColor.BLUE,
    GemColor.GREEN,
    GemColor.RED,
    GemColor.BLACK
]

COLOR_TO_INDEX = {
    color: i
    for i, color in enumerate(COLOR_ORDER)
}

### PLAYER ###
#player size ->
#player gems - 6
#player bonus - 5

#player reserved cards x3
#player card requirement - 5
#player card bonus - 5
#player card victory points - 1

#player points - 1

### PLAYER TOTAL = 45 ###

### ENEMY ###
#enemy gems - 6
#enemy bonus - 5

#enemy reserved cards x3
#enemy card requirements - 5
#enemy card bonus - 5
#enemy card victory points - 1

#enemy points - 1

### ENEMY TOTAL = 45 ###


### BANK ###
#bank -> 6
### BANK TOTAL = 6 ###

### DECK ###
#deck_size -> 3
### DECK TOTAL = 3 ###

### NOBLE ###
#nobles x3
#nobles requirement - 5
#nobles points - 1
### NOBLE TOTAL = 18 ###

### BOARD ###

#board -> cards x12
#card_cost -> 5
#card_bonus -> 5
#card_points -> 1

# board * cards = 12 * 11 = 132 
### BOARD TOTAL = 132


#PLAYER + ENEMY + BANK + DECK + NOBLE + BOARD
#45 + 45 + 6 + 3 + 18 + 132 = 249 = OBSERVATION SIZE
#249 + 3 Node type = 252 = OBSERVATION SIZE
#252 + 6 is hidden card = 258 = OBSERVATION SIZE
OBSERVATION_SIZE = 258

#### NORMALIZATION CONSTANTS ####

PLAYER_GEM_MAX = 13
PLAYER_GEM_BONUS_MAX = 5


CARD_COST_MAX = 7
CARD_POINT_MAX = 5

PLAYER_POINT_MAX  = 20

NOBLE_REQUIREMENT_MAX = 4
NOBLE_POINT_MAX = 3


# we take out 4 cards from the deck at the start
TIER_1_DECK_SIZE = 36
TIER_2_DECK_SIZE = 26
TIER_3_DECK_SIZE = 16

TWO_PLAYER_BANK_GEM_MAX = 4
TWO_PLAYER_BANK_GOLD_MAX = 5