from splendor_v1.env.core.enums import GemColor
from splendor_v1.env.core.cost_lookup_table_v3 import T1_PAYMENT_LOOKUP, T2_PAYMENT_LOOKUP, T3_PAYMENT_LOOKUP


GEM_ACTIONS = [
    (GemColor.WHITE,), (GemColor.BLUE,), (GemColor.GREEN,), (GemColor.RED,), (GemColor.BLACK,),
    (GemColor.WHITE, GemColor.WHITE), (GemColor.BLUE, GemColor.BLUE), (GemColor.GREEN, GemColor.GREEN), (GemColor.RED, GemColor.RED), (GemColor.BLACK, GemColor.BLACK),
    
    (GemColor.WHITE, GemColor.BLUE), (GemColor.WHITE, GemColor.GREEN), (GemColor.WHITE, GemColor.RED),
    (GemColor.WHITE, GemColor.BLACK), (GemColor.BLUE, GemColor.GREEN), (GemColor.BLUE, GemColor.RED),
    (GemColor.BLUE, GemColor.BLACK), (GemColor.GREEN, GemColor.RED), (GemColor.GREEN, GemColor.BLACK),
    (GemColor.RED, GemColor.BLACK),

    (GemColor.WHITE, GemColor.BLUE, GemColor.GREEN), (GemColor.WHITE, GemColor.BLUE, GemColor.RED),
    (GemColor.WHITE, GemColor.BLUE, GemColor.BLACK), (GemColor.WHITE, GemColor.GREEN, GemColor.RED),
    (GemColor.WHITE, GemColor.GREEN, GemColor.BLACK), (GemColor.WHITE, GemColor.RED, GemColor.BLACK),
    (GemColor.BLUE, GemColor.GREEN, GemColor.RED), (GemColor.BLUE, GemColor.GREEN, GemColor.BLACK),
    (GemColor.BLUE, GemColor.RED, GemColor.BLACK), (GemColor.GREEN, GemColor.RED, GemColor.BLACK)
]

GEM_ACTION_TO_ID = {
    combo: idx
    for idx, combo in enumerate(GEM_ACTIONS)
}


DISCARD_COLORS = [
    GemColor.WHITE,
    GemColor.BLUE,
    GemColor.GREEN,
    GemColor.RED,
    GemColor.BLACK,
    GemColor.GOLD
]

DISCARD_COLOR_TO_ID = {
    color: i
    for i, color in enumerate(DISCARD_COLORS)
}

MAX_VISIBLE_CARDS = 12
MAX_RESERVED_CARDS = 3
MAX_NOBLES = 5

# TAKE_GEMS_START = 0
# RESERVE_START = 20
# RESERVE_DECK_START = 32
# BUY_START = 35
# BUY_RESERVED_START = 47
# DISCARD_START = 50
# NOBLE_START = 56


T1_PAYMENT_COUNT = len(T1_PAYMENT_LOOKUP)
T2_PAYMENT_COUNT = len(T2_PAYMENT_LOOKUP)
T3_PAYMENT_COUNT = len(T3_PAYMENT_LOOKUP)


TAKE_GEMS_START = 0
RESERVE_START = 30
RESERVE_DECK_START = 42
BUY_START = 45
# BUY_RESERVED_START = 57
# DISCARD_START = 60
# NOBLE_START = 66
# ACTION_END = 71


VISIBLE_SLOTS = 4
RESERVED_SLOTS = 3

BUY_T1_START = BUY_START
BUY_T2_START = BUY_T1_START + (VISIBLE_SLOTS * T1_PAYMENT_COUNT)
BUY_T3_START = BUY_T2_START + (VISIBLE_SLOTS * T2_PAYMENT_COUNT)
BUY_RESERVED_START = BUY_T3_START + (4 * T3_PAYMENT_COUNT)


BUY_END = BUY_RESERVED_START  + (RESERVED_SLOTS * T3_PAYMENT_COUNT)

# BUY_RESERVED_START = 57
DISCARD_START = BUY_END
NOBLE_START = DISCARD_START + 6
ACTION_END = NOBLE_START + 5


# going to need to upgrade this
# Take gems     30
# Reserve       15
# Buy           15
# Discard        6
# Noble          5
# -------------
# Total         71
NOBLE_2_PLAYER = 3

ACTION_SPACE_SIZE = 128
ACTION_SPACE_SIZE = (
        len(GEM_ACTIONS) + 
        12 + #number of visible reserves 
        3 + #number of hidden reserves
        T1_PAYMENT_COUNT * VISIBLE_SLOTS + # number of ways to buy tier 1
        T2_PAYMENT_COUNT * VISIBLE_SLOTS + # number of ways to buy tier 2
        T3_PAYMENT_COUNT * VISIBLE_SLOTS + # number of ways to buy tier 3
        T3_PAYMENT_COUNT * RESERVED_SLOTS + # number of ways to buy reserve cards
        len(DISCARD_COLORS) + #number of discord actions
        NOBLE_2_PLAYER #number of nobles for two player

)