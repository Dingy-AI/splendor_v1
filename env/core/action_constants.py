from splendor_v1.env.core.enums import GemColor


ACTION_SPACE_SIZE = 128

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


TAKE_GEMS_START = 0
RESERVE_START = 30
RESERVE_DECK_START = 42
BUY_START = 45
BUY_RESERVED_START = 57
DISCARD_START = 60
NOBLE_START = 66
ACTION_END = 71