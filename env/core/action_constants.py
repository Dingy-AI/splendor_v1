from env.core.enums import GemColor


ACTION_SPACE_SIZE = 128

GEM_ACTIONS = [
    (GemColor.WHITE), (GemColor.BLUE), (GemColor.GREEN), (GemColor.RED), (GemColor.BLACK),
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


DISCARD_COLORS = [
    GemColor.WHITE,
    GemColor.BLUE,
    GemColor.GREEN,
    GemColor.RED,
    GemColor.BLACK,
    GemColor.GOLD
]
MAX_VISIBLE_CARDS = 12
MAX_RESERVED_CARDS = 3
MAX_NOBLES = 5

TAKE_GEMS_START = 0
RESERVE_START = 20
BUY_START = 35
DISCARD_START = 50
NOBLE_START = 56