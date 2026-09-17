HIDDEN_SIZE = 512
NUM_HIDDEN_LAYERS = 2
VALUE_OUTPUT_SIZE = 1

PLAYER_START = 0
PLAYER_END = 48

ENEMY_START = 48
ENEMY_END = 96

BANK_START = 96
BANK_END = 102

DECK_START = 102
DECK_END = 105

NOBLES_START = 105
NOBLES_END = 123

BOARD_START = 123
BOARD_END = 255

NODE_TYPE_START = 255
NODE_TYPE_END = 258


# OBSERVATION 258

# CURRENT PLAYER                  48
#     gems                         6
#     bonuses                      5
#     reserve 0                   12
#     reserve 1                   12
#     reserve 2                   12
#     points                       1

# OPPONENT                        48
#     same structure

# BANK                             6

# DECK SIZES                       3
#     Tier 1
#     Tier 2
#     Tier 3

# NOBLES                          18
#     noble 0                      6
#     noble 1                      6
#     noble 2                      6

# VISIBLE BOARD                  132
#     Tier 1 × 4 cards            44
#     Tier 2 × 4 cards            44
#     Tier 3 × 4 cards            44

# NODE TYPE                        3
# ----------------------------------
# TOTAL                          258



# | Action IDs  | Count | Meaning                          |
# | ----------- | ----: | -------------------------------- |
# | `0–29`      |    30 | `TAKE_GEMS`                      |
# | `30–41`     |    12 | `RESERVE_VISIBLE`                |
# | `42–44`     |     3 | `RESERVE_TOP_DECK`               |
# | `45–184`    |   140 | `BUY_VISIBLE` Tier 1 (`35 × 4`)  |
# | `185–352`   |   168 | `BUY_VISIBLE` Tier 2 (`42 × 4`)  |
# | `353–796`   |   444 | `BUY_VISIBLE` Tier 3 (`111 × 4`) |
# | `797–1129`  |   333 | `BUY_RESERVED` (`111 × 3`)       |
# | `1130–1135` |     6 | `DISCARD_GEMS`                   |
# | `1136–1138` |     3 | `TAKE_NOBLE`                     |
# 1139 actions total 

# 0–4       take one color
# 5–9       take two of same color
# 10–19     take two distinct colors
# 20–29     take three distinct colors

# 1130   discard WHITE
# 1131   discard BLUE
# 1132   discard GREEN
# 1133   discard RED
# 1134   discard BLACK
# 1135   discard GOLD

# Tier 1 BUY start = 45

# slot 0:  45–79
# slot 1:  80–114
# slot 2: 115–149
# slot 3: 150–184

# slot 0: 185–226
# slot 1: 227–268
# slot 2: 269–310
# slot 3: 311–352

# slot 0: 353–463
# slot 1: 464–574
# slot 2: 575–685
# slot 3: 686–796

# BUY_RESERVED start = 797

# reserve 0: 797–907
# reserve 1: 908–1018
# reserve 2: 1019–1129

# 0–29       TAKE_GEMS
# 30–41      RESERVE_VISIBLE
# 42–44      RESERVE_TOP_DECK
# 45–796     BUY_VISIBLE
# 797–1129   BUY_RESERVED
# 1130–1135  DISCARD_GEMS
# 1136–1138  TAKE_NOBLE