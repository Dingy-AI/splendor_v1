from dataclasses import dataclass
from enum import Enum

# class CardColor(Enum):
#     WHITE = 0
#     BLUE = 1
#     GREEN = 2
#     RED = 3
#     BLACK = 4

class GemColor(Enum):
    WHITE = 0
    BLUE = 1
    GREEN = 2
    RED = 3
    BLACK = 4
    GOLD = 5

class ActionType(Enum):
    BUY_VISIBLE = 0
    BUY_RESERVED = 1
    RESERVE_VISIBLE = 2 
    RESERVE_TOP_DECK = 3
    TAKE_GEMS = 4
    DISCARD_GEMS = 5
    TAKE_NOBLE = 6

class NodeType(Enum):
    MAIN_DECISION = 0
    NOBLE_CLAIM = 1
    OVERFLOW_DISCARD = 2
    END_TURN = 3

# MAIN_ACTION
#    ↓
# apply action
#    ↓
# check overflow/noble
#    ↓
# if overflow:
#        OVERFLOW_NODE
#             ↓
#        discard 1 gem
#             ↓
#        back to overflow check
#    ↓
# end turn
