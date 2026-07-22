from dataclasses import dataclass
from enum import Enum

# class CardColor(Enum):
#     WHITE = 0
#     BLUE = 1
#     GREEN = 2
#     RED = 3
#     BLACK = 4

class CardType(Enum):
    T1_3 = 0
    T1_21 = 1
    T1_4 = 2
    T1_22 = 3
    T1_1111 = 4
    T1_311 = 5
    T1_221 = 6 
    T1_2111 = 7
    T2_5 = 8
    T2_6 = 9
    T2_421 = 10
    T2_322 = 11
    T2_53 = 12
    T2_332 = 13
    T3_7 = 14
    T3_73 = 15
    T3_633 = 16
    T3_5333 = 17

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
    PAYMENT = 4

# MAIN_ACTION
#    ↓
# apply action -> If action is payment -> payment action -> apply gems taken
#    ↓                                                          ↓
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

#payment will have to be about how many gold gems to use instead of anything else
# there will be 5 payment options and it will depend on how many gold gems to use?

