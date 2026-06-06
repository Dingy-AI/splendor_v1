from dataclasses import dataclass
from enum import Enum

@dataclass(frozen=True)
class GemColor(Enum):
    WHITE = 0
    BLUE = 1
    GREEN = 2
    RED = 3
    BLACK = 4
    GOLD = 5

@dataclass(frozen=True)
class ActionType(Enum):
    TAKE_GEMS = 0
    BUY_CARD = 1
    RESERVE_CARD = 2
    FORCED_DISCARD = 4 #CHATGPT does not recommend this in here. Might move it to rules?
    PICK_NOBLE = 5

@dataclass(frozen=True)
class NodeType(Enum):
    MAIN_DECISION = 1
    NOBLE_CLAIM = 3
    OVERFLOW_DISCARD = 2

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
