from enum import Enum

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

class NodeType(Enum):
    MAIN_DECISION = 1
    OVERFLOW_DISCARD = 2
    NOBLE_CLAIM = 3