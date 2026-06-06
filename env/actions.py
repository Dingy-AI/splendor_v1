from enum import Enum
from dataclasses import dataclass
from decision_graph import NodeType
from state import GemColor

ACTION_SPACE_SIZE = 512

class ActionType(Enum):
    TAKE_GEMS = 0
    BUY_CARD = 1
    RESERVE_CARD = 2
    FORCED_DISCARD = 4 #CHATGPT does not recommend this in here. Might move it to rules?
    PICK_NOBLE = 5

@dataclass
class Action:
    id: int
    type: ActionType
    params: dict
    causes_node: NodeType | None = None