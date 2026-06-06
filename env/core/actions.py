from dataclasses import dataclass
from decision_graph import NodeType

from enums import ActionType

ACTION_SPACE_SIZE = 128

@dataclass
class Action:
    id: int
    type: ActionType
    params: dict
    causes_node: NodeType | None = None