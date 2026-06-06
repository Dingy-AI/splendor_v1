from dataclasses import dataclass

from core.enums import ActionType, NodeType

ACTION_SPACE_SIZE = 128

@dataclass(frozen=True)
class Action:
    id: int
    type: ActionType
    params: dict
    causes_node: NodeType | None = None