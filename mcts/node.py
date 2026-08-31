from dataclasses import dataclass, field

from splendor_v1.env.state.base import GameState
from splendor_v1.env.core.actions import Action


@dataclass
class Node:

    state: GameState

    parent: "Node | None" = None

    action: Action | None = None

    children: list["Node"] = field(default_factory=list)

    visits: int = 0

    value: float = 0.0

    untried_actions: list[Action] = field(default_factory=list)

    prior: float = 0.0