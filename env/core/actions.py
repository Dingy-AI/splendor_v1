from dataclasses import dataclass

from env.core.enums import ActionType

from typing import Optional, Tuple

@dataclass(frozen=True)
class Action:
    action_type: ActionType

    # board-related
    tier: Optional[int] = None
    slot: Optional[int] = None

    # reserved cards
    reserved_index: Optional[int] = None

    # gem actions
    gem_colors: Optional[Tuple[int, ...]] = None  # or Color enum

    # noble actions
    noble_index: Optional[int] = None


# EXAMPLES
# Action(
#     action_type=ActionType.BUY_VISIBLE,
#     tier=2,
#     slot=1,
# )
# )
# Action(
#     action_type=ActionType.TAKE_GEMS,
#     gem_colors=(Color.RED, Color.BLUE, Color.BLACK),
# )

# Action(
#     action_type=ActionType.DISCARD_GEM,
#     discard_color=Color.RED,
# )

# Action(
#     action_type=ActionType.RESERVE_VISIBLE,
#     tier=tier,
#     slot=slot,
# )

# Action(
#     action_type=ActionType.TAKE_NOBLE,
#     noble_index=4,
# )