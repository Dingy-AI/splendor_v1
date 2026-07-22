from dataclasses import dataclass

from splendor_v1.env.core.enums import ActionType, GemColor

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
    gem_colors: Optional[Tuple[GemColor, ...]] = None  # or Color enum

    # noble actions
    noble_index: Optional[int] = None

    #payment amount actions
    payment_id: int |None = None

    
# EXAMPLES
# Action(
#     action_type=ActionType.BUY_VISIBLE,
#     tier=2,
#     slot=1,
# )
# )
# Action(
#     action_type=ActionType.TAKE_GEMS,
#     gem_colors=(GemColor.RED, GemColor.BLUE, GemColor.BLACK),
# )

# Action(
#     action_type=ActionType.DISCARD_GEM,
#     discard_color=GemColor.RED,
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