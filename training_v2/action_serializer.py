def serialize_action(action):
    """
    Convert an Action into stable plain-data form.

    We preserve every semantic field currently present
    on the Action dataclass.

    The numeric canonical action ID is stored separately
    by ReplayGenerator.
    """

    if action is None:
        return None

    # --------------------------------------------------------
    # GEM COLORS
    # --------------------------------------------------------

    gem_colors = None

    if action.gem_colors is not None:

        gem_colors = [
            color.name
            for color
            in action.gem_colors
        ]

    # --------------------------------------------------------
    # GOLD PAYMENT
    # --------------------------------------------------------

    gold_payment = None

    if action.gold_payment is not None:

        gold_payment = [
            int(value)
            for value
            in action.gold_payment
        ]

    # --------------------------------------------------------
    # ACTION
    # --------------------------------------------------------

    return {

        "action_type":
            action.action_type.name,

        "tier":
            action.tier,

        "slot":
            action.slot,

        "reserved_index":
            action.reserved_index,

        "gem_colors":
            gem_colors,

        "noble_index":
            action.noble_index,

        "payment_id":
            action.payment_id,

        "gold_payment":
            gold_payment,
    }


def serialize_actions(actions):

    return [
        serialize_action(action)
        for action
        in actions
    ]