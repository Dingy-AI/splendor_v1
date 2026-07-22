def generate_all_actions():

    actions = []

    actions.extend(
        generate_take_gem_actions()
    )

    actions.extend(
        generate_reserve_actions()
    )

    actions.extend(
        generate_buy_actions()
    )

    actions.extend(
        generate_noble_actions()
    )

    return actions

# need to generate all the action ids that are possible for each tier of gem assuming 5 gold
# we want each buy action id to correspond to a tier/slot and a gold payment
# we want each buy reserve action id to also coorespond to a gold payment 