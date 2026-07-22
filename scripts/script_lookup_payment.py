from splendor_v1.env.data.data import BASE_TIER_1, BASE_TIER_2, BASE_TIER_3
from splendor_v1.env.core.enums import GemColor
from splendor_v1.env.core.card import Card
from pathlib import Path


COLORS = [
    GemColor.WHITE,
    GemColor.BLUE,
    GemColor.GREEN,
    GemColor.RED,
    GemColor.BLACK,
]

ALL_CARDS = []
ALL_CARDS.extend(BASE_TIER_1)
ALL_CARDS.extend(BASE_TIER_2)
ALL_CARDS.extend(BASE_TIER_3)


def build_payment_lookup():

    cost_to_id = {}
    payment_lookup = {}

    next_cost_id = 0

    for card in ALL_CARDS:

        cost = tuple(
            card.cost[color]
            for color in COLORS
        )

        if cost not in cost_to_id:

            cost_to_id[cost] = next_cost_id

            payment_lookup[next_cost_id] = generate_payment_patterns(cost)

            next_cost_id += 1

    return payment_lookup





def generate_payment_patterns(cost: tuple[int, ...]):

    results = []

    def dfs(index, gold_used, current):

        if index == 5:
            results.append(tuple(current))
            return

        max_gold = min(
            cost[index],
            5 - gold_used
        )

        for amount in range(max_gold + 1):

            current.append(amount)

            dfs(
                index + 1,
                gold_used + amount,
                current
            )

            current.pop()

    dfs(
        index=0,
        gold_used=0,
        current=[]
    )

    return results

def write_payment_lookup(payment_lookup):

    output = Path(
        "splendor_v1/env/core/const_payment_lookup_table.py"
    )

    with output.open("w") as f:

        f.write("# AUTO-GENERATED FILE. DO NOT EDIT.\n\n")

        f.write("COST_TYPE_PAYMENTS = {\n")

        for cost_id, payments in payment_lookup.items():

            f.write(f"    {cost_id}: [\n")

            for payment in payments:
                f.write(f"        {payment},\n")

            f.write("    ],\n")

        f.write("}\n")


if __name__ == "__main__":

    payment_lookup = build_payment_lookup()

    write_payment_lookup(payment_lookup)