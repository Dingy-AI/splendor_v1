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

cost_to_id = {}
id_to_cost = {}

next_id = 0

ALL_CARDS = []
ALL_CARDS.extend(BASE_TIER_1)
ALL_CARDS.extend(BASE_TIER_2)
ALL_CARDS.extend(BASE_TIER_3)



for card in ALL_CARDS:

    cost = tuple(
        card.cost[color]
        for color in COLORS
    )

    if cost not in cost_to_id:

        cost_to_id[cost] = next_id
        id_to_cost[next_id] = cost

        next_id += 1

output_file = Path("splendor_v1/env/core/cost_lookup_table.py")

with open(output_file, "w") as f:
    f.write("# AUTO-GENERATED FILE. DO NOT EDIT.\n\n")

    f.write("COST_TO_ID = ")
    f.write(repr(cost_to_id))
    f.write("\n\n")

    f.write("ID_TO_COST = ")
    f.write(repr(id_to_cost))
    f.write("\n")

