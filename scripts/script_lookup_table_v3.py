from splendor_v1.env.data.data import BASE_TIER_1, BASE_TIER_2, BASE_TIER_3
from splendor_v1.env.core.enums import GemColor, CardType
from splendor_v1.env.core.card import Card
from pathlib import Path

COLORS = [
    GemColor.WHITE,
    GemColor.BLUE,
    GemColor.GREEN,
    GemColor.RED,
    GemColor.BLACK,
]


def get_canonical_cost(card_type):
    """
    Convert CardType enum into canonical cost tuple.

    Example:
        T1_311 -> (3,1,1,0,0)
    """

    cost_string = card_type.name.split("_")[1]

    costs = [
        int(x)
        for x in cost_string
    ]

    while len(costs) < 5:
        costs.append(0)

    return tuple(costs)

def generate_gold_payments(cost):
    """
    Generate all possible gold substitutions.

    cost:
        canonical card cost

    returns:
        list[tuple]
    """

    payments = []

    current = [0] * 5


    def recurse(index, remaining_gold):

        if index == 5:
            payments.append(tuple(current))
            return


        # Cannot substitute more than this color costs
        max_gold = min(
            cost[index],
            remaining_gold
        )


        for amount in range(max_gold + 1):

            current[index] = amount

            recurse(
                index + 1,
                remaining_gold - amount
            )


        current[index] = 0


    recurse(0,5)

    return payments

def generate_payment_table():

    table = {}

    for card_type in CardType:

        cost = get_canonical_cost(card_type)

        payments = generate_gold_payments(cost)

        table[card_type] = payments

    return table

def generate_tier_payment_lookups(payment_table):
    """
    Creates one unique payment lookup table per tier.

    Returns
    -------
    {
        1: [...],
        2: [...],
        3: [...],
    }
    """

    tier_payments = {
        1: set(),
        2: set(),
        3: set(),
    }

    for card_type, payments in payment_table.items():

        tier = int(card_type.name[1])

        tier_payments[tier].update(payments)

    # Convert to sorted lists for deterministic action IDs
    for tier in tier_payments:

        tier_payments[tier] = sorted(
            tier_payments[tier],
            key=lambda payment: (
                sum(payment),   # fewer gold first
                payment         # lexicographic tie break
            )
        )

    return tier_payments



def run_script():

    table = generate_payment_table()
    tier_payment_lookups = generate_tier_payment_lookups(table)


    print("T1:", len(tier_payment_lookups[1]))
    print("T2:", len(tier_payment_lookups[2]))
    print("T3:", len(tier_payment_lookups[3]))    
    output_file = Path("splendor_v1/env/core/cost_lookup_table_v3.py")

    with open(output_file, "w") as f:

        f.write("# Auto-generated. Do not edit.\n\n")

        for tier in (1, 2, 3):

            f.write(f"T{tier}_PAYMENT_LOOKUP = [\n")

            for payment in tier_payment_lookups[tier]:
                f.write(f"    {payment},\n")

            f.write("]\n\n")


def print_payment_table_statistics():

    payment_table = generate_payment_table()

    tier_totals = {
        1: 0,
        2: 0,
        3: 0,
    }

    print("Card Type".ljust(15), "Payments")

    for card_type, payments in payment_table.items():

        payment_count = len(payments)

        print(
            card_type.name.ljust(15),
            payment_count
        )

        if card_type.name.startswith("T1_"):
            tier_totals[1] += payment_count

        elif card_type.name.startswith("T2_"):
            tier_totals[2] += payment_count

        elif card_type.name.startswith("T3_"):
            tier_totals[3] += payment_count

    print("\nTotals")
    print(f"T1: {tier_totals[1]}")
    print(f"T2: {tier_totals[2]}")
    print(f"T3: {tier_totals[3]}")
    print(f"Overall: {sum(tier_totals.values())}")

    print("\nTotals")
    print(f"T1 total ids: {4 * tier_totals[1]}")
    print(f"T2 total ids: {4 * tier_totals[2]}")
    print(f"T3 total ids: {4 * tier_totals[3]}")
    print(f"Overall: {3 * sum(tier_totals.values())}")

    
    # Card Type       Payments
    # T1_3            4
    # T1_21           6
    # T1_4            5
    # T1_22           9
    # T1_1111         16
    # T1_311          16
    # T1_221          18
    # T1_2111         24
    # T2_5            6
    # T2_6            6
    # T2_421          26
    # T2_322          32
    # T2_53           18
    # T2_332          38
    # T3_7            6
    # T3_73           18
    # T3_633          48
    # T3_5333         111

    # Totals
    # T1: 98
    # T2: 126
    # T3: 183
    # Overall: 407

    # Totals
    # T1 total ids: 392
    # T2 total ids: 504
    # T3 total ids: 732
    # Overall: 1221    



def print_unique_payment_statistics():

    payment_table = generate_payment_table()

    unique = {
        1: set(),
        2: set(),
        3: set(),
    }

    for card_type, payments in payment_table.items():

        tier = int(card_type.name[1])

        unique[tier].update(payments)

    print(f"T1 unique payments: {len(unique[1])}")
    print(f"T2 unique payments: {len(unique[2])}")
    print(f"T3 unique payments: {len(unique[3])}")

    print(f"T1 total_action id: {4*len(unique[1])}")
    print(f"T2 total_action id: {4*len(unique[2])}")
    print(f"T3 total_action id: {4*len(unique[3])}")
    print(f"total_action id actions: {4*len(unique[1]) + 4*len(unique[2]) + 4*len(unique[3]) + (len(unique[1]) + len(unique[2]) + len(unique[3]))*3}")

    # Totals
    # T1 total ids: 392
    # T2 total ids: 504
    # T3 total ids: 732
    # Overall: 1221
    # T1 unique payments: 35
    # T2 unique payments: 42
    # T3 unique payments: 111
    # T1 total_action id: 140
    # T2 total_action id: 168
    # T3 total_action id: 444


# print_payment_table_statistics()


# print_unique_payment_statistics()

run_script()