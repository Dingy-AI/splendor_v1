import time

from splendor_v1.env.env import SplendorEnv
from splendor_v1.env.core.enums import GemColor
from splendor_v1.env.core.card import Card


# ================================================================
# Test cards
# ================================================================

T1_SINGLE_COLOR = Card(
    id=0,
    tier=1,
    points=0,
    bonus_color=GemColor.WHITE,
    cost={
        GemColor.WHITE: 0,
        GemColor.BLUE: 3,
        GemColor.GREEN: 0,
        GemColor.RED: 0,
        GemColor.BLACK: 0,
    },
)


T1_TWO_COLOR = Card(
    id=15,
    tier=1,
    points=0,
    bonus_color=GemColor.WHITE,
    cost={
        GemColor.WHITE: 0,
        GemColor.BLUE: 2,
        GemColor.GREEN: 0,
        GemColor.RED: 0,
        GemColor.BLACK: 2,
    },
)


T2_THREE_COLOR = Card(
    id=50,
    tier=2,
    points=1,
    bonus_color=GemColor.WHITE,
    cost={
        GemColor.WHITE: 0,
        GemColor.BLUE: 0,
        GemColor.GREEN: 3,
        GemColor.RED: 2,
        GemColor.BLACK: 2,
    },
)


T2_HIGH_BLACK = Card(
    id=56,
    tier=2,
    points=2,
    bonus_color=GemColor.BLUE,
    cost={
        GemColor.WHITE: 2,
        GemColor.BLUE: 0,
        GemColor.GREEN: 0,
        GemColor.RED: 1,
        GemColor.BLACK: 4,
    },
)


T3_TWO_COLOR = Card(
    id=75,
    tier=3,
    points=5,
    bonus_color=GemColor.WHITE,
    cost={
        GemColor.WHITE: 3,
        GemColor.BLUE: 0,
        GemColor.GREEN: 0,
        GemColor.RED: 0,
        GemColor.BLACK: 7,
    },
)


T3_THREE_COLOR = Card(
    id=80,
    tier=3,
    points=4,
    bonus_color=GemColor.WHITE,
    cost={
        GemColor.WHITE: 3,
        GemColor.BLUE: 0,
        GemColor.GREEN: 0,
        GemColor.RED: 3,
        GemColor.BLACK: 6,
    },
)


ALL_TEST_CARDS = [
    T1_SINGLE_COLOR,
    T1_TWO_COLOR,
    T2_THREE_COLOR,
    T2_HIGH_BLACK,
    T3_TWO_COLOR,
    T3_THREE_COLOR,
]


GOLD_AMOUNTS = [
    0,
    1,
    2,
    3,
    4,
    5,
]


NUM_RUNS = 100_000


# ================================================================
# Benchmark helper
# ================================================================

def benchmark(
    name,
    func,
    player,
    cards,
):
    start = time.perf_counter()

    for _ in range(NUM_RUNS):
        for card in cards:
            func(
                player,
                card,
            )

    elapsed = (
        time.perf_counter()
        - start
    )

    total_calls = (
        NUM_RUNS
        * len(cards)
    )

    time_per_call = (
        elapsed
        / total_calls
    )

    calls_per_second = (
        total_calls
        / elapsed
    )

    print(
        f"{name:<36}"
        f"{time_per_call * 1_000_000:>8.3f} µs/call   "
        f"{calls_per_second:>10,.0f} calls/sec"
    )

    return elapsed


# ================================================================
# Comparison helper
# ================================================================

def print_comparison(
    label,
    old_time,
    new_time,
):
    speedup = (
        old_time
        / new_time
    )

    reduction = (
        (
            old_time
            - new_time
        )
        / old_time
        * 100
    )

    print(
        f"{label:<20}"
        f"{speedup:>7.2f}x   "
        f"{reduction:>7.1f}% reduction"
    )


# ================================================================
# Main
# ================================================================

def main():

    env = SplendorEnv()
    env.reset()

    player = env.state.players[0]

    # ------------------------------------------------------------
    # Representative midgame colored gems / bonuses
    #
    # GOLD will be changed from 0 through 5.
    # ------------------------------------------------------------

    player.bonuses = {
        GemColor.WHITE: 1,
        GemColor.BLUE: 1,
        GemColor.GREEN: 2,
        GemColor.RED: 1,
        GemColor.BLACK: 0,
    }

    base_gems = {
        GemColor.WHITE: 2,
        GemColor.BLUE: 3,
        GemColor.GREEN: 2,
        GemColor.RED: 1,
        GemColor.BLACK: 3,
    }

    # ============================================================
    # Correctness
    # ============================================================

    print("=" * 80)
    print("CORRECTNESS")
    print("=" * 80)

    for gold in GOLD_AMOUNTS:

        player.gems = {
            **base_gems,
            GemColor.GOLD: gold,
        }

        print()
        print(f"Gold = {gold}")

        for card in ALL_TEST_CARDS:

            slow_result = (
                env.slow_generate_valid_payments(
                    player,
                    card,
                )
            )

            slow_2_result = (
                env.slow_2_generate_valid_payments(
                    player,
                    card,
                )
            )

            fast_result = (
                env._generate_valid_payments(
                    player,
                    card,
                )
            )

            assert slow_result == slow_2_result
            assert slow_2_result == fast_result

            print(
                f"  Card {card.id:<3}: "
                f"{len(fast_result):>3} valid payments"
            )

    print()
    print("All implementations match for gold 0-5.")

    # ============================================================
    # Benchmarks
    # ============================================================

    total_slow_time = 0.0
    total_slow_2_time = 0.0
    total_fast_time = 0.0

    for gold in GOLD_AMOUNTS:

        player.gems = {
            **base_gems,
            GemColor.GOLD: gold,
        }

        print()
        print("=" * 80)
        print(f"GOLD = {gold}")
        print("=" * 80)

        slow_time = benchmark(
            "slow_generate_valid_payments",
            env.slow_generate_valid_payments,
            player,
            ALL_TEST_CARDS,
        )

        slow_2_time = benchmark(
            "slow_2_generate_valid_payments",
            env.slow_2_generate_valid_payments,
            player,
            ALL_TEST_CARDS,
        )

        fast_time = benchmark(
            "_generate_valid_payments",
            env._generate_valid_payments,
            player,
            ALL_TEST_CARDS,
        )

        total_slow_time += slow_time
        total_slow_2_time += slow_2_time
        total_fast_time += fast_time

        print()
        print("Comparisons:")

        print_comparison(
            "slow -> slow_2",
            slow_time,
            slow_2_time,
        )

        print_comparison(
            "slow_2 -> new",
            slow_2_time,
            fast_time,
        )

        print_comparison(
            "slow -> new",
            slow_time,
            fast_time,
        )

    # ============================================================
    # Overall comparison
    # ============================================================

    print()
    print("=" * 80)
    print("OVERALL — GOLD 0 THROUGH 5")
    print("=" * 80)

    total_calls = (
        NUM_RUNS
        * len(ALL_TEST_CARDS)
        * len(GOLD_AMOUNTS)
    )

    print()

    for name, elapsed in [
        (
            "slow_generate_valid_payments",
            total_slow_time,
        ),
        (
            "slow_2_generate_valid_payments",
            total_slow_2_time,
        ),
        (
            "_generate_valid_payments",
            total_fast_time,
        ),
    ]:

        time_per_call = (
            elapsed
            / total_calls
        )

        calls_per_second = (
            total_calls
            / elapsed
        )

        print(
            f"{name:<36}"
            f"{time_per_call * 1_000_000:>8.3f} µs/call   "
            f"{calls_per_second:>10,.0f} calls/sec"
        )

    print()
    print("Overall comparisons:")

    print_comparison(
        "slow -> slow_2",
        total_slow_time,
        total_slow_2_time,
    )

    print_comparison(
        "slow_2 -> new",
        total_slow_2_time,
        total_fast_time,
    )

    print_comparison(
        "slow -> new",
        total_slow_time,
        total_fast_time,
    )


if __name__ == "__main__":
    main()