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


NUM_RUNS = 100_000




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

    elapsed = time.perf_counter() - start

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
        f"{name}: "
        f"{time_per_call * 1_000_000:.3f} µs/call, "
        f"{calls_per_second:.0f} calls/sec"
    )

    return elapsed


def main():

    env = SplendorEnv()
    env.reset()

    player = env.state.players[0]

    # ------------------------------------------------------------
    # Representative midgame state
    # ------------------------------------------------------------

    player.gems = {
        GemColor.WHITE: 2,
        GemColor.BLUE: 3,
        GemColor.GREEN: 2,
        GemColor.RED: 1,
        GemColor.BLACK: 3,
        GemColor.GOLD: 3,
    }

    player.bonuses = {
        GemColor.WHITE: 1,
        GemColor.BLUE: 1,
        GemColor.GREEN: 2,
        GemColor.RED: 1,
        GemColor.BLACK: 0,
    }

    # ------------------------------------------------------------
    # Correctness
    # ------------------------------------------------------------

    print("Checking correctness...")

    for card in ALL_TEST_CARDS:

        slow_result = (
            env.slow_generate_valid_payments(
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

        assert slow_result == fast_result

        print(
            f"Card {card.id}: "
            f"{len(fast_result)} valid payments"
        )

    print()
    print("All results match.")
    print()

    # ------------------------------------------------------------
    # Benchmark
    # ------------------------------------------------------------

    slow_time = benchmark(
        "slow_generate_valid_payments",
        env.slow_generate_valid_payments,
        player,
        ALL_TEST_CARDS,
    )

    fast_time = benchmark(
        "_generate_valid_payments",
        env._generate_valid_payments,
        player,
        ALL_TEST_CARDS,
    )

    # ------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------

    speedup = (
        slow_time
        / fast_time
    )

    reduction = (
        (slow_time - fast_time)
        / slow_time
        * 100
    )

    print()

    print(
        f"Speedup: {speedup:.2f}x"
    )

    print(
        f"Time reduction: {reduction:.1f}%"
    )


if __name__ == "__main__":
    main()