import time

from splendor_v1.env.env import SplendorEnv


NUM_RUNS = 500_000


def benchmark(
    name,
    func,
    card,
):
    start = time.perf_counter()

    for _ in range(NUM_RUNS):
        func(card)

    elapsed = time.perf_counter() - start

    time_per_call = elapsed / NUM_RUNS
    calls_per_second = NUM_RUNS / elapsed

    print(
        f"{name}: "
        f"{time_per_call * 1_000_000:.3f} µs/call, "
        f"{calls_per_second:.0f} calls/sec"
    )

    return elapsed


def main():

    env = SplendorEnv()
    env.reset()

    state = env.state

    # Pick a real card from the game.
    card = state.visible_cards[3][0]

    # ------------------------------------------------------------
    # Correctness
    # ------------------------------------------------------------

    slow_result = env.slow_get_color_mapping(
        card
    )

    fast_result = env.get_color_mapping(
        card
    )

    assert slow_result == fast_result

    print(f"Card ID: {card.id}")
    print(f"Card cost: {card.cost}")
    print(f"Color mapping: {fast_result}")
    print()

    # ------------------------------------------------------------
    # Warm cache
    # ------------------------------------------------------------

    env.get_color_mapping(card)

    # ------------------------------------------------------------
    # Benchmark
    # ------------------------------------------------------------

    slow_time = benchmark(
        "slow_get_color_mapping",
        env.slow_get_color_mapping,
        card,
    )

    fast_time = benchmark(
        "get_color_mapping",
        env.get_color_mapping,
        card,
    )

    # ------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------

    speedup = slow_time / fast_time

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