import time

from splendor_v1.env.env import SplendorEnv


NUM_RUNS = 100_000


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
    encoder = env.observation_encoder

    card = state.visible_cards[1][0]

    # ------------------------------------------------------------
    # Correctness
    # ------------------------------------------------------------

    slow_result = encoder.slow_encode_card(
        card
    )

    cached_result = encoder._encode_card(
        card
    )

    assert slow_result == cached_result

    print(
        f"Card ID: {card.id}"
    )

    print(
        f"Encoding length: {len(cached_result)}"
    )

    print()

    # ------------------------------------------------------------
    # Warm cache
    # ------------------------------------------------------------

    # Make sure card is already cached before timing.
    encoder._encode_card(card)

    slow_time = benchmark(
        "slow_encode_card",
        encoder.slow_encode_card,
        card,
    )

    cached_time = benchmark(
        "_encode_card",
        encoder._encode_card,
        card,
    )

    # ------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------

    speedup = (
        slow_time
        / cached_time
    )

    reduction = (
        (slow_time - cached_time)
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