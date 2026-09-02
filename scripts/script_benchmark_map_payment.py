import time

from splendor_v1.env.env import SplendorEnv
from splendor_v1.env.core.enums import GemColor


NUM_RUNS = 1_000_000


def benchmark(name, func, *args):
    start = time.perf_counter()

    for _ in range(NUM_RUNS):
        func(*args)

    elapsed = time.perf_counter() - start

    time_per_call = elapsed / NUM_RUNS
    calls_per_second = NUM_RUNS / elapsed

    print(
        f"{name}: "
        f"{time_per_call * 1_000_000:.2f} µs/call, "
        f"{calls_per_second:.0f} calls/sec"
    )

    return elapsed


def main():
    env = SplendorEnv()

    canonical_payment = (
        2,
        0,
        1,
        3,
        0,
    )

    color_mapping = (
        GemColor.RED,
        GemColor.WHITE,
        GemColor.BLACK,
        GemColor.BLUE,
        GemColor.GREEN,
    )

    # Sanity check: both implementations must return the same result.
    slow_result = env.slow_map_payment_to_card(
        canonical_payment,
        color_mapping,
    )

    fast_result = env.map_payment_to_card(
        canonical_payment,
        color_mapping,
    )

    assert slow_result == fast_result

    print(f"Result: {fast_result}")
    print()

    slow_time = benchmark(
        "slow_map_payment_to_card",
        env.slow_map_payment_to_card,
        canonical_payment,
        color_mapping,
    )

    fast_time = benchmark(
        "map_payment_to_card",
        env.map_payment_to_card,
        canonical_payment,
        color_mapping,
    )

    speedup = slow_time / fast_time
    time_reduction = (
        (slow_time - fast_time)
        / slow_time
        * 100
    )

    print()
    print(f"Speedup: {speedup:.2f}x")
    print(f"Time reduction: {time_reduction:.1f}%")


if __name__ == "__main__":
    main()