import time
from itertools import combinations

from splendor_v1.env.env import SplendorEnv
from splendor_v1.env.core.enums import GemColor
from splendor_v1.env.core.actions import Action
from splendor_v1.env.core.enums import ActionType
from splendor_v1.env.core.constants import COLOR_ORDER


NUM_RUNS = 200_000

def benchmark(
    name,
    func,
    state,
):
    start = time.perf_counter()

    for _ in range(NUM_RUNS):
        func(state)

    elapsed = (
        time.perf_counter()
        - start
    )

    time_per_call = (
        elapsed / NUM_RUNS
    )

    calls_per_second = (
        NUM_RUNS / elapsed
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

    state = env.state

    # ------------------------------------------------------------
    # Representative bank
    # ------------------------------------------------------------

    state.bank = {
        GemColor.WHITE: 4,
        GemColor.BLUE: 4,
        GemColor.GREEN: 3,
        GemColor.RED: 2,
        GemColor.BLACK: 4,
        GemColor.GOLD: 5,
    }

    # ------------------------------------------------------------
    # Correctness
    # ------------------------------------------------------------

    slow_result = env.slow_legal_take_gems(
        state
    )

    fast_result = env._legal_take_gems(
        state
    )

    assert slow_result == fast_result, (
        "Results do not match!\n"
        f"Slow: {slow_result}\n"
        f"Fast: {fast_result}"
    )

    print(
        f"Results match exactly: "
        f"{len(fast_result)} actions"
    )

    print()

    # ------------------------------------------------------------
    # Benchmark
    # ------------------------------------------------------------

    slow_time = benchmark(
        "slow_legal_take_gems",
        env.slow_legal_take_gems,
        state,
    )

    fast_time = benchmark(
        "fast_legal_take_gems",
        env._legal_take_gems,
        state,
    )

    # ------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------

    speedup = (
        slow_time
        / fast_time
    )

    reduction = (
        (
            slow_time
            - fast_time
        )
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