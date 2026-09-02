import time

from splendor_v1.env.env import SplendorEnv

from splendor_v1.env.core.enums import GemColor

NUM_RUNS = 1_000


def benchmark(name, func, state):
    start = time.perf_counter()

    for _ in range(NUM_RUNS):
        func(state)

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

    # Adjust this depending on what your reset() returns.
    env.reset()

    env.state.players[0].gems[GemColor.WHITE] = 2
    env.state.players[0].gems[GemColor.BLUE] = 1
    env.state.players[0].gems[GemColor.GREEN] = 2
    env.state.players[0].gems[GemColor.RED] = 1
    env.state.players[0].gems[GemColor.BLACK] = 1
    env.state.players[0].gems[GemColor.GOLD] = 1

    env.state.players[0].bonuses[GemColor.WHITE] = 1
    env.state.players[0].bonuses[GemColor.GREEN] = 1
    env.state.players[0].bonuses[GemColor.BLACK] = 1


    state = env.state
    # If reset returns something like (obs, info),
    # replace the line above with however you normally
    # access the actual GameState.

    slow_actions = env.slow_legal_buy_visible(state)
    fast_actions = env._legal_buy_visible(state)

    assert slow_actions == fast_actions

    print(f"Legal buy actions: {len(fast_actions)}")
    print()

    slow_time = benchmark(
        "slow_legal_buy_visible",
        env.slow_legal_buy_visible,
        state,
    )

    fast_time = benchmark(
        "_legal_buy_visible",
        env._legal_buy_visible,
        state,
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