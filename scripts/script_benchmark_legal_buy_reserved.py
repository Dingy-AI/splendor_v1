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

    env.reset()

    state = env.state

    player = state.players[state.current_player]

    # player.gems.update({
    #     GemColor.WHITE: 2,
    #     GemColor.BLUE: 1,
    #     GemColor.GREEN: 2,
    #     GemColor.RED: 1,
    #     GemColor.BLACK: 1,
    #     GemColor.GOLD: 1,
    # })

    # player.bonuses.update({
    #     GemColor.WHITE: 1,
    #     GemColor.BLUE: 0,
    #     GemColor.GREEN: 1,
    #     GemColor.RED: 0,
    #     GemColor.BLACK: 1,
    # })

    # Give the player some reserved cards.
    # This assumes visible_cards already contains real Card objects.
    player.reserved_cards = [
        state.visible_cards[1][0],
        state.visible_cards[2][0],
        state.visible_cards[3][0],
    ]

    slow_actions = env.slow_legal_buy_reserved(state)
    fast_actions = env._legal_buy_reserved(state)

    assert slow_actions == fast_actions

    print(f"Legal reserved buy actions: {len(fast_actions)}")
    print()

    slow_time = benchmark(
        "slow_legal_buy_reserved",
        env.slow_legal_buy_reserved,
        state,
    )

    fast_time = benchmark(
        "_legal_buy_reserved",
        env._legal_buy_reserved,
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