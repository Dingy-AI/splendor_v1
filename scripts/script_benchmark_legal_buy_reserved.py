import time

from splendor_v1.env.env import SplendorEnv
from splendor_v1.env.core.enums import GemColor


NUM_RUNS = 1_000


def benchmark(
    name,
    func,
    state,
):
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

    # Representative player state
    player.gems.update({
        GemColor.WHITE: 2,
        GemColor.BLUE: 2,
        GemColor.GREEN: 2,
        GemColor.RED: 2,
        GemColor.BLACK: 2,
        GemColor.GOLD: 2,
    })

    player.bonuses.update({
        GemColor.WHITE: 1,
        GemColor.BLUE: 1,
        GemColor.GREEN: 1,
        GemColor.RED: 1,
        GemColor.BLACK: 1,
    })

    # Give the player one reserved card from each tier.
    player.reserved_cards = [
        state.visible_cards[1][0],
        state.visible_cards[2][0],
        state.visible_cards[3][0],
    ]

    # ------------------------------------------------------------
    # Correctness
    # ------------------------------------------------------------

    original_actions = env.slow_legal_buy_reserved(
        state
    )

    version_2_actions = env.slow_2_legal_buy_reserved(
        state
    )

    latest_actions = env._legal_buy_reserved(
        state
    )

    original_ids = sorted(
        env.action_to_id(action)
        for action in original_actions
    )

    version_2_ids = sorted(
        env.action_to_id(action)
        for action in version_2_actions
    )

    latest_ids = sorted(
        env.action_to_id(action)
        for action in latest_actions
    )

    assert original_ids == version_2_ids
    assert original_ids == latest_ids

    print(
        f"Legal reserved buy actions: "
        f"{len(latest_actions)}"
    )

    print()

    # ------------------------------------------------------------
    # Benchmark
    # ------------------------------------------------------------

    original_time = benchmark(
        "original",
        env.slow_legal_buy_reserved,
        state,
    )

    version_2_time = benchmark(
        "version_2",
        env.slow_2_legal_buy_reserved,
        state,
    )

    latest_time = benchmark(
        "latest",
        env._legal_buy_reserved,
        state,
    )

    # ------------------------------------------------------------
    # Comparisons
    # ------------------------------------------------------------

    print()

    v2_speedup = (
        original_time
        / version_2_time
    )

    v2_reduction = (
        (original_time - version_2_time)
        / original_time
        * 100
    )

    latest_vs_original_speedup = (
        original_time
        / latest_time
    )

    latest_vs_original_reduction = (
        (original_time - latest_time)
        / original_time
        * 100
    )

    latest_vs_v2_speedup = (
        version_2_time
        / latest_time
    )

    latest_vs_v2_reduction = (
        (version_2_time - latest_time)
        / version_2_time
        * 100
    )

    print(
        f"Original -> Version 2 speedup: "
        f"{v2_speedup:.2f}x"
    )

    print(
        f"Original -> Version 2 reduction: "
        f"{v2_reduction:.1f}%"
    )

    print()

    print(
        f"Original -> Latest speedup: "
        f"{latest_vs_original_speedup:.2f}x"
    )

    print(
        f"Original -> Latest reduction: "
        f"{latest_vs_original_reduction:.1f}%"
    )

    print()

    print(
        f"Version 2 -> Latest speedup: "
        f"{latest_vs_v2_speedup:.2f}x"
    )

    print(
        f"Version 2 -> Latest reduction: "
        f"{latest_vs_v2_reduction:.1f}%"
    )


if __name__ == "__main__":
    main()