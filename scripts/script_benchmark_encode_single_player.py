import time

from splendor_v1.env.env import SplendorEnv
from splendor_v1.env.core.enums import GemColor


NUM_RUNS = 100_000


def benchmark(
    name,
    func,
    player,
):
    start = time.perf_counter()

    for _ in range(NUM_RUNS):
        func(player)

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
    player = state.players[state.current_player]

    # ------------------------------------------------------------
    # Create a representative mid-game player
    # ------------------------------------------------------------

    player.gems.update({
        GemColor.WHITE: 2,
        GemColor.BLUE: 3,
        GemColor.GREEN: 1,
        GemColor.RED: 2,
        GemColor.BLACK: 1,
        GemColor.GOLD: 2,
    })

    player.bonuses.update({
        GemColor.WHITE: 2,
        GemColor.BLUE: 1,
        GemColor.GREEN: 3,
        GemColor.RED: 1,
        GemColor.BLACK: 2,
    })

    player.points = 8

    # Use three actual cards so we exercise all 33
    # reserved-card features.
    player.reserved_cards = [
        state.visible_cards[1][0],
        state.visible_cards[2][0],
        state.visible_cards[3][0],
    ]

    # ------------------------------------------------------------
    # Warm card cache
    # ------------------------------------------------------------

    for card in player.reserved_cards:
        encoder._encode_card(card)

    # ------------------------------------------------------------
    # Correctness
    # ------------------------------------------------------------

    slow_result = encoder.slow_encode_single_player(
        player
    )

    fast_result = encoder._encode_single_player(
        player
    )

    assert slow_result == fast_result

    assert len(fast_result) == 45

    print("Representative player:")
    print(f"  Gems: {player.gems}")
    print(f"  Bonuses: {player.bonuses}")
    print(f"  Points: {player.points}")
    print(
        f"  Reserved cards: "
        f"{[card.id for card in player.reserved_cards]}"
    )
    print(f"  Encoding length: {len(fast_result)}")

    print()

    # ------------------------------------------------------------
    # Benchmark
    # ------------------------------------------------------------

    slow_time = benchmark(
        "slow_encode_single_player",
        encoder.slow_encode_single_player,
        player,
    )

    fast_time = benchmark(
        "_encode_single_player",
        encoder._encode_single_player,
        player,
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