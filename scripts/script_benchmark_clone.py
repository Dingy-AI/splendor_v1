import time

from splendor_v1.env.env import SplendorEnv


NUM_RUNS = 10_000


def benchmark_clone(label, clone_fn):
    start = time.perf_counter()

    for _ in range(NUM_RUNS):
        clone_fn()

    elapsed = time.perf_counter() - start
    avg_seconds = elapsed / NUM_RUNS

    print(
        f"{label}: "
        f"{avg_seconds * 1_000_000:.2f} µs/clone, "
        f"{1 / avg_seconds:.0f} clones/sec"
    )

    return avg_seconds


env = SplendorEnv()
env.reset()
# Get an actual GameState here using your env's API.
state = env.state


slow_time = benchmark_clone(
    "slow_clone",
    state.slow_clone,
)

fast_time = benchmark_clone(
    "clone",
    state.clone,
)


print(
    f"\nSpeedup: "
    f"{slow_time / fast_time:.2f}x"
)

print(
    f"Time reduction: "
    f"{(1 - fast_time / slow_time) * 100:.1f}%"
)