import time
import random

from splendor_v1.env.env import SplendorEnv


def benchmark_function(name, func, iterations=100000):
    """
    Benchmark a function over N iterations.
    """

    start = time.perf_counter()

    for _ in range(iterations):
        func()

    end = time.perf_counter()

    total_time = end - start
    avg_time = total_time / iterations

    print(f"{name}")
    print(f"  Total: {total_time:.4f}s")
    print(f"  Avg:   {avg_time * 1e6:.2f} microseconds")
    print(f"  FPS:   {1 / avg_time:,.2f}")
    print()


def setup_environment():
    """
    Create a reasonably progressed game state.
    """

    env = SplendorEnv()
    env.reset()

    # Randomly progress the game
    for _ in range(20):

        actions = env._legal_actions(env.state)

        if not actions:
            break

        action = random.choice(actions)
        env.step(action)

    return env


def benchmark_environment(iterations=100000):

    env = setup_environment()

    print("Starting benchmark")
    print("------------------")

    benchmark_function(
        "clone()",
        lambda: env.clone(),
        iterations
    )

    benchmark_function(
        "legal_actions()",
        lambda: env._legal_actions(env.state),
        iterations
    )

    benchmark_function(
        "action_mask()",
        lambda: env.action_mask(env.state),
        iterations
    )

    benchmark_function(
        "observation()",
        lambda: env.observation_encoder.encoder(env.state),
        iterations
    )


if __name__ == "__main__":
    benchmark_environment()