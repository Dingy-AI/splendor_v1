import time
import random
import numpy as np
import torch

from splendor_v1.env.env import SplendorEnv
from splendor_v1.mcts.mcts import MCTS
from splendor_v1.mcts.node import Node
from splendor_v1.network.model import SplendorNetwork


def collect_realistic_states(
    env,
    num_states=25,
    max_steps=150,
):

    env.reset()

    states = []

    for _ in range(max_steps):

        state = env.state

        if env._check_terminated(state):
            break

        legal_actions = env._legal_actions(
            state
        )

        if not legal_actions:
            break

        # Don't save every adjacent state.
        # Spread samples across the game.
        if len(states) < num_states:

            if random.random() < 0.35:
                states.append(
                    state.clone()
                )

        action = random.choice(
            legal_actions
        )

        env.step(
            action
        )

        if len(states) >= num_states:
            break

    return states


def compare_expansion_results(
    mcts,
    env,
    state,
):

    slow_root = Node(
        state=state.clone(),
    )

    fast_root = Node(
        state=state.clone(),
    )

    mcts.slow_expand_all_with_priors(
        env,
        slow_root,
    )

    mcts.expand_all_with_priors(
        env,
        fast_root,
    )

    assert slow_root.expanded
    assert fast_root.expanded

    assert len(slow_root.children) == len(
        fast_root.children
    )

    for slow_child, fast_child in zip(
        slow_root.children,
        fast_root.children,
    ):

        assert (
            slow_child.action
            ==
            fast_child.action
        )

        assert np.isclose(
            slow_child.prior,
            fast_child.prior,
            atol=1e-6,
        )


def benchmark_expand_realistic(
    mcts,
    env,
    states,
    repeats=20,
):

    # -------------------------
    # Correctness
    # -------------------------

    for state in states:

        compare_expansion_results(
            mcts,
            env,
            state,
        )

    print(
        f"Correctness passed on "
        f"{len(states)} states."
    )

    # -------------------------
    # Warmup
    # -------------------------

    for state in states[:5]:

        slow_root = Node(
            state=state.clone(),
        )

        mcts.slow_expand_all_with_priors(
            env,
            slow_root,
        )

        fast_root = Node(
            state=state.clone(),
        )

        mcts.expand_all_with_priors(
            env,
            fast_root,
        )

    # -------------------------
    # Slow
    # -------------------------

    start = time.perf_counter()

    slow_calls = 0

    for _ in range(repeats):

        for state in states:

            root = Node(
                state=state.clone(),
            )

            mcts.slow_expand_all_with_priors(
                env,
                root,
            )

            slow_calls += 1

    slow_time = (
        time.perf_counter()
        - start
    )

    # -------------------------
    # Fast
    # -------------------------

    start = time.perf_counter()

    fast_calls = 0

    for _ in range(repeats):

        for state in states:

            root = Node(
                state=state.clone(),
            )

            mcts.expand_all_with_priors(
                env,
                root,
            )

            fast_calls += 1

    fast_time = (
        time.perf_counter()
        - start
    )

    # -------------------------
    # Results
    # -------------------------

    slow_us = (
        slow_time
        / slow_calls
        * 1_000_000
    )

    fast_us = (
        fast_time
        / fast_calls
        * 1_000_000
    )

    speedup = (
        slow_time
        / fast_time
    )

    reduction = (
        1
        - fast_time / slow_time
    ) * 100

    print()

    print(
        f"States tested: "
        f"{len(states)}"
    )

    print(
        f"Calls per implementation: "
        f"{slow_calls}"
    )

    print()

    print(
        f"slow_expand_all_with_priors: "
        f"{slow_us:.2f} µs/call"
    )

    print(
        f"expand_all_with_priors:      "
        f"{fast_us:.2f} µs/call"
    )

    print()

    print(
        f"Slow calls/sec: "
        f"{1_000_000 / slow_us:,.0f}"
    )

    print(
        f"Fast calls/sec: "
        f"{1_000_000 / fast_us:,.0f}"
    )

    print()

    print(
        f"Speedup: "
        f"{speedup:.2f}x"
    )

    print(
        f"Time reduction: "
        f"{reduction:.2f}%"
    )


if __name__ == "__main__":

    random.seed(0)
    torch.manual_seed(0)

    env = SplendorEnv()

    model = SplendorNetwork()

    mcts = MCTS(
        simulations=160,
        rollout_type="neural",
        selection_type="puct",
        model=model,
    )

    states = collect_realistic_states(
        env,
        num_states=25,
        max_steps=150,
    )

    print(
        f"Collected "
        f"{len(states)} realistic states."
    )

    benchmark_expand_realistic(
        mcts,
        env,
        states,
        repeats=20,
    )