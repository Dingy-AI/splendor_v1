"""
Three-way smoke test for Splendor neural evaluation and MCTS.

Compares:

    1. Original neural_evaluate(...)
    2. DirectNeuralEvaluator
    3. BatchedNeuralEvaluator

The batched evaluator is tested with simultaneous requests so this
script verifies BOTH:

    - numerical equivalence
    - actual batching (observed batch size > 1)

It then compares three MCTS paths:

    A. original MCTS V5 pruning
    B. evaluator-injected MCTS + DirectNeuralEvaluator
    C. evaluator-injected MCTS + BatchedNeuralEvaluator

For the MCTS comparison root Dirichlet noise is disabled by default
to keep the searches deterministic.

This script does not train the model and does not modify replay data.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import math
import random
from pathlib import Path
import threading

import numpy as np
import torch

from splendor_v1.env.env import SplendorEnv

from splendor_v1.network.model_4_legal_scorer import (
    SplendorNetwork,
)

from splendor_v1.mcts.neural_evaluator_v4 import (
    neural_evaluate,
)

from splendor_v1.mcts.mcts_v5_pruning import (
    MCTS as OriginalMCTS,
)

from splendor_v1.mcts_batched.direct_neural_evaluator import (
    DirectNeuralEvaluator,
)

from splendor_v1.mcts_batched.batched_neural_evaluator import (
    BatchedNeuralEvaluator,
)

from splendor_v1.mcts_batched.mcts_v5_direct import (
    MCTS as EvaluatorMCTS,
)


# ============================================================
# DEFAULTS
# ============================================================

DEFAULT_CHECKPOINT = (
    "splendor_v1/training_v5/data/"
    "model_1900_games.pt"
)


# ============================================================
# MODEL LOADING
# ============================================================

def extract_model_state_dict(
    checkpoint,
):
    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        return checkpoint[
            "model_state_dict"
        ]

    if (
        isinstance(checkpoint, dict)
        and checkpoint
        and all(
            torch.is_tensor(value)
            for value in checkpoint.values()
        )
    ):
        return checkpoint

    raise RuntimeError(
        "Checkpoint must be a raw model state_dict "
        "or contain 'model_state_dict'."
    )


def load_model(
    checkpoint_path,
    device,
):
    checkpoint_path = Path(
        checkpoint_path
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "Checkpoint does not exist: "
            f"{checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model = SplendorNetwork()

    model.load_state_dict(
        extract_model_state_dict(
            checkpoint
        ),
        strict=True,
    )

    model.to(
        device
    )

    model.eval()

    return model


# ============================================================
# REACHABLE TEST STATES
# ============================================================

def sample_reachable_states(
    num_states,
    seed,
    stride,
):
    """
    Create deterministic reachable Splendor states by playing
    random legal moves from seeded games.
    """

    states = []

    chooser = random.Random(
        seed
    )

    game_index = 0

    while len(states) < num_states:

        env = SplendorEnv()

        env.reset(
            seed=(
                int(seed)
                + game_index
            )
        )

        step_index = 0

        while (
            len(states) < num_states
            and step_index < 300
        ):
            state = env.state

            if env._check_terminated(
                state
            ):
                break

            legal_actions = (
                env._legal_actions(
                    state
                )
            )

            if not legal_actions:
                break

            if (
                step_index == 0
                or step_index % stride == 0
            ):
                states.append(
                    state.clone()
                )

                if len(states) >= num_states:
                    break

            action = legal_actions[
                chooser.randrange(
                    len(legal_actions)
                )
            ]

            env.step(
                action
            )

            step_index += 1

        game_index += 1

        if game_index > 100:
            raise RuntimeError(
                "Unable to sample enough "
                "reachable states."
            )

    return states


# ============================================================
# GENERIC COMPARISON HELPERS
# ============================================================

def action_key(
    env,
    action,
):
    if action is None:
        return None

    try:
        return int(
            env.action_to_id(
                action
            )
        )

    except Exception:
        return repr(
            action
        )


def max_tensor_difference(
    a,
    b,
):
    a_cpu = (
        a.detach()
        .float()
        .cpu()
    )

    b_cpu = (
        b.detach()
        .float()
        .cpu()
    )

    if a_cpu.shape != b_cpu.shape:
        raise AssertionError(
            "Tensor shape mismatch: "
            f"{tuple(a_cpu.shape)} != "
            f"{tuple(b_cpu.shape)}"
        )

    if a_cpu.numel() == 0:
        return 0.0

    return float(
        torch.max(
            torch.abs(
                a_cpu
                - b_cpu
            )
        ).item()
    )


def assert_tensor_close(
    label,
    a,
    b,
    atol,
    rtol,
):
    diff = max_tensor_difference(
        a,
        b,
    )

    if not torch.allclose(
        a.detach().float().cpu(),
        b.detach().float().cpu(),
        atol=atol,
        rtol=rtol,
    ):
        raise AssertionError(
            f"{label} mismatch. "
            f"max_abs_diff={diff:.12g}"
        )

    return diff


def assert_float_close(
    label,
    a,
    b,
    atol,
    rtol,
):
    if not math.isclose(
        float(a),
        float(b),
        abs_tol=atol,
        rel_tol=rtol,
    ):
        raise AssertionError(
            f"{label} mismatch: "
            f"{a!r} != {b!r}"
        )

    return abs(
        float(a)
        - float(b)
    )


# ============================================================
# TEST 1: ORIGINAL VS DIRECT VS BATCHED EVALUATION
# ============================================================

def evaluate_original(
    model,
    state,
):
    env = SplendorEnv()

    legal_actions = (
        env._legal_actions(
            state
        )
    )

    if not legal_actions:
        raise AssertionError(
            "Sampled state has no legal actions."
        )

    probs, value = neural_evaluate(
        env=env,
        model=model,
        state=state,
        legal_actions=legal_actions,
    )

    return {
        "probs":
            probs,

        "value":
            value,

        "num_legal":
            len(
                legal_actions
            ),
    }


def evaluate_direct(
    evaluator,
    state,
):
    env = SplendorEnv()

    legal_actions = (
        env._legal_actions(
            state
        )
    )

    probs, value = (
        evaluator.evaluate(
            env=env,
            state=state,
            legal_actions=legal_actions,
        )
    )

    return {
        "probs":
            probs,

        "value":
            value,

        "num_legal":
            len(
                legal_actions
            ),
    }


def evaluate_batched_worker(
    evaluator,
    state,
    start_barrier,
):
    """
    Synchronize caller threads so their requests reach the queue
    close together and form a real batch.
    """

    env = SplendorEnv()

    legal_actions = (
        env._legal_actions(
            state
        )
    )

    if not legal_actions:
        raise AssertionError(
            "Sampled state has no legal actions."
        )

    start_barrier.wait()

    probs, value = (
        evaluator.evaluate(
            env=env,
            state=state,
            legal_actions=legal_actions,
        )
    )

    return {
        "probs":
            probs,

        "value":
            value,

        "num_legal":
            len(
                legal_actions
            ),
    }


def test_three_way_evaluator_equivalence(
    model,
    states,
    max_batch_size,
    batch_wait_ms,
    atol,
    rtol,
    require_full_batch,
):
    print()
    print("=" * 78)
    print(
        "TEST 1: ORIGINAL vs DIRECT vs BATCHED "
        "NEURAL EVALUATION"
    )
    print("=" * 78)

    direct = DirectNeuralEvaluator(
        model=model,
    )

    original_results = [
        evaluate_original(
            model=model,
            state=state,
        )
        for state in states
    ]

    direct_results = [
        evaluate_direct(
            evaluator=direct,
            state=state,
        )
        for state in states
    ]

    effective_max_batch = min(
        max_batch_size,
        len(states),
    )

    with BatchedNeuralEvaluator(
        model=model,
        max_batch_size=effective_max_batch,
        batch_wait_ms=batch_wait_ms,
        request_timeout_s=30.0,
    ) as batched:

        # One barrier party per worker plus the main thread.
        barrier = threading.Barrier(
            len(states) + 1
        )

        with ThreadPoolExecutor(
            max_workers=len(states)
        ) as executor:

            futures = [
                executor.submit(
                    evaluate_batched_worker,
                    batched,
                    state,
                    barrier,
                )
                for state in states
            ]

            # Release all evaluator calls together.
            barrier.wait()

            batched_results = [
                future.result(
                    timeout=30.0
                )
                for future in futures
            ]

        batch_stats = (
            batched.stats_snapshot()
        )

    max_original_direct_policy = 0.0
    max_original_batched_policy = 0.0
    max_direct_batched_policy = 0.0

    max_original_direct_value = 0.0
    max_original_batched_value = 0.0
    max_direct_batched_value = 0.0

    for index, (
        original,
        direct_result,
        batched_result,
    ) in enumerate(
        zip(
            original_results,
            direct_results,
            batched_results,
        ),
        start=1,
    ):
        if not (
            original["num_legal"]
            == direct_result["num_legal"]
            == batched_result["num_legal"]
        ):
            raise AssertionError(
                f"State {index} legal-action "
                "count mismatch."
            )

        od_policy = assert_tensor_close(
            (
                f"State {index} "
                "original/direct policy"
            ),
            original["probs"],
            direct_result["probs"],
            atol=atol,
            rtol=rtol,
        )

        ob_policy = assert_tensor_close(
            (
                f"State {index} "
                "original/batched policy"
            ),
            original["probs"],
            batched_result["probs"],
            atol=atol,
            rtol=rtol,
        )

        db_policy = assert_tensor_close(
            (
                f"State {index} "
                "direct/batched policy"
            ),
            direct_result["probs"],
            batched_result["probs"],
            atol=atol,
            rtol=rtol,
        )

        od_value = assert_float_close(
            (
                f"State {index} "
                "original/direct value"
            ),
            original["value"],
            direct_result["value"],
            atol=atol,
            rtol=rtol,
        )

        ob_value = assert_float_close(
            (
                f"State {index} "
                "original/batched value"
            ),
            original["value"],
            batched_result["value"],
            atol=atol,
            rtol=rtol,
        )

        db_value = assert_float_close(
            (
                f"State {index} "
                "direct/batched value"
            ),
            direct_result["value"],
            batched_result["value"],
            atol=atol,
            rtol=rtol,
        )

        max_original_direct_policy = max(
            max_original_direct_policy,
            od_policy,
        )

        max_original_batched_policy = max(
            max_original_batched_policy,
            ob_policy,
        )

        max_direct_batched_policy = max(
            max_direct_batched_policy,
            db_policy,
        )

        max_original_direct_value = max(
            max_original_direct_value,
            od_value,
        )

        max_original_batched_value = max(
            max_original_batched_value,
            ob_value,
        )

        max_direct_batched_value = max(
            max_direct_batched_value,
            db_value,
        )

        print(
            f"State {index:>2}: PASS | "
            f"legal="
            f"{original['num_legal']:>2} | "
            f"O-D policy={od_policy:.3e} | "
            f"O-B policy={ob_policy:.3e} | "
            f"O-B value={ob_value:.3e}"
        )

    observed_max = int(
        batch_stats[
            "max_observed_batch_size"
        ]
    )

    average_batch = float(
        batch_stats[
            "average_batch_size"
        ]
    )

    if len(states) > 1 and observed_max <= 1:
        raise AssertionError(
            "Batched evaluator produced correct "
            "outputs but never formed a batch "
            "larger than 1."
        )

    expected_full_batch = min(
        len(states),
        max_batch_size,
    )

    if (
        require_full_batch
        and observed_max
        != expected_full_batch
    ):
        raise AssertionError(
            "Full batch was required but "
            f"max observed batch was "
            f"{observed_max}; expected "
            f"{expected_full_batch}."
        )

    print()
    print("Three-way evaluator equivalence: PASS")

    print(
        "Max policy diff original/direct:",
        f"{max_original_direct_policy:.3e}",
    )

    print(
        "Max policy diff original/batched:",
        f"{max_original_batched_policy:.3e}",
    )

    print(
        "Max policy diff direct/batched:",
        f"{max_direct_batched_policy:.3e}",
    )

    print(
        "Max value diff original/direct:",
        f"{max_original_direct_value:.3e}",
    )

    print(
        "Max value diff original/batched:",
        f"{max_original_batched_value:.3e}",
    )

    print(
        "Max value diff direct/batched:",
        f"{max_direct_batched_value:.3e}",
    )

    print()
    print("Batch telemetry:")

    for key in (
        "requests_submitted",
        "requests_completed",
        "batches_completed",
        "average_batch_size",
        "max_observed_batch_size",
        "batch_fill_fraction",
        "average_collection_wait_ms",
        "average_batch_build_ms",
        "average_batch_inference_ms",
        "inference_positions_per_second",
    ):
        print(
            f"  {key}: "
            f"{batch_stats[key]}"
        )

    if (
        observed_max
        < expected_full_batch
    ):
        print()
        print(
            "NOTE: batching worked, but this "
            "run did not fill the configured "
            "maximum batch. That is not a "
            "correctness failure."
        )

    return batch_stats


# ============================================================
# MCTS COMPARISON HELPERS
# ============================================================

def make_original_mcts(
    model,
    simulations,
):
    return OriginalMCTS(
        simulations=simulations,
        rollout_type="neural",
        selection_type="puct",
        model=model,
        c_puct=3.0,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.25,
        adaptive_simulations=True,
        min_simulations=min(
            80,
            simulations,
        ),
        check_interval=20,
        target_visits_per_action=20.0,
        single_action_simulations=4,
        stability_checks=3,
    )


def make_evaluator_mcts(
    evaluator,
    simulations,
):
    return EvaluatorMCTS(
        simulations=simulations,
        rollout_type="neural",
        selection_type="puct",
        evaluator=evaluator,
        c_puct=3.0,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.25,
        adaptive_simulations=True,
        min_simulations=min(
            80,
            simulations,
        ),
        check_interval=20,
        target_visits_per_action=20.0,
        single_action_simulations=4,
        stability_checks=3,
    )


def root_stats(
    env,
    root,
):
    result = {}

    for child in root.children:

        key = action_key(
            env,
            child.action,
        )

        visits = int(
            child.visits
        )

        value = float(
            child.value
        )

        result[key] = {
            "visits":
                visits,

            "value":
                value,

            "q":
                (
                    value / visits
                    if visits > 0
                    else 0.0
                ),

            "prior":
                float(
                    getattr(
                        child,
                        "prior",
                        0.0,
                    )
                ),

            "network_prior":
                float(
                    getattr(
                        child,
                        "network_prior",
                        getattr(
                            child,
                            "prior",
                            0.0,
                        ),
                    )
                ),
        }

    return result


def compare_root_pair(
    label,
    env_a,
    root_a,
    env_b,
    root_b,
    atol,
    rtol,
):
    if int(root_a.visits) != int(
        root_b.visits
    ):
        raise AssertionError(
            f"{label}: root visits differ: "
            f"{root_a.visits} != "
            f"{root_b.visits}"
        )

    stats_a = root_stats(
        env_a,
        root_a,
    )

    stats_b = root_stats(
        env_b,
        root_b,
    )

    if set(stats_a) != set(
        stats_b
    ):
        raise AssertionError(
            f"{label}: root action sets differ."
        )

    for key in stats_a:

        a = stats_a[key]
        b = stats_b[key]

        if (
            a["visits"]
            != b["visits"]
        ):
            raise AssertionError(
                f"{label}: action {key} "
                "visit mismatch: "
                f"{a['visits']} != "
                f"{b['visits']}"
            )

        for field in (
            "value",
            "q",
            "prior",
            "network_prior",
        ):
            assert_float_close(
                (
                    f"{label}: "
                    f"action {key} "
                    f"{field}"
                ),
                a[field],
                b[field],
                atol=atol,
                rtol=rtol,
            )


def compare_search_metadata(
    label,
    a,
    b,
    atol,
    rtol,
):
    if set(a) != set(b):
        raise AssertionError(
            f"{label}: metadata keys differ."
        )

    for key in a:

        value_a = a[key]
        value_b = b[key]

        if isinstance(
            value_a,
            (float, np.floating),
        ):
            assert_float_close(
                (
                    f"{label}: "
                    f"metadata {key}"
                ),
                value_a,
                value_b,
                atol=atol,
                rtol=rtol,
            )

        elif value_a != value_b:
            raise AssertionError(
                f"{label}: metadata "
                f"{key} mismatch: "
                f"{value_a!r} != "
                f"{value_b!r}"
            )


# ============================================================
# TEST 2: ORIGINAL VS DIRECT VS BATCHED MCTS
# ============================================================

def test_three_way_mcts_equivalence(
    model,
    states,
    simulations,
    max_batch_size,
    batch_wait_ms,
    atol,
    rtol,
):
    print()
    print("=" * 78)
    print(
        "TEST 2: ORIGINAL vs DIRECT vs BATCHED "
        "MCTS SEARCH"
    )
    print("=" * 78)

    direct_evaluator = (
        DirectNeuralEvaluator(
            model=model
        )
    )

    # A single MCTS search is sequential, so this evaluator will
    # normally observe batch size 1 in this section. Test 1 is what
    # proves real batching.
    with BatchedNeuralEvaluator(
        model=model,
        max_batch_size=max_batch_size,
        batch_wait_ms=batch_wait_ms,
        request_timeout_s=30.0,
    ) as batched_evaluator:

        for index, state in enumerate(
            states,
            start=1,
        ):
            original = make_original_mcts(
                model=model,
                simulations=simulations,
            )

            direct = make_evaluator_mcts(
                evaluator=direct_evaluator,
                simulations=simulations,
            )

            batched = make_evaluator_mcts(
                evaluator=batched_evaluator,
                simulations=simulations,
            )

            env_original = SplendorEnv()
            env_direct = SplendorEnv()
            env_batched = SplendorEnv()

            action_original, root_original = (
                original.search(
                    env=env_original,
                    state=state.clone(),
                    root=None,
                    return_root=True,
                    debug=False,
                    add_root_noise=False,
                    teacher_mode=False,
                )
            )

            action_direct, root_direct = (
                direct.search(
                    env=env_direct,
                    state=state.clone(),
                    root=None,
                    return_root=True,
                    debug=False,
                    add_root_noise=False,
                    teacher_mode=False,
                )
            )

            action_batched, root_batched = (
                batched.search(
                    env=env_batched,
                    state=state.clone(),
                    root=None,
                    return_root=True,
                    debug=False,
                    add_root_noise=False,
                    teacher_mode=False,
                )
            )

            key_original = action_key(
                env_original,
                action_original,
            )

            key_direct = action_key(
                env_direct,
                action_direct,
            )

            key_batched = action_key(
                env_batched,
                action_batched,
            )

            if not (
                key_original
                == key_direct
                == key_batched
            ):
                raise AssertionError(
                    f"State {index}: chosen "
                    "actions differ: "
                    f"original={key_original}, "
                    f"direct={key_direct}, "
                    f"batched={key_batched}"
                )

            compare_root_pair(
                (
                    f"State {index} "
                    "original/direct"
                ),
                env_original,
                root_original,
                env_direct,
                root_direct,
                atol=atol,
                rtol=rtol,
            )

            compare_root_pair(
                (
                    f"State {index} "
                    "original/batched"
                ),
                env_original,
                root_original,
                env_batched,
                root_batched,
                atol=atol,
                rtol=rtol,
            )

            compare_search_metadata(
                (
                    f"State {index} "
                    "original/direct"
                ),
                original.last_search_metadata,
                direct.last_search_metadata,
                atol=atol,
                rtol=rtol,
            )

            compare_search_metadata(
                (
                    f"State {index} "
                    "original/batched"
                ),
                original.last_search_metadata,
                batched.last_search_metadata,
                atol=atol,
                rtol=rtol,
            )

            metadata = (
                original
                .last_search_metadata
            )

            print(
                f"State {index:>2}: PASS | "
                f"action={key_original} | "
                f"legal="
                f"{metadata['num_legal_actions']} | "
                f"sims="
                f"{metadata['actual_simulations']} | "
                f"stop="
                f"{metadata['stop_reason']}"
            )

        mcts_batch_stats = (
            batched_evaluator
            .stats_snapshot()
        )

    print()
    print(
        "Three-way MCTS equivalence: PASS"
    )

    print(
        "Batched evaluator during sequential "
        "MCTS had average batch size:",
        (
            mcts_batch_stats[
                "average_batch_size"
            ]
        ),
    )

    print(
        "(Batch size ~1 here is expected; "
        "real batching was verified in Test 1.)"
    )


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare original, direct, and batched "
            "Model 4 neural evaluation and MCTS."
        )
    )

    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
    )

    parser.add_argument(
        "--evaluator-states",
        type=int,
        default=8,
        help=(
            "Number of simultaneous states used "
            "for the three-way evaluator test."
        ),
    )

    parser.add_argument(
        "--mcts-states",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--simulations",
        type=int,
        default=100,
        help=(
            "MCTS hard limit for the smoke test. "
            "Use 400 for a full-budget check."
        ),
    )

    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--batch-wait-ms",
        type=float,
        default=10.0,
        help=(
            "Longer than production on purpose "
            "so the smoke test reliably forms "
            "a multi-request batch."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=32000,
    )

    parser.add_argument(
        "--state-stride",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--atol",
        type=float,
        default=1e-6,
    )

    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-5,
    )

    parser.add_argument(
        "--require-full-batch",
        action="store_true",
        help=(
            "Fail unless the largest observed "
            "batch equals min(evaluator_states, "
            "max_batch_size)."
        ),
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()

    if args.evaluator_states < 2:
        raise ValueError(
            "--evaluator-states must be >= 2 "
            "to verify real batching."
        )

    if args.mcts_states < 1:
        raise ValueError(
            "--mcts-states must be >= 1."
        )

    if args.simulations < 1:
        raise ValueError(
            "--simulations must be >= 1."
        )

    if args.max_batch_size < 2:
        raise ValueError(
            "--max-batch-size must be >= 2 "
            "to verify real batching."
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 78)
    print(
        "THREE-WAY NEURAL + MCTS "
        "SMOKE TEST"
    )
    print("=" * 78)

    print(
        "Device:",
        device,
    )

    print(
        "Checkpoint:",
        args.checkpoint,
    )

    print(
        "Evaluator states:",
        args.evaluator_states,
    )

    print(
        "MCTS states:",
        args.mcts_states,
    )

    print(
        "MCTS simulations:",
        args.simulations,
    )

    print(
        "Max batch size:",
        args.max_batch_size,
    )

    print(
        "Smoke-test batch wait:",
        f"{args.batch_wait_ms} ms",
    )

    print()
    print("Loading model...")

    model = load_model(
        checkpoint_path=(
            args.checkpoint
        ),
        device=device,
    )

    print("Model loaded.")

    num_states = max(
        args.evaluator_states,
        args.mcts_states,
    )

    print()
    print(
        f"Sampling {num_states} "
        "reachable Splendor states..."
    )

    states = sample_reachable_states(
        num_states=num_states,
        seed=args.seed,
        stride=max(
            1,
            args.state_stride,
        ),
    )

    print("States sampled.")

    batch_stats = (
        test_three_way_evaluator_equivalence(
            model=model,
            states=states[
                :args.evaluator_states
            ],
            max_batch_size=(
                args.max_batch_size
            ),
            batch_wait_ms=(
                args.batch_wait_ms
            ),
            atol=args.atol,
            rtol=args.rtol,
            require_full_batch=(
                args.require_full_batch
            ),
        )
    )

    test_three_way_mcts_equivalence(
        model=model,
        states=states[
            :args.mcts_states
        ],
        simulations=(
            args.simulations
        ),
        max_batch_size=(
            args.max_batch_size
        ),
        batch_wait_ms=(
            args.batch_wait_ms
        ),
        atol=args.atol,
        rtol=args.rtol,
    )

    print()
    print("=" * 78)
    print("ALL THREE-WAY SMOKE TESTS PASSED")
    print("=" * 78)

    print(
        "Original neural evaluation, "
        "DirectNeuralEvaluator, and "
        "BatchedNeuralEvaluator matched "
        "within tolerance."
    )

    print(
        "Original V5 MCTS, direct-evaluator "
        "MCTS, and batched-evaluator MCTS "
        "also matched on the tested states."
    )

    print(
        "Largest real concurrent batch observed:",
        batch_stats[
            "max_observed_batch_size"
        ],
    )


if __name__ == "__main__":

    try:
        main()

    except Exception as exc:

        print()
        print("=" * 78)
        print("SMOKE TEST FAILED")
        print("=" * 78)

        print(
            f"{type(exc).__name__}: "
            f"{exc}"
        )

        raise
