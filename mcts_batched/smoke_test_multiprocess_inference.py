"""
End-to-end smoke test for centralized multiprocess neural inference.

Compares:

    DirectNeuralEvaluator
        vs
    MultiprocessNeuralEvaluator -> GPUInferenceServerProcess

on identical reachable Splendor states.

The test intentionally avoids keeping two CUDA copies of Model 4
alive at once:

    Phase A
        load Model 4 in parent
        compute direct reference outputs
        free parent model / CUDA cache

    Phase B
        spawn GPU inference server
        load Model 4 inside child
        evaluate the same states through multiprocessing IPC
        compare policy/value outputs

This test does not train, mutate replay data, or run full MCTS.

Windows
-------
Run as a module from the repository root:

    python -m splendor_v1.mcts_batched.smoke_test_multiprocess_inference

The __main__ guard and multiprocessing.freeze_support() are required
for Windows spawn semantics.
"""

from __future__ import annotations

import argparse
import gc
import math
import multiprocessing as mp
from pathlib import Path
import random
import time

import numpy as np
import torch

from splendor_v1.env.env import (
    SplendorEnv,
)

from splendor_v1.network.model_4_legal_scorer import (
    SplendorNetwork,
)

from splendor_v1.mcts_batched.direct_neural_evaluator import (
    DirectNeuralEvaluator,
)

from splendor_v1.mcts_batched.gpu_inference_server import (
    GPUInferenceServerConfig,
    GPUInferenceServerProcess,
    create_inference_ipc,
)

from splendor_v1.mcts_batched.multiprocess_neural_evaluator import (
    MultiprocessNeuralEvaluator,
)


# ============================================================
# DEFAULTS
# ============================================================


DEFAULT_CHECKPOINT = (
    "splendor_v1/training_v5/data/"
    "model_1900_games.pt"
)

DEFAULT_STATES = 8
DEFAULT_SEED = 42000
DEFAULT_STRIDE = 5

DEFAULT_MAX_BATCH_SIZE = 64
DEFAULT_BATCH_WAIT_MS = 0.5

DEFAULT_ATOL = 1e-6
DEFAULT_RTOL = 1e-5


# ============================================================
# MODEL LOADING FOR DIRECT REFERENCE
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
            for value
            in checkpoint.values()
        )
    ):
        return checkpoint

    raise RuntimeError(
        "Checkpoint must be a raw model state_dict "
        "or contain 'model_state_dict'."
    )


def normalize_device(
    device_string,
):
    device = torch.device(
        device_string
    )

    if (
        device.type == "cuda"
        and device.index is None
    ):
        device = torch.device(
            "cuda:0"
        )

    return device


def load_direct_model(
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
        map_location="cpu",
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
    Deterministically sample reachable states by playing random
    legal moves from seeded games.
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
# REFERENCE OUTPUTS
# ============================================================


def compute_direct_references(
    *,
    checkpoint_path,
    states,
    device,
):
    """
    Compute reference outputs, then return only CPU data.
    """

    print()
    print("=" * 78)
    print(
        "PHASE A: DIRECT NEURAL EVALUATOR "
        "REFERENCE"
    )
    print("=" * 78)

    print(
        "Loading direct Model 4 on",
        device,
    )

    model = load_direct_model(
        checkpoint_path=(
            checkpoint_path
        ),
        device=device,
    )

    evaluator = DirectNeuralEvaluator(
        model=model
    )

    references = []

    start = time.perf_counter()

    for index, state in enumerate(
        states
    ):
        env = SplendorEnv()

        legal_actions = (
            env._legal_actions(
                state
            )
        )

        if not legal_actions:
            raise AssertionError(
                f"State {index} has no legal actions."
            )

        probs, value = (
            evaluator.evaluate(
                env=env,
                state=state,
                legal_actions=(
                    legal_actions
                ),
            )
        )

        probs_cpu = (
            probs.detach()
            .float()
            .cpu()
            .clone()
        )

        references.append(
            {
                "probs":
                    probs_cpu,

                "value":
                    float(
                        value
                    ),

                "num_legal":
                    len(
                        legal_actions
                    ),
            }
        )

        print(
            f"  state {index:02d}: "
            f"legal={len(legal_actions):2d} "
            f"value={float(value): .8f}"
        )

    elapsed = (
        time.perf_counter()
        - start
    )

    print(
        f"Direct reference time: "
        f"{elapsed:.3f}s"
    )

    # Ensure all GPU work from the direct path is done before we
    # destroy the parent copy.
    if device.type == "cuda":
        torch.cuda.synchronize(
            device
        )

    del evaluator
    del model

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(
        "Parent Model 4 released before "
        "starting GPU server."
    )

    return references


# ============================================================
# COMPARISON
# ============================================================


def max_abs_tensor_difference(
    a,
    b,
):
    a = (
        a.detach()
        .float()
        .cpu()
    )

    b = (
        b.detach()
        .float()
        .cpu()
    )

    if a.shape != b.shape:
        raise AssertionError(
            "Policy shape mismatch: "
            f"{tuple(a.shape)} != "
            f"{tuple(b.shape)}"
        )

    if a.numel() == 0:
        return 0.0

    return float(
        torch.max(
            torch.abs(
                a - b
            )
        ).item()
    )


def compare_policy(
    *,
    index,
    expected,
    actual,
    atol,
    rtol,
):
    diff = (
        max_abs_tensor_difference(
            expected,
            actual,
        )
    )

    if not torch.allclose(
        expected.detach().float().cpu(),
        actual.detach().float().cpu(),
        atol=atol,
        rtol=rtol,
    ):
        raise AssertionError(
            f"State {index} policy mismatch. "
            f"max_abs_diff={diff:.12g}"
        )

    return diff


def compare_value(
    *,
    index,
    expected,
    actual,
    atol,
    rtol,
):
    expected = float(
        expected
    )

    actual = float(
        actual
    )

    diff = abs(
        expected - actual
    )

    if not math.isclose(
        expected,
        actual,
        abs_tol=atol,
        rel_tol=rtol,
    ):
        raise AssertionError(
            f"State {index} value mismatch: "
            f"direct={expected:.12g}, "
            f"multiprocess={actual:.12g}, "
            f"abs_diff={diff:.12g}"
        )

    return diff


# ============================================================
# MULTIPROCESS TEST
# ============================================================


def run_multiprocess_comparison(
    *,
    checkpoint_path,
    states,
    references,
    device_string,
    max_batch_size,
    batch_wait_ms,
    request_timeout_s,
    atol,
    rtol,
):
    print()
    print("=" * 78)
    print(
        "PHASE B: MULTIPROCESS CLIENT -> "
        "GPU INFERENCE SERVER"
    )
    print("=" * 78)

    # Explicit spawn context keeps the test representative of the
    # intended Windows production architecture.
    context = mp.get_context(
        "spawn"
    )

    ipc = create_inference_ipc(
        num_workers=1,
        mp_context=context,
    )

    server = GPUInferenceServerProcess(
        config=GPUInferenceServerConfig(
            checkpoint_path=(
                checkpoint_path
            ),
            max_batch_size=(
                max_batch_size
            ),
            batch_wait_ms=(
                batch_wait_ms
            ),
            device=(
                device_string
            ),
            stats_report_interval_batches=0,
        ),
        ipc=ipc,
        mp_context=context,
    )

    server.start()

    try:
        ready = (
            server.wait_until_ready(
                timeout_s=(
                    request_timeout_s
                ),
            )
        )

        print(
            "GPU server READY "
            f"(pid={ready.pid})."
        )

        evaluator = (
            MultiprocessNeuralEvaluator(
                worker_id=0,
                request_queue=(
                    ipc.request_queue
                ),
                response_queue=(
                    ipc.response_queues[0]
                ),
                request_timeout_s=(
                    request_timeout_s
                ),
            )
        )

        max_policy_diff = 0.0
        max_value_diff = 0.0

        comparison_start = (
            time.perf_counter()
        )

        for index, state in enumerate(
            states
        ):
            env = SplendorEnv()

            legal_actions = (
                env._legal_actions(
                    state
                )
            )

            expected = (
                references[
                    index
                ]
            )

            if (
                len(legal_actions)
                != expected["num_legal"]
            ):
                raise AssertionError(
                    f"State {index} legal-action "
                    "count changed between phases: "
                    f"{len(legal_actions)} != "
                    f"{expected['num_legal']}."
                )

            probs, value = (
                evaluator.evaluate(
                    env=env,
                    state=state,
                    legal_actions=(
                        legal_actions
                    ),
                )
            )

            policy_diff = (
                compare_policy(
                    index=index,
                    expected=(
                        expected["probs"]
                    ),
                    actual=probs,
                    atol=atol,
                    rtol=rtol,
                )
            )

            value_diff = (
                compare_value(
                    index=index,
                    expected=(
                        expected["value"]
                    ),
                    actual=value,
                    atol=atol,
                    rtol=rtol,
                )
            )

            max_policy_diff = max(
                max_policy_diff,
                policy_diff,
            )

            max_value_diff = max(
                max_value_diff,
                value_diff,
            )

            print(
                f"  state {index:02d}: PASS  "
                f"legal={len(legal_actions):2d}  "
                f"policy_diff={policy_diff:.3e}  "
                f"value_diff={value_diff:.3e}"
            )

        elapsed = (
            time.perf_counter()
            - comparison_start
        )

        client_stats = (
            evaluator.stats_snapshot()
        )

        print()
        print(
            "Multiprocess evaluation time: "
            f"{elapsed:.3f}s"
        )

        print(
            "Client average round trip: "
            f"{client_stats['average_round_trip_ms']:.3f} ms"
        )

        print(
            "Client max round trip: "
            f"{client_stats['max_round_trip_ms']:.3f} ms"
        )

        print(
            "Client responses: "
            f"{client_stats['responses_completed']} completed, "
            f"{client_stats['responses_failed']} failed"
        )

        if (
            client_stats[
                "responses_completed"
            ]
            != len(states)
        ):
            raise AssertionError(
                "Client completed response count "
                "does not match state count."
            )

        if (
            client_stats[
                "responses_failed"
            ]
            != 0
        ):
            raise AssertionError(
                "Client reported failed responses."
            )

        print()
        print(
            "Maximum policy absolute difference: "
            f"{max_policy_diff:.12g}"
        )

        print(
            "Maximum value absolute difference:  "
            f"{max_value_diff:.12g}"
        )

        return {
            "max_policy_diff":
                max_policy_diff,

            "max_value_diff":
                max_value_diff,

            "client_stats":
                client_stats,
        }

    finally:
        server.close(
            timeout_s=30.0,
            terminate_if_needed=True,
        )

        final_stats = (
            server.final_stats
        )

        print()
        print(
            "GPU SERVER FINAL TELEMETRY"
        )

        if final_stats is None:
            print(
                "  Final telemetry unavailable."
            )

        else:
            keys = [
                "requests_received",
                "requests_completed",
                "requests_failed",
                "batches_completed",
                "average_batch_size",
                "max_observed_batch_size",
                "average_request_queue_delay_ms",
                "average_batch_build_ms",
                "average_batch_inference_ms",
                "average_response_route_ms",
                "inference_positions_per_second",
                "end_to_end_positions_per_second",
            ]

            for key in keys:
                print(
                    f"  {key}: "
                    f"{final_stats.get(key)}"
                )

        if (
            server.exitcode is not None
            and server.exitcode != 0
        ):
            raise RuntimeError(
                "GPU inference server exited "
                f"with code {server.exitcode}."
            )


# ============================================================
# CLI
# ============================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare DirectNeuralEvaluator "
            "against centralized multiprocess "
            "GPU inference."
        )
    )

    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
    )

    parser.add_argument(
        "--device",
        default="cuda",
    )

    parser.add_argument(
        "--states",
        type=int,
        default=DEFAULT_STATES,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )

    parser.add_argument(
        "--stride",
        type=int,
        default=DEFAULT_STRIDE,
    )

    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=DEFAULT_MAX_BATCH_SIZE,
    )

    parser.add_argument(
        "--batch-wait-ms",
        type=float,
        default=DEFAULT_BATCH_WAIT_MS,
    )

    parser.add_argument(
        "--request-timeout-s",
        type=float,
        default=120.0,
    )

    parser.add_argument(
        "--atol",
        type=float,
        default=DEFAULT_ATOL,
    )

    parser.add_argument(
        "--rtol",
        type=float,
        default=DEFAULT_RTOL,
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================


def main():
    args = parse_args()

    if args.states < 1:
        raise ValueError(
            "--states must be >= 1."
        )

    if args.stride < 1:
        raise ValueError(
            "--stride must be >= 1."
        )

    checkpoint_path = Path(
        args.checkpoint
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            checkpoint_path
        )

    direct_device = (
        normalize_device(
            args.device
        )
    )

    if (
        direct_device.type == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA requested but unavailable."
        )

    print("=" * 78)
    print(
        "MULTIPROCESS GPU INFERENCE "
        "EQUIVALENCE SMOKE TEST"
    )
    print("=" * 78)

    print(
        "checkpoint:",
        checkpoint_path,
    )

    print(
        "device:",
        direct_device,
    )

    print(
        "states:",
        args.states,
    )

    print(
        "max_batch_size:",
        args.max_batch_size,
    )

    print(
        "batch_wait_ms:",
        args.batch_wait_ms,
    )

    print()
    print(
        "Sampling deterministic reachable states..."
    )

    states = sample_reachable_states(
        num_states=args.states,
        seed=args.seed,
        stride=args.stride,
    )

    print(
        f"Sampled {len(states)} states."
    )

    references = (
        compute_direct_references(
            checkpoint_path=(
                checkpoint_path
            ),
            states=states,
            device=direct_device,
        )
    )

    result = (
        run_multiprocess_comparison(
            checkpoint_path=(
                str(
                    checkpoint_path
                )
            ),
            states=states,
            references=references,
            device_string=(
                str(
                    direct_device
                )
            ),
            max_batch_size=(
                args.max_batch_size
            ),
            batch_wait_ms=(
                args.batch_wait_ms
            ),
            request_timeout_s=(
                args.request_timeout_s
            ),
            atol=args.atol,
            rtol=args.rtol,
        )
    )

    print()
    print("=" * 78)
    print(
        "PASS: DIRECT AND MULTIPROCESS "
        "INFERENCE MATCH"
    )
    print("=" * 78)

    print(
        f"states checked: {len(states)}"
    )

    print(
        "max policy diff: "
        f"{result['max_policy_diff']:.12g}"
    )

    print(
        "max value diff:  "
        f"{result['max_value_diff']:.12g}"
    )

    print()
    print(
        "Note: with one synchronous evaluator client, "
        "the server should normally observe batch size 1. "
        "This smoke test validates numerical equivalence "
        "and IPC correctness, not cross-worker batching."
    )


if __name__ == "__main__":
    mp.freeze_support()
    main()
