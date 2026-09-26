"""
Parallel end-to-end smoke test for centralized multiprocess inference.

This test proves all of the following together:

    1. multiple spawned CPU worker processes can run concurrently
    2. every worker uses MultiprocessNeuralEvaluator
    3. all workers share one GPUInferenceServerProcess
    4. requests from different processes are dynamically batched
    5. every returned policy/value matches DirectNeuralEvaluator
    6. worker processes remain CPU-only
    7. request IDs / response routing remain correct under concurrency

Unlike the one-worker equivalence smoke test, this test intentionally
synchronizes worker requests in rounds so the GPU server has several
requests available at the same time.

Run from repository root:

    python -m splendor_v1.mcts_batched.smoke_test_parallel_multiprocess_inference

Windows spawn semantics are explicitly supported.
"""

from __future__ import annotations

import argparse
import gc
import math
import multiprocessing as mp
import os
from pathlib import Path
from queue import Empty
import random
import time
import traceback

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

DEFAULT_WORKERS = 8
DEFAULT_REQUESTS_PER_WORKER = 6

DEFAULT_SEED = 43000
DEFAULT_STRIDE = 4

# Slightly wider than production's 0.5ms so this smoke test reliably
# observes cross-process batching even under Windows process scheduling.
DEFAULT_BATCH_WAIT_MS = 2.0

DEFAULT_ATOL = 1e-6
DEFAULT_RTOL = 1e-5

DEFAULT_READY_TIMEOUT_S = 120.0
DEFAULT_REQUEST_TIMEOUT_S = 120.0
DEFAULT_WORKER_TIMEOUT_S = 180.0


# ============================================================
# CHECKPOINT / DIRECT MODEL
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
            checkpoint_path
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
# REACHABLE CASES
# ============================================================


def sample_reachable_states(
    *,
    num_states,
    seed,
    stride,
):
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

        if game_index > 200:
            raise RuntimeError(
                "Unable to collect enough "
                "reachable test states."
            )

    return states


def build_reference_cases(
    *,
    checkpoint_path,
    device,
    num_cases,
    seed,
    stride,
):
    """
    Build direct reference outputs plus pure CPU IPC payloads.
    """

    print()
    print("=" * 78)
    print("PHASE A: BUILD DIRECT REFERENCES")
    print("=" * 78)

    states = sample_reachable_states(
        num_states=num_cases,
        seed=seed,
        stride=stride,
    )

    print(
        f"Collected {len(states)} reachable states."
    )

    model = load_direct_model(
        checkpoint_path=checkpoint_path,
        device=device,
    )

    evaluator = DirectNeuralEvaluator(
        model=model
    )

    cases = []

    started = time.perf_counter()

    for case_id, state in enumerate(
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
                f"Case {case_id} has no legal actions."
            )

        direct_probs, direct_value = (
            evaluator.evaluate(
                env=env,
                state=state,
                legal_actions=legal_actions,
            )
        )

        observation = (
            env.observation_encoder.encoder(
                state
            )
        )

        observation = np.ascontiguousarray(
            np.asarray(
                observation,
                dtype=np.float32,
            )
        )

        legal_action_ids = [
            int(
                env.action_to_id(
                    action
                )
            )
            for action in legal_actions
        ]

        cases.append(
            {
                "case_id":
                    int(
                        case_id
                    ),

                "observation":
                    observation,

                "legal_action_ids":
                    legal_action_ids,

                "expected_probs":
                    (
                        direct_probs
                        .detach()
                        .float()
                        .cpu()
                        .numpy()
                        .astype(
                            np.float32,
                            copy=True,
                        )
                    ),

                "expected_value":
                    float(
                        direct_value
                    ),
            }
        )

    elapsed = (
        time.perf_counter()
        - started
    )

    if device.type == "cuda":
        torch.cuda.synchronize(
            device
        )

    del evaluator
    del model
    del states

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(
        "Direct references built in "
        f"{elapsed:.3f}s."
    )

    print(
        "Parent Model 4 released before "
        "starting centralized GPU server."
    )

    return cases


# ============================================================
# WORKER PROCESS
# ============================================================


def _parallel_worker_main(
    *,
    worker_id,
    assigned_cases,
    request_queue,
    response_queue,
    round_barrier,
    result_queue,
    request_timeout_s,
):
    """
    Top-level spawned worker entry point.

    The worker never loads Model 4 and never initializes CUDA.
    """

    try:
        evaluator = (
            MultiprocessNeuralEvaluator(
                worker_id=worker_id,
                request_queue=request_queue,
                response_queue=response_queue,
                request_timeout_s=request_timeout_s,
                return_policy_device="cpu",
            )
        )

        pid = os.getpid()

        result_queue.put(
            {
                "kind": "worker_ready",
                "worker_id": int(worker_id),
                "pid": int(pid),
            }
        )

        worker_results = []

        for round_index, case in enumerate(
            assigned_cases
        ):
            # Release all workers into evaluate_encoded together.
            round_barrier.wait(
                timeout=60.0
            )

            started = (
                time.perf_counter()
            )

            probs, value = (
                evaluator.evaluate_encoded(
                    observation=(
                        case["observation"]
                    ),
                    legal_action_ids=(
                        case["legal_action_ids"]
                    ),
                )
            )

            elapsed = (
                time.perf_counter()
                - started
            )

            worker_results.append(
                {
                    "case_id":
                        int(
                            case["case_id"]
                        ),

                    "round_index":
                        int(
                            round_index
                        ),

                    "probs":
                        (
                            probs
                            .detach()
                            .float()
                            .cpu()
                            .numpy()
                            .astype(
                                np.float32,
                                copy=True,
                            )
                        ),

                    "value":
                        float(
                            value
                        ),

                    "elapsed_seconds":
                        float(
                            elapsed
                        ),
                }
            )

        result_queue.put(
            {
                "kind": "worker_complete",
                "worker_id": int(worker_id),
                "pid": int(pid),
                "results": worker_results,
                "client_stats":
                    evaluator.stats_snapshot(),
            }
        )

    except BaseException as exc:
        try:
            result_queue.put(
                {
                    "kind": "worker_error",
                    "worker_id": int(worker_id),
                    "pid": int(
                        os.getpid()
                    ),
                    "error": (
                        f"{type(exc).__name__}: "
                        f"{exc}\n"
                        f"{traceback.format_exc()}"
                    ),
                }
            )

        finally:
            raise


# ============================================================
# COMPARISON HELPERS
# ============================================================


def compare_case(
    *,
    case,
    actual_probs,
    actual_value,
    atol,
    rtol,
):
    expected_probs = np.asarray(
        case["expected_probs"],
        dtype=np.float32,
    )

    actual_probs = np.asarray(
        actual_probs,
        dtype=np.float32,
    )

    if (
        expected_probs.shape
        != actual_probs.shape
    ):
        raise AssertionError(
            f"Case {case['case_id']} policy "
            "shape mismatch: "
            f"{expected_probs.shape} != "
            f"{actual_probs.shape}"
        )

    policy_diff = float(
        np.max(
            np.abs(
                expected_probs
                - actual_probs
            )
        )
        if expected_probs.size > 0
        else 0.0
    )

    if not np.allclose(
        expected_probs,
        actual_probs,
        atol=atol,
        rtol=rtol,
    ):
        raise AssertionError(
            f"Case {case['case_id']} policy "
            "mismatch. "
            f"max_abs_diff={policy_diff:.12g}"
        )

    expected_value = float(
        case["expected_value"]
    )

    actual_value = float(
        actual_value
    )

    value_diff = abs(
        expected_value
        - actual_value
    )

    if not math.isclose(
        expected_value,
        actual_value,
        abs_tol=atol,
        rel_tol=rtol,
    ):
        raise AssertionError(
            f"Case {case['case_id']} value "
            "mismatch: "
            f"direct={expected_value:.12g}, "
            f"parallel={actual_value:.12g}, "
            f"abs_diff={value_diff:.12g}"
        )

    return (
        policy_diff,
        value_diff,
    )


# ============================================================
# PARALLEL PHASE
# ============================================================


def run_parallel_phase(
    *,
    checkpoint_path,
    device_string,
    cases,
    num_workers,
    requests_per_worker,
    batch_wait_ms,
    request_timeout_s,
    worker_timeout_s,
    atol,
    rtol,
    require_full_batch,
):
    print()
    print("=" * 78)
    print("PHASE B: PARALLEL CPU WORKERS -> ONE GPU SERVER")
    print("=" * 78)

    context = mp.get_context(
        "spawn"
    )

    ipc = create_inference_ipc(
        num_workers=num_workers,
        mp_context=context,
    )

    # One reusable synchronization barrier for every request round.
    round_barrier = context.Barrier(
        num_workers
    )

    result_queue = context.Queue()

    server = GPUInferenceServerProcess(
        config=GPUInferenceServerConfig(
            checkpoint_path=(
                checkpoint_path
            ),
            max_batch_size=(
                num_workers
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

    workers = []

    try:
        ready = (
            server.wait_until_ready(
                timeout_s=120.0
            )
        )

        print(
            f"GPU server READY pid={ready.pid}"
        )

        # Assign one case per worker per round so the barrier creates
        # exactly num_workers simultaneous outstanding requests.
        assignments = [
            []
            for _ in range(
                num_workers
            )
        ]

        case_lookup = {}

        for case in cases:
            case_lookup[
                int(
                    case["case_id"]
                )
            ] = case

        expected_case_count = (
            num_workers
            * requests_per_worker
        )

        if len(cases) != expected_case_count:
            raise AssertionError(
                "Internal case count mismatch."
            )

        cursor = 0

        for round_index in range(
            requests_per_worker
        ):
            for worker_id in range(
                num_workers
            ):
                assignments[
                    worker_id
                ].append(
                    cases[
                        cursor
                    ]
                )

                cursor += 1

        for worker_id in range(
            num_workers
        ):
            process = context.Process(
                target=_parallel_worker_main,
                kwargs={
                    "worker_id":
                        worker_id,

                    "assigned_cases":
                        assignments[
                            worker_id
                        ],

                    "request_queue":
                        ipc.request_queue,

                    "response_queue":
                        ipc.response_queues[
                            worker_id
                        ],

                    "round_barrier":
                        round_barrier,

                    "result_queue":
                        result_queue,

                    "request_timeout_s":
                        request_timeout_s,
                },
                name=(
                    "SplendorParallelSmokeWorker"
                    f"-{worker_id}"
                ),
                daemon=False,
            )

            process.start()

            workers.append(
                process
            )

        # ----------------------------------------------------
        # Read all worker status/results
        # ----------------------------------------------------

        ready_workers = {}
        completed_workers = {}
        worker_errors = []

        deadline = (
            time.perf_counter()
            + worker_timeout_s
        )

        while (
            len(completed_workers)
            + len(worker_errors)
            < num_workers
        ):
            remaining = (
                deadline
                - time.perf_counter()
            )

            if remaining <= 0:
                raise TimeoutError(
                    "Timed out waiting for parallel "
                    "smoke-test workers."
                )

            try:
                message = result_queue.get(
                    timeout=min(
                        remaining,
                        1.0,
                    )
                )

            except Empty:
                # Detect hard worker crashes that could otherwise
                # leave us waiting until the global timeout.
                for worker_id, process in enumerate(
                    workers
                ):
                    if (
                        process.exitcode is not None
                        and process.exitcode != 0
                        and worker_id not in completed_workers
                        and not any(
                            error[
                                "worker_id"
                            ]
                            == worker_id
                            for error
                            in worker_errors
                        )
                    ):
                        worker_errors.append(
                            {
                                "worker_id":
                                    worker_id,

                                "error":
                                    (
                                        "Worker exited "
                                        f"with code "
                                        f"{process.exitcode} "
                                        "without a normal "
                                        "result message."
                                    ),
                            }
                        )

                continue

            kind = message.get(
                "kind"
            )

            if kind == "worker_ready":
                ready_workers[
                    int(
                        message[
                            "worker_id"
                        ]
                    )
                ] = int(
                    message[
                        "pid"
                    ]
                )

                print(
                    "  worker "
                    f"{message['worker_id']} READY "
                    f"pid={message['pid']}"
                )

            elif kind == "worker_complete":
                completed_workers[
                    int(
                        message[
                            "worker_id"
                        ]
                    )
                ] = message

            elif kind == "worker_error":
                worker_errors.append(
                    message
                )

        # ----------------------------------------------------
        # Join and validate process exit
        # ----------------------------------------------------

        for process in workers:
            process.join(
                timeout=10.0
            )

        for worker_id, process in enumerate(
            workers
        ):
            if process.is_alive():
                process.terminate()
                process.join(
                    timeout=5.0
                )

                worker_errors.append(
                    {
                        "worker_id":
                            worker_id,

                        "error":
                            "Worker failed to exit.",
                    }
                )

            elif process.exitcode != 0:
                if not any(
                    int(
                        error[
                            "worker_id"
                        ]
                    )
                    == worker_id
                    for error
                    in worker_errors
                ):
                    worker_errors.append(
                        {
                            "worker_id":
                                worker_id,

                            "error":
                                (
                                    "Worker exitcode="
                                    f"{process.exitcode}"
                                ),
                        }
                    )

        if worker_errors:
            formatted = "\n\n".join(
                (
                    "worker "
                    f"{error['worker_id']}:\n"
                    f"{error['error']}"
                )
                for error
                in worker_errors
            )

            raise RuntimeError(
                "Parallel worker failures:\n"
                + formatted
            )

        if len(completed_workers) != num_workers:
            raise AssertionError(
                "Not all workers completed."
            )

        # ----------------------------------------------------
        # Numerical comparisons
        # ----------------------------------------------------

        max_policy_diff = 0.0
        max_value_diff = 0.0

        round_trip_ms = []
        worker_pids = set()

        result_count = 0

        for worker_id in sorted(
            completed_workers
        ):
            message = (
                completed_workers[
                    worker_id
                ]
            )

            worker_pids.add(
                int(
                    message[
                        "pid"
                    ]
                )
            )

            client_stats = (
                message[
                    "client_stats"
                ]
            )

            if (
                client_stats[
                    "responses_completed"
                ]
                != requests_per_worker
            ):
                raise AssertionError(
                    f"Worker {worker_id} completed "
                    f"{client_stats['responses_completed']} "
                    "responses, expected "
                    f"{requests_per_worker}."
                )

            if (
                client_stats[
                    "responses_failed"
                ]
                != 0
            ):
                raise AssertionError(
                    f"Worker {worker_id} reported "
                    "failed responses."
                )

            round_trip_ms.append(
                float(
                    client_stats[
                        "average_round_trip_ms"
                    ]
                )
            )

            for actual in message[
                "results"
            ]:
                case = case_lookup[
                    int(
                        actual[
                            "case_id"
                        ]
                    )
                ]

                (
                    policy_diff,
                    value_diff,
                ) = compare_case(
                    case=case,
                    actual_probs=(
                        actual[
                            "probs"
                        ]
                    ),
                    actual_value=(
                        actual[
                            "value"
                        ]
                    ),
                    atol=atol,
                    rtol=rtol,
                )

                max_policy_diff = max(
                    max_policy_diff,
                    policy_diff,
                )

                max_value_diff = max(
                    max_value_diff,
                    value_diff,
                )

                result_count += 1

        if result_count != len(cases):
            raise AssertionError(
                "Result count mismatch: "
                f"{result_count} != "
                f"{len(cases)}."
            )

        if len(worker_pids) != num_workers:
            raise AssertionError(
                "Workers did not run in distinct "
                "OS processes."
            )

        if int(
            ready.pid
        ) in worker_pids:
            raise AssertionError(
                "GPU server PID unexpectedly "
                "matches a worker PID."
            )

        print()
        print(
            "Numerical comparisons: PASS"
        )

        print(
            "Distinct CPU worker PIDs: "
            f"{len(worker_pids)}"
        )

        print(
            "Max policy absolute difference: "
            f"{max_policy_diff:.12g}"
        )

        print(
            "Max value absolute difference:  "
            f"{max_value_diff:.12g}"
        )

        if round_trip_ms:
            print(
                "Mean worker average round trip: "
                f"{np.mean(round_trip_ms):.3f} ms"
            )

        return {
            "max_policy_diff":
                max_policy_diff,

            "max_value_diff":
                max_value_diff,

            "worker_pids":
                sorted(
                    worker_pids
                ),

            "worker_average_round_trip_ms":
                round_trip_ms,
        }

    finally:
        for process in workers:
            if process.is_alive():
                process.terminate()

            process.join(
                timeout=5.0
            )

        server.close(
            timeout_s=30.0,
            terminate_if_needed=True,
        )

        final_stats = (
            server.final_stats
        )

        print()
        print("=" * 78)
        print("GPU SERVER FINAL TELEMETRY")
        print("=" * 78)

        if final_stats is None:
            print(
                "Final server telemetry unavailable."
            )

        else:
            telemetry_keys = [
                "requests_received",
                "requests_completed",
                "requests_failed",
                "batches_completed",
                "average_batch_size",
                "max_observed_batch_size",
                "configured_max_batch_size",
                "batch_fill_fraction",
                "average_collection_wait_ms",
                "average_request_queue_delay_ms",
                "average_batch_build_ms",
                "average_batch_inference_ms",
                "average_response_route_ms",
                "inference_positions_per_second",
                "end_to_end_positions_per_second",
                "batch_size_histogram",
            ]

            for key in telemetry_keys:
                print(
                    f"{key}: "
                    f"{final_stats.get(key)}"
                )

            if (
                int(
                    final_stats[
                        "requests_completed"
                    ]
                )
                != len(cases)
            ):
                raise AssertionError(
                    "GPU server completed-request "
                    "count mismatch."
                )

            if (
                int(
                    final_stats[
                        "requests_failed"
                    ]
                )
                != 0
            ):
                raise AssertionError(
                    "GPU server reported failed "
                    "requests."
                )

            if (
                int(
                    final_stats[
                        "max_observed_batch_size"
                    ]
                )
                <= 1
            ):
                raise AssertionError(
                    "No cross-process batching was "
                    "observed: max batch size <= 1."
                )

            if require_full_batch:
                if (
                    int(
                        final_stats[
                            "max_observed_batch_size"
                        ]
                    )
                    < num_workers
                ):
                    raise AssertionError(
                        "Full batch was required but "
                        "never observed: "
                        f"max={final_stats['max_observed_batch_size']} "
                        f"workers={num_workers}."
                    )

        if (
            server.exitcode is not None
            and server.exitcode != 0
        ):
            raise RuntimeError(
                "GPU server exitcode="
                f"{server.exitcode}"
            )


# ============================================================
# CLI
# ============================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Parallel multiprocessing smoke test "
            "for centralized GPU inference."
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
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
    )

    parser.add_argument(
        "--requests-per-worker",
        type=int,
        default=DEFAULT_REQUESTS_PER_WORKER,
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
        "--batch-wait-ms",
        type=float,
        default=DEFAULT_BATCH_WAIT_MS,
    )

    parser.add_argument(
        "--request-timeout-s",
        type=float,
        default=DEFAULT_REQUEST_TIMEOUT_S,
    )

    parser.add_argument(
        "--worker-timeout-s",
        type=float,
        default=DEFAULT_WORKER_TIMEOUT_S,
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

    parser.add_argument(
        "--require-full-batch",
        action="store_true",
        help=(
            "Require at least one GPU batch "
            "whose size equals --workers."
        ),
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================


def main():
    args = parse_args()

    if args.workers < 2:
        raise ValueError(
            "--workers must be >= 2 for "
            "the parallel smoke test."
        )

    if args.requests_per_worker < 1:
        raise ValueError(
            "--requests-per-worker must be >= 1."
        )

    if args.stride < 1:
        raise ValueError(
            "--stride must be >= 1."
        )

    checkpoint = Path(
        args.checkpoint
    )

    if not checkpoint.exists():
        raise FileNotFoundError(
            checkpoint
        )

    device = normalize_device(
        args.device
    )

    if (
        device.type == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA requested but unavailable."
        )

    num_cases = (
        args.workers
        * args.requests_per_worker
    )

    print("=" * 78)
    print("PARALLEL MULTIPROCESS GPU INFERENCE SMOKE TEST")
    print("=" * 78)

    print(
        "checkpoint:",
        checkpoint,
    )

    print(
        "device:",
        device,
    )

    print(
        "workers:",
        args.workers,
    )

    print(
        "requests_per_worker:",
        args.requests_per_worker,
    )

    print(
        "total_requests:",
        num_cases,
    )

    print(
        "server_max_batch_size:",
        args.workers,
    )

    print(
        "batch_wait_ms:",
        args.batch_wait_ms,
    )

    cases = build_reference_cases(
        checkpoint_path=checkpoint,
        device=device,
        num_cases=num_cases,
        seed=args.seed,
        stride=args.stride,
    )

    result = run_parallel_phase(
        checkpoint_path=str(
            checkpoint
        ),
        device_string=str(
            device
        ),
        cases=cases,
        num_workers=args.workers,
        requests_per_worker=(
            args.requests_per_worker
        ),
        batch_wait_ms=(
            args.batch_wait_ms
        ),
        request_timeout_s=(
            args.request_timeout_s
        ),
        worker_timeout_s=(
            args.worker_timeout_s
        ),
        atol=args.atol,
        rtol=args.rtol,
        require_full_batch=(
            args.require_full_batch
        ),
    )

    print()
    print("=" * 78)
    print("PASS: PARALLEL MULTIPROCESS INFERENCE IS CORRECT")
    print("=" * 78)

    print(
        "workers checked:",
        args.workers,
    )

    print(
        "requests checked:",
        num_cases,
    )

    print(
        "max policy diff:",
        f"{result['max_policy_diff']:.12g}",
    )

    print(
        "max value diff:",
        f"{result['max_value_diff']:.12g}",
    )

    print()
    print(
        "Success means:"
    )

    print(
        "  - workers ran in distinct OS processes"
    )

    print(
        "  - one shared GPU process served them"
    )

    print(
        "  - cross-process neural batching occurred"
    )

    print(
        "  - every policy/value matched direct inference"
    )


if __name__ == "__main__":
    mp.freeze_support()
    main()
