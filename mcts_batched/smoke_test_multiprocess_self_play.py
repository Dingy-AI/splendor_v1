"""
Full-game multiprocessing smoke test.

This replaces synthetic inference requests with real independent
Splendor self-play games.

The test launches:

    one centralized GPU inference process
        +
    N spawned CPU self-play workers
        +
    one real V5 MCTS / ModelReplayGenerator game per worker

The search budget is intentionally reduced by default so this remains
a smoke test rather than a long benchmark.

Run:

    python -m splendor_v1.mcts_batched.smoke_test_multiprocess_self_play

A successful run proves:

    - real Splendor games run in distinct OS processes
    - V5 MCTS can use MultiprocessNeuralEvaluator
    - one GPU process serves all games
    - real MCTS leaf requests batch across processes
    - replay samples/search metadata are generated
    - no CUDA model exists in the game workers
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
from pathlib import Path
from queue import Empty
import statistics
import time

from splendor_v1.mcts_batched.gpu_inference_server import (
    GPUInferenceServerConfig,
    GPUInferenceServerProcess,
    create_inference_ipc,
)

from splendor_v1.mcts_batched.multiprocess_self_play_worker import (
    MultiprocessSelfPlayPool,
    SelfPlayGameJob,
    SelfPlayGameResult,
    SelfPlaySearchConfig,
    SelfPlayWorkerConfig,
    SelfPlayWorkerReady,
)


DEFAULT_CHECKPOINT = (
    "splendor_v1/training_v5/data/"
    "model_1900_games.pt"
)

DEFAULT_WORKERS = 4
DEFAULT_SEED = 44000

# Deliberately smaller than production 400/80.
DEFAULT_SIMULATIONS = 40
DEFAULT_MIN_SIMULATIONS = 20
DEFAULT_CHECK_INTERVAL = 10
DEFAULT_TARGET_VISITS_PER_ACTION = 2.0
DEFAULT_SINGLE_ACTION_SIMULATIONS = 2
DEFAULT_STABILITY_CHECKS = 2

DEFAULT_BATCH_WAIT_MS = 2.0
DEFAULT_TIMEOUT_S = 300.0


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run real V5 self-play games in "
            "separate CPU processes against one "
            "centralized GPU inference server."
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
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )

    parser.add_argument(
        "--simulations",
        type=int,
        default=DEFAULT_SIMULATIONS,
    )

    parser.add_argument(
        "--min-simulations",
        type=int,
        default=DEFAULT_MIN_SIMULATIONS,
    )

    parser.add_argument(
        "--check-interval",
        type=int,
        default=DEFAULT_CHECK_INTERVAL,
    )

    parser.add_argument(
        "--target-visits-per-action",
        type=float,
        default=DEFAULT_TARGET_VISITS_PER_ACTION,
    )

    parser.add_argument(
        "--single-action-simulations",
        type=int,
        default=DEFAULT_SINGLE_ACTION_SIMULATIONS,
    )

    parser.add_argument(
        "--stability-checks",
        type=int,
        default=DEFAULT_STABILITY_CHECKS,
    )

    parser.add_argument(
        "--batch-wait-ms",
        type=float,
        default=DEFAULT_BATCH_WAIT_MS,
    )

    parser.add_argument(
        "--timeout-s",
        type=float,
        default=DEFAULT_TIMEOUT_S,
    )

    parser.add_argument(
        "--require-full-batch",
        action="store_true",
        help=(
            "Require the GPU server to observe "
            "at least one batch equal to the "
            "number of workers."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.workers < 2:
        raise ValueError(
            "--workers must be >= 2."
        )

    checkpoint = Path(
        args.checkpoint
    )

    if not checkpoint.exists():
        raise FileNotFoundError(
            checkpoint
        )

    context = mp.get_context(
        "spawn"
    )

    ipc = create_inference_ipc(
        num_workers=args.workers,
        mp_context=context,
    )

    server = GPUInferenceServerProcess(
        config=GPUInferenceServerConfig(
            checkpoint_path=str(
                checkpoint
            ),
            max_batch_size=(
                args.workers
            ),
            batch_wait_ms=(
                args.batch_wait_ms
            ),
            device=args.device,
            stats_report_interval_batches=0,
        ),
        ipc=ipc,
        mp_context=context,
    )

    worker_config = (
        SelfPlayWorkerConfig(
            search=SelfPlaySearchConfig(
                simulations=(
                    args.simulations
                ),
                min_simulations=(
                    args.min_simulations
                ),
                check_interval=(
                    args.check_interval
                ),
                target_visits_per_action=(
                    args.target_visits_per_action
                ),
                single_action_simulations=(
                    args.single_action_simulations
                ),
                stability_checks=(
                    args.stability_checks
                ),
                max_game_steps=300,
            ),
            request_timeout_s=(
                args.timeout_s
            ),
            model_generation=4,
            model_checkpoint_label=(
                "multiprocess_full_game_smoke"
            ),
            split="train",
            return_samples=False,
        )
    )

    pool = MultiprocessSelfPlayPool(
        num_workers=args.workers,
        inference_ipc=ipc,
        config=worker_config,
        mp_context=context,
    )

    print("=" * 78)
    print("MULTIPROCESS REAL SELF-PLAY SMOKE TEST")
    print("=" * 78)

    print(
        "workers:",
        args.workers,
    )

    print(
        "simulations:",
        args.simulations,
    )

    print(
        "min_simulations:",
        args.min_simulations,
    )

    print(
        "batch_wait_ms:",
        args.batch_wait_ms,
    )

    print(
        "checkpoint:",
        checkpoint,
    )

    wall_start = (
        time.perf_counter()
    )

    results = []
    ready_workers = {}

    try:
        server.start()

        ready = server.wait_until_ready(
            timeout_s=args.timeout_s
        )

        print(
            f"GPU server READY pid={ready.pid}"
        )

        pool.start()

        # Wait until every CPU process has initialized its evaluator
        # and is blocked waiting for a game job.
        ready_deadline = (
            time.perf_counter()
            + args.timeout_s
        )

        while (
            len(ready_workers)
            < args.workers
        ):
            remaining = (
                ready_deadline
                - time.perf_counter()
            )

            if remaining <= 0:
                raise TimeoutError(
                    "Timed out waiting for workers "
                    "to report READY."
                )

            try:
                message = pool.get_result(
                    timeout=min(
                        remaining,
                        1.0,
                    )
                )

            except Empty:
                # Normal on Windows spawn: importing the project and
                # initializing a fresh Python process can easily take
                # longer than one second. Keep polling unless a worker
                # has actually crashed.
                crashed = [
                    (
                        index,
                        process.exitcode,
                    )
                    for index, process
                    in enumerate(
                        pool.processes
                    )
                    if (
                        process.exitcode is not None
                        and process.exitcode != 0
                    )
                ]

                if crashed:
                    raise RuntimeError(
                        "One or more self-play workers "
                        "exited before reporting READY: "
                        f"{crashed}"
                    )

                continue

            if isinstance(
                message,
                SelfPlayWorkerReady,
            ):
                ready_workers[
                    int(
                        message.worker_id
                    )
                ] = int(
                    message.pid
                )

                print(
                    f"CPU worker "
                    f"{message.worker_id} READY "
                    f"pid={message.pid}"
                )

        if (
            len(
                set(
                    ready_workers.values()
                )
            )
            != args.workers
        ):
            raise AssertionError(
                "Worker PIDs are not distinct."
            )

        if int(
            ready.pid
        ) in set(
            ready_workers.values()
        ):
            raise AssertionError(
                "GPU server PID matches a CPU "
                "worker PID."
            )

        # Submit exactly one game per waiting worker.
        jobs = [
            SelfPlayGameJob(
                game_id=index,
                seed=(
                    int(
                        args.seed
                    )
                    + index
                ),
                extra_game_metadata={
                    "smoke_test":
                        True,
                },
            )
            for index in range(
                args.workers
            )
        ]

        pool.submit_many(
            jobs
        )

        result_deadline = (
            time.perf_counter()
            + args.timeout_s
        )

        while len(results) < args.workers:
            remaining = (
                result_deadline
                - time.perf_counter()
            )

            if remaining <= 0:
                raise TimeoutError(
                    "Timed out waiting for real "
                    "self-play games."
                )

            try:
                message = pool.get_result(
                    timeout=min(
                        remaining,
                        1.0,
                    )
                )

            except Empty:
                # A one-second empty poll is expected while real MCTS
                # games are running. Only fail early if a worker has
                # actually died.
                crashed = [
                    (
                        index,
                        process.exitcode,
                    )
                    for index, process
                    in enumerate(
                        pool.processes
                    )
                    if (
                        process.exitcode is not None
                        and process.exitcode != 0
                    )
                ]

                if crashed:
                    raise RuntimeError(
                        "One or more self-play workers "
                        "crashed while games were running: "
                        f"{crashed}"
                    )

                continue

            if isinstance(
                message,
                SelfPlayGameResult,
            ):
                results.append(
                    message
                )

                status = (
                    "PASS"
                    if message.success
                    else "FAIL"
                )

                print(
                    f"game={message.game_id} "
                    f"seed={message.seed} "
                    f"worker={message.worker_id} "
                    f"pid={message.pid} "
                    f"{status} "
                    f"positions={message.num_positions} "
                    f"seconds={message.seconds:.2f}"
                )

                if not message.success:
                    print(
                        message.traceback
                    )

        failures = [
            result
            for result in results
            if not result.success
        ]

        if failures:
            raise RuntimeError(
                f"{len(failures)} real self-play "
                "game(s) failed."
            )

        result_worker_ids = {
            int(
                result.worker_id
            )
            for result
            in results
        }

        if len(
            result_worker_ids
        ) != args.workers:
            raise AssertionError(
                "Expected one completed game per "
                "worker in smoke test."
            )

        for result in results:
            if result.num_positions <= 0:
                raise AssertionError(
                    "A completed game generated zero "
                    "replay positions."
                )

            search = (
                result.search_summary
                or {}
            )

            if (
                search.get(
                    "search_samples_with_actual_simulations",
                    0,
                )
                <= 0
            ):
                raise AssertionError(
                    "A game did not record V5 search "
                    "simulation metadata."
                )

            evaluator_stats = (
                result.evaluator_stats
                or {}
            )

            if (
                evaluator_stats.get(
                    "responses_completed",
                    0,
                )
                <= 0
            ):
                raise AssertionError(
                    "A game worker completed no "
                    "neural inference responses."
                )

            if (
                evaluator_stats.get(
                    "responses_failed",
                    0,
                )
                != 0
            ):
                raise AssertionError(
                    "A game worker reported failed "
                    "neural responses."
                )

    finally:
        pool.close(
            timeout_s=30.0,
            terminate_if_needed=True,
        )

        server.close(
            timeout_s=30.0,
            terminate_if_needed=True,
        )

    wall_seconds = (
        time.perf_counter()
        - wall_start
    )

    server_stats = (
        server.final_stats
    )

    print()
    print("=" * 78)
    print("GPU SERVER TELEMETRY")
    print("=" * 78)

    if server_stats is None:
        raise AssertionError(
            "GPU server final telemetry missing."
        )

    telemetry_keys = [
        "requests_received",
        "requests_completed",
        "requests_failed",
        "batches_completed",
        "average_batch_size",
        "max_observed_batch_size",
        "batch_fill_fraction",
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
            f"{server_stats.get(key)}"
        )

    if (
        int(
            server_stats[
                "requests_failed"
            ]
        )
        != 0
    ):
        raise AssertionError(
            "GPU server reported failed "
            "inference requests."
        )

    if (
        int(
            server_stats[
                "max_observed_batch_size"
            ]
        )
        <= 1
    ):
        raise AssertionError(
            "Real MCTS games did not create "
            "cross-process GPU batches."
        )

    if args.require_full_batch:
        if (
            int(
                server_stats[
                    "max_observed_batch_size"
                ]
            )
            < args.workers
        ):
            raise AssertionError(
                "Full worker-sized GPU batch "
                "was required but not observed."
            )

    total_positions = sum(
        result.num_positions
        for result in results
    )

    game_seconds = [
        result.seconds
        for result in results
    ]

    games_per_hour = (
        len(results)
        / wall_seconds
        * 3600.0
        if wall_seconds > 0
        else 0.0
    )

    print()
    print("=" * 78)
    print("PASS: REAL MULTIPROCESS SELF-PLAY WORKS")
    print("=" * 78)

    print(
        f"games: {len(results)}"
    )

    print(
        f"total_positions: {total_positions}"
    )

    print(
        "wall_seconds: "
        f"{wall_seconds:.3f}"
    )

    print(
        "games_per_hour: "
        f"{games_per_hour:.2f}"
    )

    print(
        "average_individual_game_seconds: "
        f"{statistics.mean(game_seconds):.3f}"
    )

    print(
        "max_observed_gpu_batch: "
        f"{server_stats['max_observed_batch_size']}"
    )

    print()
    print(
        "This validates the production-shaped path:"
    )

    print(
        "CPU game processes -> V5 MCTS -> "
        "MultiprocessNeuralEvaluator -> "
        "one GPU server."
    )


if __name__ == "__main__":
    mp.freeze_support()
    main()
