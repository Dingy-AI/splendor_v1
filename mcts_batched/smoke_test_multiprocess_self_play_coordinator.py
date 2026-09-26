"""
Smoke test for MultiprocessSelfPlayCoordinator.

This validates the final production-shaped ownership model:

    coordinator/main process
        owns persistent ReplayBuffer

    CPU worker processes
        generate complete real games

    centralized GPU process
        owns Model 4 / CUDA

    completed games
        are committed in MAIN via ReplayBuffer.add_game(...)

The default search budget is deliberately small.

Run from repository root:

    python -m splendor_v1.mcts_batched.smoke_test_multiprocess_self_play_coordinator
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
from pathlib import Path

from splendor_v1.training_v2.replay_buffer import (
    ReplayBuffer,
)

from splendor_v1.mcts_batched.multiprocess_self_play_worker import (
    SelfPlaySearchConfig,
    SelfPlayWorkerConfig,
)

from splendor_v1.mcts_batched.multiprocess_self_play_coordinator import (
    MultiprocessSelfPlayCoordinator,
    MultiprocessSelfPlayCoordinatorConfig,
)


DEFAULT_CHECKPOINT = (
    "splendor_v1/training_v5/data/"
    "model_1900_games.pt"
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--games",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=45000,
    )

    parser.add_argument(
        "--simulations",
        type=int,
        default=40,
    )

    parser.add_argument(
        "--min-simulations",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--batch-wait-ms",
        type=float,
        default=2.0,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    checkpoint = Path(
        args.checkpoint
    )

    if not checkpoint.exists():
        raise FileNotFoundError(
            checkpoint
        )

    replay_buffer = ReplayBuffer(
        capacity=100_000
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
                check_interval=10,
                target_visits_per_action=2.0,
                single_action_simulations=2,
                stability_checks=2,
                max_game_steps=300,
            ),
            replay_capacity=10_000,
            request_timeout_s=300.0,
            request_put_timeout_s=30.0,
            model_generation=4,
            model_checkpoint_label=(
                "coordinator_smoke"
            ),
            split="train",
            return_samples=True,
        )
    )

    config = (
        MultiprocessSelfPlayCoordinatorConfig(
            num_workers=(
                args.workers
            ),
            checkpoint_path=str(
                checkpoint
            ),
            worker_config=(
                worker_config
            ),
            device="cuda",
            max_batch_size=(
                args.workers
            ),
            batch_wait_ms=(
                args.batch_wait_ms
            ),
            startup_timeout_s=180.0,
            game_result_timeout_s=600.0,
            shutdown_timeout_s=30.0,
        )
    )

    context = mp.get_context(
        "spawn"
    )

    coordinator = (
        MultiprocessSelfPlayCoordinator(
            replay_buffer=(
                replay_buffer
            ),
            config=config,
            mp_context=context,
        )
    )

    def progress(
        info,
    ):
        print(
            "committed="
            f"{info['committed_games']}/"
            f"{info['requested_games']} "
            "in_flight="
            f"{info['in_flight_games']} "
            "games/hour="
            f"{info['games_per_hour']:.2f}"
        )

    summary = coordinator.run_block(
        num_games=args.games,
        seed_start=args.seed,
        game_id_start=0,
        extra_game_metadata={
            "coordinator_smoke":
                True,
        },
        progress_callback=progress,
    )

    if (
        summary[
            "committed_games"
        ]
        != args.games
    ):
        raise AssertionError(
            "Committed game count mismatch."
        )

    if (
        summary[
            "total_positions"
        ]
        <= 0
    ):
        raise AssertionError(
            "No replay positions were committed."
        )

    # ReplayBuffer tracks completed game boundaries separately.
    replay_games = getattr(
        replay_buffer,
        "games",
        None,
    )

    if replay_games is not None:
        if len(
            replay_games
        ) < args.games:
            raise AssertionError(
                "Persistent replay buffer did not "
                "record all completed games."
            )

    gpu = (
        summary.get(
            "gpu_server"
        )
        or {}
    )

    if (
        int(
            gpu.get(
                "requests_failed",
                0,
            )
        )
        != 0
    ):
        raise AssertionError(
            "GPU server reported inference failures."
        )

    if (
        int(
            gpu.get(
                "max_observed_batch_size",
                0,
            )
        )
        <= 1
    ):
        raise AssertionError(
            "No cross-process GPU batching observed."
        )

    print()
    print("=" * 78)
    print("PASS: MULTIPROCESS SELF-PLAY COORDINATOR")
    print("=" * 78)

    for key in (
        "committed_games",
        "total_positions",
        "wall_seconds",
        "games_per_hour",
        "average_positions_per_game",
        "average_individual_game_seconds",
        "average_actual_simulations",
        "simulation_savings_fraction",
    ):
        print(
            f"{key}: "
            f"{summary.get(key)}"
        )

    print(
        "gpu average_batch_size:",
        gpu.get(
            "average_batch_size"
        ),
    )

    print(
        "gpu max_observed_batch_size:",
        gpu.get(
            "max_observed_batch_size"
        ),
    )


if __name__ == "__main__":
    mp.freeze_support()
    main()
