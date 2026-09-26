import argparse
import json
import os
import pickle
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from splendor_v1.env.env import SplendorEnv
from splendor_v1.network.model_4_legal_scorer import SplendorNetwork
from splendor_v1.training_v2.replay_buffer import ReplayBuffer
from splendor_v1.training_v2.state_serializer import serialize_state
from splendor_v1.training_v5.model_replay_generator_v5_pruning import (
    ModelReplayGenerator,
)
from splendor_v1.training_v5.train_v5 import build_training_batch

import splendor_v1.mcts.mcts_v5_pruning as mcts_v5_module
from splendor_v1.mcts.mcts_v5_pruning import MCTS


DEFAULT_CHECKPOINT = (
    "splendor_v1/training_v5/data/"
    "model_1900_games.pt"
)

DEFAULT_REPLAY = (
    "splendor_v1/training_v5/data/"
    "replay_buffer_model4_mcts_v5_pruning.pkl"
)

DEFAULT_OUTPUT = (
    "splendor_v1/training_v5/data/"
    "profile_v5_performance.json"
)


def extract_model_state_dict(checkpoint):
    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        return checkpoint["model_state_dict"]

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
        "Checkpoint must be a raw state_dict or contain "
        "'model_state_dict'."
    )


def load_model(checkpoint_path, device):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint does not exist: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model = SplendorNetwork()
    model.load_state_dict(
        extract_model_state_dict(checkpoint),
        strict=True,
    )
    model.to(device)
    model.eval()

    return model


def load_rich_replay_buffer(replay_path):
    if not os.path.exists(replay_path):
        raise FileNotFoundError(
            f"Replay does not exist: {replay_path}"
        )

    with open(replay_path, "rb") as file:
        data = pickle.load(file)

    if isinstance(data, ReplayBuffer):
        return data

    if not isinstance(data, dict):
        raise RuntimeError(
            f"Unsupported replay type: {type(data).__name__}"
        )

    required = (
        "capacity",
        "buffer",
        "position",
        "games",
        "game_sample_counts",
    )

    missing = [
        field
        for field in required
        if field not in data
    ]

    if missing:
        raise RuntimeError(
            "Replay is missing required rich-replay fields: "
            f"{missing}"
        )

    metadata = data.get("metadata", {})

    try:
        replay_buffer = ReplayBuffer(
            capacity=int(data["capacity"]),
            metadata=metadata,
        )
    except TypeError:
        replay_buffer = ReplayBuffer(
            int(data["capacity"])
        )
        replay_buffer.metadata = metadata

    replay_buffer.buffer = data["buffer"]
    replay_buffer.position = int(data["position"])
    replay_buffer.games = data["games"]
    replay_buffer.game_sample_counts = (
        data["game_sample_counts"]
    )
    replay_buffer.next_game_id = int(
        data.get(
            "next_game_id",
            len(replay_buffer.games),
        )
    )
    replay_buffer.metadata = metadata

    return replay_buffer


def cuda_sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class TimingAccumulator:
    def __init__(self):
        self.total_seconds = defaultdict(float)
        self.calls = defaultdict(int)

    def add(self, name, elapsed):
        self.total_seconds[name] += float(elapsed)
        self.calls[name] += 1

    def summary(self):
        result = {}

        for name in sorted(self.total_seconds):
            total = self.total_seconds[name]
            calls = self.calls[name]

            result[name] = {
                "calls": int(calls),
                "total_seconds": float(total),
                "mean_ms": (
                    float(total / calls * 1000.0)
                    if calls > 0
                    else 0.0
                ),
            }

        return result


def wrap_bound_method(
    obj,
    method_name,
    timer,
    timer_name=None,
):
    original = getattr(
        obj,
        method_name,
    )

    name = (
        timer_name
        if timer_name is not None
        else method_name
    )

    def wrapped(*args, **kwargs):
        start = time.perf_counter()

        try:
            return original(
                *args,
                **kwargs,
            )
        finally:
            timer.add(
                name,
                time.perf_counter() - start,
            )

    setattr(
        obj,
        method_name,
        wrapped,
    )

    return original


def profile_self_play(
    model,
    num_games,
    seed,
    simulations,
    min_simulations,
    check_interval,
    target_visits_per_action,
    single_action_simulations,
    stability_checks,
):
    env = SplendorEnv()

    mcts = MCTS(
        simulations=simulations,
        rollout_type="neural",
        selection_type="puct",
        model=model,
        c_puct=3.0,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.25,
        adaptive_simulations=True,
        min_simulations=min_simulations,
        check_interval=check_interval,
        target_visits_per_action=(
            target_visits_per_action
        ),
        single_action_simulations=(
            single_action_simulations
        ),
        stability_checks=stability_checks,
    )

    scratch_replay = ReplayBuffer(
        capacity=100_000
    )

    generator = ModelReplayGenerator(
        env=env,
        mcts=mcts,
        replay_buffer=scratch_replay,
        state_serializer=serialize_state,
        temperature=1.0,
        temperature_fn=lambda turn: (
            1.0
            if turn < 20
            else (
                0.5
                if turn < 40
                else 0.0
            )
        ),
        add_root_noise=True,
        root_noise_fn=lambda turn: (
            turn < 40
        ),
        teacher_mode=False,
        action_space_version=1,
        max_game_steps=300,
    )

    timer = TimingAccumulator()

    for method_name in (
        "search",
        "select",
        "expand_all_with_priors",
        "backup",
        "materialize_state",
        "add_dirichlet_noise",
    ):
        wrap_bound_method(
            mcts,
            method_name,
            timer,
        )

    original_neural_evaluate = (
        mcts_v5_module.neural_evaluate
    )

    def timed_neural_evaluate(
        *args,
        **kwargs,
    ):
        start = time.perf_counter()

        try:
            return original_neural_evaluate(
                *args,
                **kwargs,
            )
        finally:
            timer.add(
                "neural_evaluate",
                time.perf_counter() - start,
            )

    mcts_v5_module.neural_evaluate = (
        timed_neural_evaluate
    )

    game_results = []

    try:
        for game_offset in range(
            num_games
        ):
            game_seed = (
                int(seed)
                + game_offset
            )

            start = time.perf_counter()

            result = generator.generate_game(
                seed=game_seed,
                split="train",
                model_generation=4,
                model_checkpoint="profile",
                extra_game_metadata={
                    "profile": True,
                    "game_index":
                        int(game_offset),
                    "trainer_version": 5,
                },
            )

            elapsed = (
                time.perf_counter()
                - start
            )

            game_results.append(
                {
                    "seed":
                        int(game_seed),

                    "seconds":
                        float(elapsed),

                    "positions":
                        int(
                            result[
                                "num_positions"
                            ]
                        ),
                }
            )

            print(
                f"Profile game "
                f"{game_offset + 1}/"
                f"{num_games}: "
                f"{elapsed:.2f}s, "
                f"{result['num_positions']} "
                f"positions"
            )

    finally:
        mcts_v5_module.neural_evaluate = (
            original_neural_evaluate
        )

    search_time = timer.total_seconds.get(
        "search",
        0.0,
    )

    neural_time = timer.total_seconds.get(
        "neural_evaluate",
        0.0,
    )

    total_game_time = sum(
        game["seconds"]
        for game in game_results
    )

    return {
        "num_games":
            int(num_games),

        "total_game_seconds":
            float(total_game_time),

        "average_game_seconds":
            float(
                total_game_time
                / max(num_games, 1)
            ),

        "average_positions_per_game":
            float(
                np.mean(
                    [
                        game["positions"]
                        for game
                        in game_results
                    ]
                )
            ),

        "search_seconds":
            float(search_time),

        "search_fraction_of_game_time":
            float(
                search_time
                / total_game_time
                if total_game_time > 0
                else 0.0
            ),

        "neural_seconds":
            float(neural_time),

        "neural_fraction_of_search_time":
            float(
                neural_time
                / search_time
                if search_time > 0
                else 0.0
            ),

        "neural_fraction_of_game_time":
            float(
                neural_time
                / total_game_time
                if total_game_time > 0
                else 0.0
            ),

        "neural_calls":
            int(
                timer.calls.get(
                    "neural_evaluate",
                    0,
                )
            ),

        "timers":
            timer.summary(),

        "games":
            game_results,
    }


def prepare_benchmark_batches(
    replay_buffer,
    batch_size,
    num_batches,
    device,
    split,
):
    prepared = []

    attempts = 0
    max_attempts = (
        num_batches * 5
    )

    while (
        len(prepared) < num_batches
        and attempts < max_attempts
    ):
        attempts += 1

        if split is None:
            raw_batch = replay_buffer.sample(
                batch_size
            )
        else:
            raw_batch = replay_buffer.sample(
                batch_size,
                split=split,
            )

        if not raw_batch:
            continue

        batch = build_training_batch(
            batch=raw_batch,
            replay_buffer=replay_buffer,
        )

        prepared.append(
            (
                batch[
                    "observations"
                ].to(
                    device,
                    non_blocking=False,
                ),
                batch[
                    "legal_action_ids"
                ].to(
                    device,
                    non_blocking=False,
                ),
                batch[
                    "legal_action_mask"
                ].to(
                    device,
                    non_blocking=False,
                ),
            )
        )

    if not prepared:
        raise RuntimeError(
            "Could not prepare any benchmark batches. "
            "Try --split none if the requested split is empty."
        )

    return prepared


def benchmark_model_batches(
    model,
    replay_buffer,
    device,
    batch_sizes,
    warmup,
    iterations,
    prepared_batches_per_size,
    split,
):
    results = []

    for batch_size in batch_sizes:
        prepared = prepare_benchmark_batches(
            replay_buffer=replay_buffer,
            batch_size=batch_size,
            num_batches=prepared_batches_per_size,
            device=device,
            split=split,
        )

        with torch.inference_mode():
            for i in range(warmup):
                observations, ids, mask = (
                    prepared[
                        i % len(prepared)
                    ]
                )

                model(
                    observations,
                    ids,
                    mask,
                )

        cuda_sync(device)

        batch_times = []

        with torch.inference_mode():
            for i in range(iterations):
                observations, ids, mask = (
                    prepared[
                        i % len(prepared)
                    ]
                )

                cuda_sync(device)
                start = time.perf_counter()

                model(
                    observations,
                    ids,
                    mask,
                )

                cuda_sync(device)

                batch_times.append(
                    time.perf_counter()
                    - start
                )

        mean_seconds = float(
            np.mean(
                batch_times
            )
        )

        median_seconds = float(
            np.median(
                batch_times
            )
        )

        effective_batch_size = int(
            prepared[0][0].shape[0]
        )

        positions_per_second = (
            effective_batch_size
            / mean_seconds
        )

        result = {
            "requested_batch_size":
                int(batch_size),

            "effective_batch_size":
                effective_batch_size,

            "mean_batch_ms":
                float(
                    mean_seconds
                    * 1000.0
                ),

            "median_batch_ms":
                float(
                    median_seconds
                    * 1000.0
                ),

            "mean_ms_per_position":
                float(
                    mean_seconds
                    * 1000.0
                    / effective_batch_size
                ),

            "positions_per_second":
                float(
                    positions_per_second
                ),
        }

        results.append(result)

        print(
            f"Batch {batch_size:>2}: "
            f"{result['mean_batch_ms']:.3f} "
            f"ms/batch | "
            f"{result['mean_ms_per_position']:.3f} "
            f"ms/position | "
            f"{result['positions_per_second']:.1f} "
            f"positions/sec"
        )

    if results:
        baseline = (
            results[0][
                "positions_per_second"
            ]
        )

        for result in results:
            result["throughput_speedup_vs_first"] = (
                result[
                    "positions_per_second"
                ]
                / baseline
                if baseline > 0.0
                else 0.0
            )

    return results


def print_self_play_summary(summary):
    print()
    print("=" * 72)
    print("V5 SELF-PLAY PROFILE")
    print("=" * 72)

    print(
        "Average game time:",
        f"{summary['average_game_seconds']:.2f}s",
    )

    print(
        "Average positions/game:",
        f"{summary['average_positions_per_game']:.2f}",
    )

    print(
        "MCTS search fraction of game:",
        f"{100.0 * summary['search_fraction_of_game_time']:.2f}%",
    )

    print(
        "Neural fraction of MCTS search:",
        f"{100.0 * summary['neural_fraction_of_search_time']:.2f}%",
    )

    print(
        "Neural fraction of whole game:",
        f"{100.0 * summary['neural_fraction_of_game_time']:.2f}%",
    )

    print(
        "Neural evaluation calls:",
        f"{summary['neural_calls']:,}",
    )

    print()
    print("Inclusive timers:")

    for name, stats in (
        summary["timers"].items()
    ):
        print(
            f"  {name:<24} "
            f"{stats['total_seconds']:>10.3f}s "
            f"| calls={stats['calls']:>8,} "
            f"| mean={stats['mean_ms']:.4f} ms"
        )

    print()
    print(
        "NOTE: timers are nested/inclusive. "
        "search includes selection/expansion/neural/backup, "
        "and expand_all_with_priors includes neural_evaluate."
    )


def print_batch_summary(results):
    if not results:
        return

    print()
    print("=" * 72)
    print("MODEL 4 GPU BATCH BENCHMARK")
    print("=" * 72)

    print(
        f"{'Batch':>6} "
        f"{'ms/batch':>12} "
        f"{'ms/pos':>12} "
        f"{'pos/sec':>12} "
        f"{'speedup':>10}"
    )

    for result in results:
        print(
            f"{result['effective_batch_size']:>6} "
            f"{result['mean_batch_ms']:>12.3f} "
            f"{result['mean_ms_per_position']:>12.3f} "
            f"{result['positions_per_second']:>12.1f} "
            f"{result['throughput_speedup_vs_first']:>9.2f}x"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Profile Model 4 + MCTS V5 and benchmark "
            "Model 4 inference throughput by batch size."
        )
    )

    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
    )

    parser.add_argument(
        "--replay",
        default=DEFAULT_REPLAY,
    )

    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
    )

    parser.add_argument(
        "--profile-games",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--profile-seed",
        type=int,
        default=20000,
    )

    parser.add_argument(
        "--simulations",
        type=int,
        default=400,
    )

    parser.add_argument(
        "--min-simulations",
        type=int,
        default=80,
    )

    parser.add_argument(
        "--check-interval",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--target-visits-per-action",
        type=float,
        default=20.0,
    )

    parser.add_argument(
        "--single-action-simulations",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--stability-checks",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[
            1,
            2,
            4,
            8,
            16,
            32,
        ],
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--benchmark-iterations",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--prepared-batches-per-size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--split",
        choices=[
            "train",
            "val",
            "none",
        ],
        default="train",
    )

    parser.add_argument(
        "--skip-game-profile",
        action="store_true",
    )

    parser.add_argument(
        "--skip-batch-benchmark",
        action="store_true",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)
    print(
        "Checkpoint:",
        args.checkpoint,
    )

    model = load_model(
        args.checkpoint,
        device,
    )

    report = {
        "device":
            str(device),

        "checkpoint":
            args.checkpoint,

        "replay":
            args.replay,

        "v5_config": {
            "simulations":
                int(args.simulations),

            "min_simulations":
                int(
                    args.min_simulations
                ),

            "check_interval":
                int(
                    args.check_interval
                ),

            "target_visits_per_action":
                float(
                    args.target_visits_per_action
                ),

            "single_action_simulations":
                int(
                    args.single_action_simulations
                ),

            "stability_checks":
                int(
                    args.stability_checks
                ),
        },
    }

    if not args.skip_game_profile:
        self_play_summary = (
            profile_self_play(
                model=model,
                num_games=max(
                    1,
                    int(
                        args.profile_games
                    ),
                ),
                seed=args.profile_seed,
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
            )
        )

        report[
            "self_play_profile"
        ] = self_play_summary

        print_self_play_summary(
            self_play_summary
        )

    if not args.skip_batch_benchmark:
        replay_buffer = (
            load_rich_replay_buffer(
                args.replay
            )
        )

        print()
        print(
            "Replay positions:",
            f"{len(replay_buffer):,}",
        )

        split = (
            None
            if args.split == "none"
            else args.split
        )

        batch_results = (
            benchmark_model_batches(
                model=model,
                replay_buffer=replay_buffer,
                device=device,
                batch_sizes=(
                    args.batch_sizes
                ),
                warmup=max(
                    0,
                    args.warmup,
                ),
                iterations=max(
                    1,
                    args.benchmark_iterations,
                ),
                prepared_batches_per_size=max(
                    1,
                    args.prepared_batches_per_size,
                ),
                split=split,
            )
        )

        report[
            "batch_benchmark"
        ] = batch_results

        print_batch_summary(
            batch_results
        )

    output_path = Path(
        args.output
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
        )

    print()
    print(
        "Profile saved to:",
        output_path,
    )


if __name__ == "__main__":
    main()
