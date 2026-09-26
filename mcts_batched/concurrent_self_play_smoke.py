"""
Concurrent self-play batching benchmark.

Purpose
-------
Measure the real throughput benefit of cross-game neural batching.

The benchmark runs the SAME game seeds in two modes:

1. Sequential baseline
       DirectNeuralEvaluator
       one game at a time

2. Concurrent batched
       several independent games in parallel
       one shared BatchedNeuralEvaluator
       one shared Model 4 GPU inference worker

Each game keeps its own:
    - SplendorEnv
    - MCTS tree
    - RNG
    - replay buffer
    - ModelReplayGenerator

Only neural inference is shared in the concurrent batched mode.

This script does not train the model and does not modify an existing
replay buffer. Each worker uses a private scratch ReplayBuffer.
"""

from __future__ import annotations

import argparse
from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
)
import json
import math
from pathlib import Path
import statistics
import time
import traceback

import numpy as np
import torch

from splendor_v1.env.env import SplendorEnv

from splendor_v1.network.model_4_legal_scorer import (
    SplendorNetwork,
)

from splendor_v1.training_v2.replay_buffer import (
    ReplayBuffer,
)

from splendor_v1.training_v2.state_serializer import (
    serialize_state,
)

from splendor_v1.training_v5.model_replay_generator_v5_pruning import (
    ModelReplayGenerator,
)

from splendor_v1.mcts_batched.direct_neural_evaluator import (
    DirectNeuralEvaluator,
)

from splendor_v1.mcts_batched.batched_neural_evaluator import (
    BatchedNeuralEvaluator,
)

from splendor_v1.mcts_batched.mcts_v5_direct import (
    MCTS,
)


# ============================================================
# DEFAULTS
# ============================================================

DEFAULT_CHECKPOINT = (
    "splendor_v1/training_v5/data/"
    "model_1900_games.pt"
)

DEFAULT_OUTPUT = (
    "splendor_v1/training_v5/data/"
    "concurrent_batching_benchmark.json"
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
# SHARED SEARCH CONFIG
# ============================================================

def make_mcts(
    evaluator,
    simulations,
    min_simulations,
    check_interval,
    target_visits_per_action,
    single_action_simulations,
    stability_checks,
    seed,
):
    mcts = MCTS(
        simulations=simulations,
        rollout_type="neural",
        selection_type="puct",
        evaluator=evaluator,
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

    # Keep each game's MCTS RNG independent and deterministic.
    mcts.rng = np.random.default_rng(
        int(seed) + 500_000
    )

    return mcts


def make_generator(
    env,
    mcts,
    replay_buffer,
):
    return ModelReplayGenerator(
        env=env,
        mcts=mcts,
        replay_buffer=replay_buffer,
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


# ============================================================
# REPLAY SAMPLE TELEMETRY
# ============================================================

def replay_samples(
    replay_buffer,
):
    """
    Return currently stored non-None replay samples.

    The benchmark gives each game a fresh replay buffer, so we do not
    need to reconstruct circular-buffer chronology here.
    """

    buffer = getattr(
        replay_buffer,
        "buffer",
        [],
    )

    return [
        sample
        for sample in buffer
        if sample is not None
    ]


def summarize_search_samples(
    replay_buffer,
):
    samples = replay_samples(
        replay_buffer
    )

    actual_simulations = []
    max_simulations = []
    initial_root_visits = []
    final_root_visits = []
    legal_counts = []
    stop_reasons = {}

    for sample in samples:
        if not isinstance(
            sample,
            dict,
        ):
            continue

        actual = sample.get(
            "search_actual_simulations"
        )

        maximum = sample.get(
            "search_max_simulations"
        )

        initial = sample.get(
            "search_initial_root_visits"
        )

        final = sample.get(
            "search_final_root_visits"
        )

        legal = sample.get(
            "search_num_legal_actions"
        )

        reason = sample.get(
            "search_stop_reason"
        )

        if actual is not None:
            actual_simulations.append(
                float(actual)
            )

        if maximum is not None:
            max_simulations.append(
                float(maximum)
            )

        if initial is not None:
            initial_root_visits.append(
                float(initial)
            )

        if final is not None:
            final_root_visits.append(
                float(final)
            )

        if legal is not None:
            legal_counts.append(
                float(legal)
            )

        if reason is not None:
            reason = str(
                reason
            )

            stop_reasons[reason] = (
                stop_reasons.get(
                    reason,
                    0,
                )
                + 1
            )

    if actual_simulations:
        average_actual = statistics.mean(
            actual_simulations
        )

        median_actual = statistics.median(
            actual_simulations
        )
    else:
        average_actual = None
        median_actual = None

    savings_fraction = None

    if (
        actual_simulations
        and max_simulations
        and len(actual_simulations)
        == len(max_simulations)
    ):
        total_actual = sum(
            actual_simulations
        )

        total_max = sum(
            max_simulations
        )

        if total_max > 0:
            savings_fraction = (
                1.0
                - total_actual / total_max
            )

    return {
        "samples":
            int(
                len(samples)
            ),

        "search_samples_with_actual_simulations":
            int(
                len(actual_simulations)
            ),

        "average_actual_simulations":
            (
                float(average_actual)
                if average_actual is not None
                else None
            ),

        "median_actual_simulations":
            (
                float(median_actual)
                if median_actual is not None
                else None
            ),

        "simulation_savings_fraction":
            (
                float(savings_fraction)
                if savings_fraction is not None
                else None
            ),

        "average_initial_root_visits":
            (
                float(
                    statistics.mean(
                        initial_root_visits
                    )
                )
                if initial_root_visits
                else None
            ),

        "average_final_root_visits":
            (
                float(
                    statistics.mean(
                        final_root_visits
                    )
                )
                if final_root_visits
                else None
            ),

        "average_legal_actions":
            (
                float(
                    statistics.mean(
                        legal_counts
                    )
                )
                if legal_counts
                else None
            ),

        "stop_reasons":
            stop_reasons,
    }


# ============================================================
# ONE GAME
# ============================================================

def run_one_game(
    evaluator,
    seed,
    simulations,
    min_simulations,
    check_interval,
    target_visits_per_action,
    single_action_simulations,
    stability_checks,
    mode,
):
    """
    Run one complete self-play game.

    Safe for concurrent use as long as:
      - this game's env/MCTS/replay/generator are private
      - evaluator is thread-safe
    """

    env = SplendorEnv()

    mcts = make_mcts(
        evaluator=evaluator,
        simulations=simulations,
        min_simulations=min_simulations,
        check_interval=check_interval,
        target_visits_per_action=(
            target_visits_per_action
        ),
        single_action_simulations=(
            single_action_simulations
        ),
        stability_checks=stability_checks,
        seed=seed,
    )

    scratch_replay = ReplayBuffer(
        capacity=10_000
    )

    generator = make_generator(
        env=env,
        mcts=mcts,
        replay_buffer=scratch_replay,
    )

    start = time.perf_counter()

    try:
        result = generator.generate_game(
            seed=int(seed),
            split="train",
            model_generation=4,
            model_checkpoint=(
                f"benchmark_{mode}"
            ),
            extra_game_metadata={
                "benchmark":
                    True,

                "benchmark_mode":
                    mode,

                "trainer_version":
                    5,
            },
        )

        elapsed = (
            time.perf_counter()
            - start
        )

        search_summary = (
            summarize_search_samples(
                scratch_replay
            )
        )

        return {
            "success":
                True,

            "seed":
                int(seed),

            "seconds":
                float(elapsed),

            "num_positions":
                int(
                    result.get(
                        "num_positions",
                        len(
                            replay_samples(
                                scratch_replay
                            )
                        ),
                    )
                ),

            "search":
                search_summary,
        }

    except Exception as exc:
        elapsed = (
            time.perf_counter()
            - start
        )

        return {
            "success":
                False,

            "seed":
                int(seed),

            "seconds":
                float(elapsed),

            "error_type":
                type(exc).__name__,

            "error":
                str(exc),

            "traceback":
                traceback.format_exc(),
        }


# ============================================================
# MODE SUMMARIES
# ============================================================

def summarize_mode(
    mode,
    results,
    wall_seconds,
):
    successful = [
        result
        for result in results
        if result.get(
            "success",
            False,
        )
    ]

    failed = [
        result
        for result in results
        if not result.get(
            "success",
            False,
        )
    ]

    positions = [
        result["num_positions"]
        for result in successful
    ]

    individual_seconds = [
        result["seconds"]
        for result in successful
    ]

    all_actual_sims = []

    search_sample_count = 0

    savings_weighted_actual = 0.0
    savings_weighted_max = 0.0

    stop_reasons = {}

    for result in successful:
        search = result.get(
            "search",
            {},
        )

        count = (
            search.get(
                "search_samples_with_actual_simulations",
                0,
            )
            or 0
        )

        avg_actual = search.get(
            "average_actual_simulations"
        )

        savings = search.get(
            "simulation_savings_fraction"
        )

        if (
            count > 0
            and avg_actual is not None
        ):
            all_actual_sims.extend(
                [
                    float(
                        avg_actual
                    )
                ]
                * int(count)
            )

            search_sample_count += int(
                count
            )

            if savings is not None:
                actual_total = (
                    float(avg_actual)
                    * count
                )

                if savings < 1.0:
                    max_total = (
                        actual_total
                        / (
                            1.0
                            - savings
                        )
                    )

                    savings_weighted_actual += (
                        actual_total
                    )

                    savings_weighted_max += (
                        max_total
                    )

        for reason, n in (
            search.get(
                "stop_reasons",
                {}
            )
            .items()
        ):
            stop_reasons[reason] = (
                stop_reasons.get(
                    reason,
                    0,
                )
                + int(n)
            )

    combined_savings = None

    if savings_weighted_max > 0:
        combined_savings = (
            1.0
            - (
                savings_weighted_actual
                / savings_weighted_max
            )
        )

    games_per_hour = (
        len(successful)
        / wall_seconds
        * 3600.0
        if wall_seconds > 0
        else 0.0
    )

    return {
        "mode":
            mode,

        "wall_seconds":
            float(
                wall_seconds
            ),

        "successful_games":
            int(
                len(successful)
            ),

        "failed_games":
            int(
                len(failed)
            ),

        "games_per_hour":
            float(
                games_per_hour
            ),

        "average_individual_game_seconds":
            (
                float(
                    statistics.mean(
                        individual_seconds
                    )
                )
                if individual_seconds
                else None
            ),

        "average_positions_per_game":
            (
                float(
                    statistics.mean(
                        positions
                    )
                )
                if positions
                else None
            ),

        "average_actual_simulations":
            (
                float(
                    statistics.mean(
                        all_actual_sims
                    )
                )
                if all_actual_sims
                else None
            ),

        "simulation_savings_fraction":
            (
                float(
                    combined_savings
                )
                if combined_savings is not None
                else None
            ),

        "stop_reasons":
            stop_reasons,

        "results":
            results,
    }


# ============================================================
# GPU WARMUP
# ============================================================

def warmup_direct_evaluator(
    evaluator,
    iterations=5,
    seed=12345,
):
    env = SplendorEnv()

    env.reset(
        seed=seed
    )

    state = env.state

    legal_actions = (
        env._legal_actions(
            state
        )
    )

    for _ in range(
        max(
            0,
            int(iterations),
        )
    ):
        evaluator.evaluate(
            env=env,
            state=state,
            legal_actions=legal_actions,
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()


# ============================================================
# SEQUENTIAL DIRECT BASELINE
# ============================================================

def run_sequential_direct(
    model,
    seeds,
    args,
):
    print()
    print("=" * 78)
    print(
        "MODE A: SEQUENTIAL DIRECT BASELINE"
    )
    print("=" * 78)

    evaluator = (
        DirectNeuralEvaluator(
            model=model
        )
    )

    warmup_direct_evaluator(
        evaluator=evaluator,
        iterations=args.warmup_calls,
        seed=args.seed - 1,
    )

    results = []

    wall_start = (
        time.perf_counter()
    )

    for index, seed in enumerate(
        seeds,
        start=1,
    ):
        result = run_one_game(
            evaluator=evaluator,
            seed=seed,
            simulations=args.simulations,
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
            mode="sequential_direct",
        )

        results.append(
            result
        )

        if result["success"]:
            print(
                f"[Direct "
                f"{index:>2}/{len(seeds)}] "
                f"seed={seed} | "
                f"{result['seconds']:.2f}s | "
                f"positions="
                f"{result['num_positions']}"
            )

        else:
            print(
                f"[Direct "
                f"{index:>2}/{len(seeds)}] "
                f"seed={seed} | FAILED | "
                f"{result['error_type']}: "
                f"{result['error']}"
            )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    wall_seconds = (
        time.perf_counter()
        - wall_start
    )

    summary = summarize_mode(
        mode="sequential_direct",
        results=results,
        wall_seconds=wall_seconds,
    )

    print_mode_summary(
        summary
    )

    return summary


# ============================================================
# CONCURRENT BATCHED RUN
# ============================================================

def run_concurrent_batched(
    model,
    seeds,
    args,
):
    print()
    print("=" * 78)
    print(
        "MODE B: CONCURRENT SHARED BATCHED EVALUATOR"
    )
    print("=" * 78)

    max_workers = min(
        int(
            args.concurrent_games
        ),
        len(seeds),
    )

    evaluator = (
        BatchedNeuralEvaluator(
            model=model,
            max_batch_size=(
                args.max_batch_size
            ),
            batch_wait_ms=(
                args.batch_wait_ms
            ),
            request_timeout_s=(
                args.request_timeout_s
            ),
        )
    )

    # One tiny warmup call initializes CUDA kernels without polluting
    # benchmark telemetry.
    warmup_direct = (
        DirectNeuralEvaluator(
            model=model
        )
    )

    warmup_direct_evaluator(
        evaluator=warmup_direct,
        iterations=args.warmup_calls,
        seed=args.seed - 2,
    )

    evaluator.reset_stats()

    results = []

    wall_start = (
        time.perf_counter()
    )

    try:
        with ThreadPoolExecutor(
            max_workers=max_workers
        ) as executor:

            future_to_seed = {
                executor.submit(
                    run_one_game,
                    evaluator,
                    seed,
                    args.simulations,
                    args.min_simulations,
                    args.check_interval,
                    args.target_visits_per_action,
                    args.single_action_simulations,
                    args.stability_checks,
                    "concurrent_batched",
                ):
                seed
                for seed in seeds
            }

            completed = 0

            for future in as_completed(
                future_to_seed
            ):
                seed = future_to_seed[
                    future
                ]

                completed += 1

                try:
                    result = future.result()

                except Exception as exc:
                    result = {
                        "success":
                            False,

                        "seed":
                            int(seed),

                        "seconds":
                            0.0,

                        "error_type":
                            type(exc).__name__,

                        "error":
                            str(exc),

                        "traceback":
                            traceback.format_exc(),
                    }

                results.append(
                    result
                )

                if result["success"]:
                    print(
                        f"[Batched "
                        f"{completed:>2}/"
                        f"{len(seeds)}] "
                        f"seed={seed} | "
                        f"{result['seconds']:.2f}s | "
                        f"positions="
                        f"{result['num_positions']}"
                    )

                else:
                    print(
                        f"[Batched "
                        f"{completed:>2}/"
                        f"{len(seeds)}] "
                        f"seed={seed} | FAILED | "
                        f"{result['error_type']}: "
                        f"{result['error']}"
                    )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        wall_seconds = (
            time.perf_counter()
            - wall_start
        )

        batch_stats = (
            evaluator.stats_snapshot()
        )

    finally:
        evaluator.close()

    # Put results back into seed order for easier A/B inspection.
    results.sort(
        key=lambda result: result[
            "seed"
        ]
    )

    summary = summarize_mode(
        mode="concurrent_batched",
        results=results,
        wall_seconds=wall_seconds,
    )

    summary[
        "batch_stats"
    ] = batch_stats

    summary[
        "configured_concurrent_games"
    ] = int(
        args.concurrent_games
    )

    summary[
        "actual_worker_count"
    ] = int(
        max_workers
    )

    print_mode_summary(
        summary
    )

    print()
    print("Batch telemetry:")

    for key in (
        "requests_submitted",
        "requests_completed",
        "batches_completed",
        "average_batch_size",
        "max_observed_batch_size",
        "configured_max_batch_size",
        "batch_fill_fraction",
        "configured_batch_wait_ms",
        "average_collection_wait_ms",
        "average_batch_build_ms",
        "average_batch_inference_ms",
        "total_batch_inference_seconds",
        "inference_positions_per_second",
    ):
        print(
            f"  {key}: "
            f"{batch_stats[key]}"
        )

    return summary


# ============================================================
# REPORTING
# ============================================================

def percent_string(
    value,
):
    if value is None:
        return "n/a"

    return (
        f"{100.0 * value:.2f}%"
    )


def number_string(
    value,
    digits=2,
):
    if value is None:
        return "n/a"

    return f"{value:.{digits}f}"


def print_mode_summary(
    summary,
):
    print()
    print(
        "Wall time:",
        f"{summary['wall_seconds']:.2f}s",
    )

    print(
        "Successful games:",
        summary["successful_games"],
    )

    print(
        "Failed games:",
        summary["failed_games"],
    )

    print(
        "Games/hour:",
        f"{summary['games_per_hour']:.2f}",
    )

    print(
        "Average individual game time:",
        (
            number_string(
                summary[
                    "average_individual_game_seconds"
                ]
            )
            + "s"
        ),
    )

    print(
        "Average positions/game:",
        number_string(
            summary[
                "average_positions_per_game"
            ]
        ),
    )

    print(
        "Average V5 simulations/search:",
        number_string(
            summary[
                "average_actual_simulations"
            ]
        ),
    )

    print(
        "V5 simulation savings:",
        percent_string(
            summary[
                "simulation_savings_fraction"
            ]
        ),
    )

    print(
        "Stop reasons:",
        summary[
            "stop_reasons"
        ],
    )


def compare_seed_results(
    direct_summary,
    batched_summary,
):
    direct_by_seed = {
        result["seed"]:
            result
        for result
        in direct_summary["results"]
    }

    batched_by_seed = {
        result["seed"]:
            result
        for result
        in batched_summary["results"]
    }

    common = sorted(
        set(
            direct_by_seed
        )
        & set(
            batched_by_seed
        )
    )

    comparisons = []

    for seed in common:
        direct = direct_by_seed[
            seed
        ]

        batched = batched_by_seed[
            seed
        ]

        if not (
            direct.get(
                "success",
                False,
            )
            and batched.get(
                "success",
                False,
            )
        ):
            continue

        comparisons.append(
            {
                "seed":
                    int(seed),

                "direct_positions":
                    int(
                        direct[
                            "num_positions"
                        ]
                    ),

                "batched_positions":
                    int(
                        batched[
                            "num_positions"
                        ]
                    ),

                "same_position_count":
                    bool(
                        direct[
                            "num_positions"
                        ]
                        == batched[
                            "num_positions"
                        ]
                    ),

                "direct_seconds":
                    float(
                        direct[
                            "seconds"
                        ]
                    ),

                "batched_individual_seconds":
                    float(
                        batched[
                            "seconds"
                        ]
                    ),
            }
        )

    return comparisons


def print_final_comparison(
    direct,
    batched,
):
    print()
    print("=" * 78)
    print("A/B THROUGHPUT COMPARISON")
    print("=" * 78)

    if (
        direct["wall_seconds"] > 0
        and batched["wall_seconds"] > 0
    ):
        wall_speedup = (
            direct["wall_seconds"]
            / batched["wall_seconds"]
        )
    else:
        wall_speedup = 0.0

    if direct["games_per_hour"] > 0:
        throughput_speedup = (
            batched["games_per_hour"]
            / direct["games_per_hour"]
        )
    else:
        throughput_speedup = 0.0

    print(
        "Sequential direct wall time:",
        f"{direct['wall_seconds']:.2f}s",
    )

    print(
        "Concurrent batched wall time:",
        f"{batched['wall_seconds']:.2f}s",
    )

    print(
        "Wall-clock speedup:",
        f"{wall_speedup:.2f}x",
    )

    print()

    print(
        "Sequential direct games/hour:",
        f"{direct['games_per_hour']:.2f}",
    )

    print(
        "Concurrent batched games/hour:",
        f"{batched['games_per_hour']:.2f}",
    )

    print(
        "Throughput speedup:",
        f"{throughput_speedup:.2f}x",
    )

    batch_stats = batched.get(
        "batch_stats",
        {},
    )

    if batch_stats:
        print()
        print(
            "Average real batch size:",
            f"{batch_stats['average_batch_size']:.2f}",
        )

        print(
            "Max real batch size:",
            batch_stats[
                "max_observed_batch_size"
            ],
        )

        print(
            "Batch fill fraction:",
            percent_string(
                batch_stats[
                    "batch_fill_fraction"
                ]
            ),
        )

        print(
            "Batched inference throughput:",
            (
                f"{batch_stats['inference_positions_per_second']:.1f} "
                "positions/sec"
            ),
        )

    print()
    print(
        "NOTE: average individual game time in concurrent mode "
        "is not the primary metric. Several games overlap. "
        "Wall time and games/hour are the important metrics."
    )

    return {
        "wall_clock_speedup":
            float(
                wall_speedup
            ),

        "throughput_speedup":
            float(
                throughput_speedup
            ),
    }


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark sequential DirectNeuralEvaluator self-play "
            "against concurrent cross-game BatchedNeuralEvaluator "
            "self-play."
        )
    )

    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
    )

    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
    )

    parser.add_argument(
        "--games",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--concurrent-games",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--batch-wait-ms",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--request-timeout-s",
        type=float,
        default=120.0,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=40000,
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
        "--warmup-calls",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--skip-direct",
        action="store_true",
    )

    parser.add_argument(
        "--skip-batched",
        action="store_true",
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()

    if args.games < 1:
        raise ValueError(
            "--games must be >= 1."
        )

    if args.concurrent_games < 1:
        raise ValueError(
            "--concurrent-games must be >= 1."
        )

    if args.max_batch_size < 1:
        raise ValueError(
            "--max-batch-size must be >= 1."
        )

    if args.simulations < 1:
        raise ValueError(
            "--simulations must be >= 1."
        )

    if (
        args.skip_direct
        and args.skip_batched
    ):
        raise ValueError(
            "Cannot skip both benchmark modes."
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    seeds = [
        int(args.seed) + i
        for i in range(
            args.games
        )
    ]

    print("=" * 78)
    print(
        "CONCURRENT CROSS-GAME BATCHING BENCHMARK"
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
        "Seeds:",
        seeds,
    )

    print(
        "Games:",
        args.games,
    )

    print(
        "Concurrent games:",
        args.concurrent_games,
    )

    print(
        "Max NN batch size:",
        args.max_batch_size,
    )

    print(
        "Batch wait:",
        f"{args.batch_wait_ms} ms",
    )

    print(
        "MCTS hard max:",
        args.simulations,
    )

    print()
    print("Loading Model 4...")

    model = load_model(
        checkpoint_path=(
            args.checkpoint
        ),
        device=device,
    )

    print("Model loaded.")

    report = {
        "device":
            str(device),

        "checkpoint":
            args.checkpoint,

        "seeds":
            seeds,

        "config": {
            "games":
                int(
                    args.games
                ),

            "concurrent_games":
                int(
                    args.concurrent_games
                ),

            "max_batch_size":
                int(
                    args.max_batch_size
                ),

            "batch_wait_ms":
                float(
                    args.batch_wait_ms
                ),

            "simulations":
                int(
                    args.simulations
                ),

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

    direct_summary = None
    batched_summary = None

    if not args.skip_direct:
        direct_summary = (
            run_sequential_direct(
                model=model,
                seeds=seeds,
                args=args,
            )
        )

        report[
            "sequential_direct"
        ] = direct_summary

    if not args.skip_batched:
        batched_summary = (
            run_concurrent_batched(
                model=model,
                seeds=seeds,
                args=args,
            )
        )

        report[
            "concurrent_batched"
        ] = batched_summary

    if (
        direct_summary is not None
        and batched_summary is not None
    ):
        comparison = (
            print_final_comparison(
                direct=direct_summary,
                batched=batched_summary,
            )
        )

        comparison[
            "same_seed_results"
        ] = compare_seed_results(
            direct_summary,
            batched_summary,
        )

        report[
            "comparison"
        ] = comparison

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
        "Benchmark report saved to:",
        output_path,
    )

    print()
    print("=" * 78)
    print("BENCHMARK COMPLETE")
    print("=" * 78)


if __name__ == "__main__":

    try:
        main()

    except Exception as exc:

        print()
        print("=" * 78)
        print("BENCHMARK FAILED")
        print("=" * 78)

        print(
            f"{type(exc).__name__}: "
            f"{exc}"
        )

        raise
