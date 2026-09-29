"""
Parallel Model 4 vs fixed-baseline evaluator for Splendor V6.

Supported baselines
-------------------
    random  - uniformly random legal action
    h3      - HeuristicAgent3
    h12     - HeuristicAgent12
    h16     - HeuristicAgent16

Evaluation rules
----------------
- The neural checkpoint is loaded once by the existing centralized
  GPUInferenceServerProcess and forced into eval() mode by that loader.
- CPU game workers own environments, MCTS state, and baseline agents.
- No training occurs and no replay samples are written.
- Neural MCTS uses the existing V5 adaptive search settings.
- Dirichlet root noise is disabled.
- Neural action selection is greedy from root visit counts (temperature 0).
- Seats are paired: each environment seed is used twice, once with the model
  as player 0 and once as player 1.
- Baseline agents are reconstructed for every game with deterministic seeds,
  preventing RNG/state leakage between games.
- MCTS root reuse is retained only while the neural player keeps control
  through a forced sub-decision. The tree is dropped when control changes.

Architecture
------------
Main process
    -> one GPU inference server (model checkpoint)
    -> N CPU evaluation workers
         -> neural turns: MCTS -> MultiprocessNeuralEvaluator -> GPU server
         -> baseline turns: baseline.select_action(env, state) on CPU

Examples
--------
    python -m splendor_v1.evaluation.run_parallel_baseline_evaluation_v6 \
        --model splendor_v1/training_v6/data/model_4000_games.pt \
        --opponent h16 \
        --games 500 \
        --workers 12

    python -m splendor_v1.evaluation.run_parallel_baseline_evaluation_v6 \
        --model splendor_v1/training_v6/data/model_4000_games.pt \
        --opponent random \
        --games 200 \
        --workers 24
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass, field
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
from queue import Empty
import random
import statistics
import time
import traceback
from typing import Any, Optional

import numpy as np

from splendor_v1.env.env import SplendorEnv
from splendor_v1.mcts_batched.gpu_inference_server import (
    GPUInferenceServerConfig,
    GPUInferenceServerProcess,
    create_inference_ipc,
)
from splendor_v1.mcts_batched.mcts_v5_direct import MCTS
from splendor_v1.mcts_batched.multiprocess_neural_evaluator import (
    MultiprocessNeuralEvaluator,
)


# ============================================================================
# DEFAULTS
# ============================================================================

DEFAULT_NUM_GAMES = 500
DEFAULT_NUM_WORKERS = 12
DEFAULT_GPU_MAX_BATCH_SIZE = 12
DEFAULT_GPU_BATCH_WAIT_MS = 1.0
DEFAULT_BASE_SEED = 600_000
DEFAULT_DEVICE = "cuda"
DEFAULT_MAX_GAME_STEPS = 300

DEFAULT_SIMULATIONS = 400
DEFAULT_MIN_SIMULATIONS = 80
DEFAULT_CHECK_INTERVAL = 20
DEFAULT_TARGET_VISITS_PER_ACTION = 20.0
DEFAULT_SINGLE_ACTION_SIMULATIONS = 4
DEFAULT_STABILITY_CHECKS = 3
DEFAULT_C_PUCT = 3.0

DEFAULT_H12_ROLLOUTS = 8
DEFAULT_H16_ROLLOUTS = 8

DEFAULT_STARTUP_TIMEOUT_S = 180.0
DEFAULT_RESULT_STALL_TIMEOUT_S = 1800.0
DEFAULT_SHUTDOWN_TIMEOUT_S = 30.0
DEFAULT_INFERENCE_TIMEOUT_S = 180.0
DEFAULT_REQUEST_PUT_TIMEOUT_S = 30.0

SUPPORTED_OPPONENTS = ("random", "h3", "h12", "h16")


# ============================================================================
# CONFIG / IPC MESSAGES
# ============================================================================


@dataclass(slots=True)
class EvaluationSearchConfig:
    simulations: int = DEFAULT_SIMULATIONS
    min_simulations: int = DEFAULT_MIN_SIMULATIONS
    check_interval: int = DEFAULT_CHECK_INTERVAL
    target_visits_per_action: float = DEFAULT_TARGET_VISITS_PER_ACTION
    single_action_simulations: int = DEFAULT_SINGLE_ACTION_SIMULATIONS
    stability_checks: int = DEFAULT_STABILITY_CHECKS
    c_puct: float = DEFAULT_C_PUCT
    max_game_steps: int = DEFAULT_MAX_GAME_STEPS

    def validate(self) -> None:
        if self.simulations < 1:
            raise ValueError("simulations must be >= 1")
        if self.min_simulations < 1:
            raise ValueError("min_simulations must be >= 1")
        if self.check_interval < 1:
            raise ValueError("check_interval must be >= 1")
        if self.target_visits_per_action <= 0:
            raise ValueError("target_visits_per_action must be > 0")
        if self.single_action_simulations < 1:
            raise ValueError("single_action_simulations must be >= 1")
        if self.stability_checks < 1:
            raise ValueError("stability_checks must be >= 1")
        if self.max_game_steps < 1:
            raise ValueError("max_game_steps must be >= 1")


@dataclass(slots=True)
class EvaluationWorkerConfig:
    opponent: str
    search: EvaluationSearchConfig = field(default_factory=EvaluationSearchConfig)
    h12_rollouts: int = DEFAULT_H12_ROLLOUTS
    h16_rollouts: int = DEFAULT_H16_ROLLOUTS
    inference_timeout_s: float = DEFAULT_INFERENCE_TIMEOUT_S
    request_put_timeout_s: float = DEFAULT_REQUEST_PUT_TIMEOUT_S

    def validate(self) -> None:
        self.search.validate()
        if self.opponent not in SUPPORTED_OPPONENTS:
            raise ValueError(
                f"Unsupported opponent {self.opponent!r}; "
                f"choose from {SUPPORTED_OPPONENTS}"
            )
        if self.h12_rollouts < 1:
            raise ValueError("h12_rollouts must be >= 1")
        if self.h16_rollouts < 1:
            raise ValueError("h16_rollouts must be >= 1")
        if self.inference_timeout_s <= 0:
            raise ValueError("inference_timeout_s must be > 0")
        if self.request_put_timeout_s <= 0:
            raise ValueError("request_put_timeout_s must be > 0")


@dataclass(slots=True)
class EvaluationGameJob:
    game_id: int
    seed: int
    model_player: int


@dataclass(slots=True)
class EvaluationWorkerReady:
    worker_id: int
    pid: int


@dataclass(slots=True)
class EvaluationWorkerFatal:
    worker_id: int
    pid: int
    error_type: str
    error: str
    traceback: str


@dataclass(slots=True)
class EvaluationWorkerShutdown:
    reason: str = "shutdown"


@dataclass(slots=True)
class EvaluationGameResult:
    success: bool
    worker_id: int
    pid: int
    game_id: int
    seed: int
    model_player: int
    opponent: str
    seconds: float

    outcome: Optional[str] = None  # "model", "baseline", "draw"
    winner_ids: Optional[list[int]] = None
    final_scores: Optional[list[Optional[int]]] = None
    model_score: Optional[int] = None
    baseline_score: Optional[int] = None
    score_difference: Optional[int] = None
    positions: int = 0
    final_turn_number: Optional[int] = None

    model_searches: int = 0
    model_simulations: int = 0
    model_stop_reasons: Optional[dict[str, int]] = None

    baseline_actions: int = 0
    baseline_decision_seconds: float = 0.0

    evaluator_stats: Optional[dict[str, Any]] = None

    error_type: Optional[str] = None
    error: Optional[str] = None
    traceback: Optional[str] = None


# ============================================================================
# BASELINE AGENTS
# ============================================================================


class RandomBaselineAgent:
    """Uniform random legal-action agent with a private deterministic RNG."""

    def __init__(self, seed: int):
        self.rng = random.Random(int(seed))

    def select_action(self, env, state):
        legal_actions = list(env._legal_actions(state))
        if not legal_actions:
            return None
        return self.rng.choice(legal_actions)


def baseline_seed(game_seed: int, model_player: int) -> int:
    """Derive a stable baseline RNG seed independent of worker scheduling."""

    return int(
        (
            (int(game_seed) * 1_000_003)
            ^ (int(model_player) * 0x9E3779B1)
            ^ 0x5F3759DF
        )
        & 0x7FFFFFFF
    )


def build_baseline_agent(
    *,
    opponent: str,
    game_seed: int,
    model_player: int,
    h12_rollouts: int,
    h16_rollouts: int,
):
    seed = baseline_seed(game_seed, model_player)

    if opponent == "random":
        return RandomBaselineAgent(seed)

    if opponent == "h3":
        # Canonical path used by the current H12 implementation/tests.
        from splendor_v1.agents.heuristic_agent3 import HeuristicAgent3

        return HeuristicAgent3()

    if opponent == "h12":
        from splendor_v1.agents.heuristic_agent_12 import HeuristicAgent12

        return HeuristicAgent12(
            num_rollouts=int(h12_rollouts),
            random_seed=seed,
        )

    if opponent == "h16":
        # Canonical path used by heuristic_generate_training_set.py.
        from splendor_v1.agents.heuristic_agent_16.heuristic_agent_16 import (
            HeuristicAgent16,
        )

        return HeuristicAgent16(
            num_rollouts=int(h16_rollouts),
            random_seed=seed,
        )

    raise ValueError(f"Unsupported baseline opponent: {opponent!r}")


# ============================================================================
# MCTS / GAME HELPERS
# ============================================================================


def make_evaluation_mcts(
    *,
    evaluator: MultiprocessNeuralEvaluator,
    search: EvaluationSearchConfig,
) -> MCTS:
    search.validate()

    return MCTS(
        simulations=search.simulations,
        rollout_type="neural",
        selection_type="puct",
        evaluator=evaluator,
        c_puct=search.c_puct,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.25,
        adaptive_simulations=True,
        min_simulations=search.min_simulations,
        check_interval=search.check_interval,
        target_visits_per_action=search.target_visits_per_action,
        single_action_simulations=search.single_action_simulations,
        stability_checks=search.stability_checks,
    )


def greedy_action_from_root(root, fallback_action=None):
    children = list(getattr(root, "children", []))

    if not children:
        if fallback_action is not None:
            return fallback_action
        raise RuntimeError("MCTS root has no children")

    visits = np.asarray(
        [int(getattr(child, "visits", 0)) for child in children],
        dtype=np.int64,
    )

    if visits.sum() <= 0:
        if fallback_action is not None:
            return fallback_action
        return children[0].action

    return children[int(np.argmax(visits))].action


def find_selected_child(root, action):
    for child in getattr(root, "children", []):
        if child.action == action:
            return child
    return None


def apply_env_action(env: SplendorEnv, action) -> bool:
    result = env.step(action)

    if isinstance(result, tuple):
        if len(result) == 5:
            _, _, terminated, truncated, _ = result
            return bool(terminated or truncated)
        if len(result) == 4:
            _, _, terminated, _ = result
            return bool(terminated)

    check = getattr(env, "_check_terminated", None)
    if callable(check):
        return bool(check(env.state))

    return bool(getattr(env.state, "game_over", False))


def final_scores(state) -> list[Optional[int]]:
    scores: list[Optional[int]] = []
    for player in getattr(state, "players", []):
        points = getattr(player, "points", None)
        scores.append(None if points is None else int(points))
    return scores


def classify_outcome(*, winner_ids: list[int], model_player: int) -> str:
    baseline_player = 1 - int(model_player)

    model_won = int(model_player) in winner_ids
    baseline_won = baseline_player in winner_ids

    if model_won and not baseline_won:
        return "model"
    if baseline_won and not model_won:
        return "baseline"
    return "draw"


def search_metadata(mcts: MCTS) -> dict[str, Any]:
    data = getattr(mcts, "last_search_metadata", None)
    return data if isinstance(data, dict) else {}


# ============================================================================
# ONE MODEL-VS-BASELINE GAME
# ============================================================================


def run_evaluation_game(
    *,
    worker_id: int,
    evaluator: MultiprocessNeuralEvaluator,
    job: EvaluationGameJob,
    config: EvaluationWorkerConfig,
) -> EvaluationGameResult:
    config.validate()
    started = time.perf_counter()

    try:
        env = SplendorEnv()
        env.reset(seed=int(job.seed))

        model_player = int(job.model_player)
        if model_player not in (0, 1):
            raise ValueError("model_player must be 0 or 1")

        baseline = build_baseline_agent(
            opponent=config.opponent,
            game_seed=int(job.seed),
            model_player=model_player,
            h12_rollouts=config.h12_rollouts,
            h16_rollouts=config.h16_rollouts,
        )

        mcts = make_evaluation_mcts(
            evaluator=evaluator,
            search=config.search,
        )

        # Scheduling-independent deterministic MCTS RNG.
        seed_sequence = np.random.SeedSequence(
            [int(job.seed), int(model_player), 0xBADC0DE]
        )
        mcts.rng = np.random.default_rng(seed_sequence)

        terminated = False
        step_index = 0
        model_root = None

        model_searches = 0
        model_simulations = 0
        model_stop_reasons: Counter[str] = Counter()

        baseline_actions = 0
        baseline_decision_seconds = 0.0

        while not terminated:
            if step_index >= config.search.max_game_steps:
                raise RuntimeError(
                    "Evaluation exceeded max_game_steps: "
                    f"seed={job.seed}, step={step_index}, "
                    f"turn={getattr(env.state, 'turn_number', None)}, "
                    f"current_player={getattr(env.state, 'current_player', None)}, "
                    f"scores={final_scores(env.state)}"
                )

            state = env.state
            current_player = int(state.current_player)
            previous_player = current_player

            if current_player == model_player:
                search_result = mcts.search(
                    env,
                    state,
                    root=model_root,
                    return_root=True,
                    add_root_noise=False,
                    teacher_mode=False,
                )

                if not isinstance(search_result, tuple) or len(search_result) != 2:
                    raise RuntimeError(
                        "Expected MCTS.search(return_root=True) to return "
                        "(action, root)"
                    )

                fallback_action, root = search_result
                if root is None:
                    raise RuntimeError("MCTS returned root=None")

                action = greedy_action_from_root(
                    root,
                    fallback_action=fallback_action,
                )
                if action is None:
                    raise RuntimeError("MCTS selected action=None")

                legal_actions = list(env._legal_actions(state))
                if action not in legal_actions:
                    raise RuntimeError(
                        "MCTS selected an illegal action during baseline evaluation"
                    )

                selected_child = find_selected_child(root, action)

                metadata = search_metadata(mcts)
                model_searches += 1
                model_simulations += int(
                    metadata.get("actual_simulations", 0) or 0
                )
                reason = metadata.get("stop_reason")
                if reason is not None:
                    model_stop_reasons[str(reason)] += 1

            else:
                selected_child = None
                decision_started = time.perf_counter()
                action = baseline.select_action(env, state)
                baseline_decision_seconds += time.perf_counter() - decision_started
                baseline_actions += 1

                if action is None:
                    raise RuntimeError(
                        f"Baseline {config.opponent} selected action=None"
                    )

                legal_actions = list(env._legal_actions(state))
                if action not in legal_actions:
                    raise RuntimeError(
                        f"Baseline {config.opponent} selected an illegal action"
                    )

            terminated = apply_env_action(env, action)
            step_index += 1

            if terminated:
                model_root = None
                break

            next_player = int(env.state.current_player)

            # Only a neural move can provide a reusable child root, and only
            # when the same real player retains control through a sub-decision.
            if (
                previous_player == model_player
                and selected_child is not None
                and next_player == previous_player
            ):
                selected_child.parent = None
                model_root = selected_child
            else:
                model_root = None

        state = env.state
        winner_ids = [
            int(player_id)
            for player_id in (getattr(state, "winners", None) or [])
        ]
        scores = final_scores(state)
        baseline_player = 1 - model_player

        model_score = (
            int(scores[model_player])
            if len(scores) > model_player and scores[model_player] is not None
            else None
        )
        baseline_score = (
            int(scores[baseline_player])
            if len(scores) > baseline_player and scores[baseline_player] is not None
            else None
        )
        score_difference = (
            model_score - baseline_score
            if model_score is not None and baseline_score is not None
            else None
        )

        elapsed = time.perf_counter() - started

        return EvaluationGameResult(
            success=True,
            worker_id=int(worker_id),
            pid=os.getpid(),
            game_id=int(job.game_id),
            seed=int(job.seed),
            model_player=model_player,
            opponent=config.opponent,
            seconds=float(elapsed),
            outcome=classify_outcome(
                winner_ids=winner_ids,
                model_player=model_player,
            ),
            winner_ids=winner_ids,
            final_scores=scores,
            model_score=model_score,
            baseline_score=baseline_score,
            score_difference=score_difference,
            positions=int(step_index),
            final_turn_number=int(getattr(state, "turn_number", 0)),
            model_searches=int(model_searches),
            model_simulations=int(model_simulations),
            model_stop_reasons=dict(model_stop_reasons),
            baseline_actions=int(baseline_actions),
            baseline_decision_seconds=float(baseline_decision_seconds),
            evaluator_stats=evaluator.stats_snapshot(),
        )

    except BaseException as exc:
        elapsed = time.perf_counter() - started

        return EvaluationGameResult(
            success=False,
            worker_id=int(worker_id),
            pid=os.getpid(),
            game_id=int(job.game_id),
            seed=int(job.seed),
            model_player=int(job.model_player),
            opponent=config.opponent,
            seconds=float(elapsed),
            evaluator_stats=evaluator.stats_snapshot(),
            error_type=type(exc).__name__,
            error=str(exc),
            traceback=traceback.format_exc(),
        )


# ============================================================================
# WORKER PROCESS
# ============================================================================


def evaluation_worker_main(
    *,
    worker_id: int,
    job_queue,
    result_queue,
    inference_request_queue,
    inference_response_queue,
    config: EvaluationWorkerConfig,
) -> None:
    """Long-lived CPU worker. It never owns the neural model or CUDA."""

    try:
        config.validate()

        # Import/construct one baseline during startup so missing modules or
        # constructor incompatibilities fail before a long evaluation begins.
        _ = build_baseline_agent(
            opponent=config.opponent,
            game_seed=0,
            model_player=0,
            h12_rollouts=config.h12_rollouts,
            h16_rollouts=config.h16_rollouts,
        )

        evaluator = MultiprocessNeuralEvaluator(
            worker_id=int(worker_id),
            request_queue=inference_request_queue,
            response_queue=inference_response_queue,
            request_timeout_s=config.inference_timeout_s,
            request_put_timeout_s=config.request_put_timeout_s,
            return_policy_device="cpu",
        )

        result_queue.put(
            EvaluationWorkerReady(
                worker_id=int(worker_id),
                pid=os.getpid(),
            )
        )

    except BaseException as exc:
        result_queue.put(
            EvaluationWorkerFatal(
                worker_id=int(worker_id),
                pid=os.getpid(),
                error_type=type(exc).__name__,
                error=str(exc),
                traceback=traceback.format_exc(),
            )
        )
        return

    try:
        while True:
            job = job_queue.get()

            if isinstance(job, EvaluationWorkerShutdown):
                break
            if not isinstance(job, EvaluationGameJob):
                continue

            result_queue.put(
                run_evaluation_game(
                    worker_id=worker_id,
                    evaluator=evaluator,
                    job=job,
                    config=config,
                )
            )

    finally:
        evaluator.close()


# ============================================================================
# AGGREGATION
# ============================================================================


def safe_mean(values) -> Optional[float]:
    values = [float(v) for v in values if v is not None]
    if not values:
        return None
    return float(statistics.mean(values))


def merge_counters(dicts) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for item in dicts:
        if item:
            counter.update({str(k): int(v) for k, v in item.items()})
    return dict(counter)


def latest_worker_stats(
    results: list[EvaluationGameResult],
) -> dict[int, dict[str, Any]]:
    latest: dict[int, dict[str, Any]] = {}
    for result in results:
        if result.evaluator_stats:
            latest[int(result.worker_id)] = result.evaluator_stats
    return latest


def aggregate_client_stats(latest: dict[int, dict[str, Any]]) -> dict[str, Any]:
    if not latest:
        return {}

    submitted = sum(int(x.get("requests_submitted", 0)) for x in latest.values())
    completed = sum(int(x.get("responses_completed", 0)) for x in latest.values())
    failed = sum(int(x.get("responses_failed", 0)) for x in latest.values())

    weighted_wait = 0.0
    weighted_round_trip = 0.0
    weight = 0

    for stats in latest.values():
        n = int(stats.get("responses_completed", 0))
        weight += n
        weighted_wait += float(
            stats.get("average_response_wait_ms", 0.0) or 0.0
        ) * n
        weighted_round_trip += float(
            stats.get("average_round_trip_ms", 0.0) or 0.0
        ) * n

    return {
        "requests_submitted": submitted,
        "responses_completed": completed,
        "responses_failed": failed,
        "average_response_wait_ms": weighted_wait / weight if weight else None,
        "average_round_trip_ms": weighted_round_trip / weight if weight else None,
    }


def normal_score_interval(
    wins: int,
    draws: int,
    games: int,
    z: float = 1.96,
) -> Optional[tuple[float, float, float]]:
    if games <= 0:
        return None

    score = (wins + 0.5 * draws) / games
    se = math.sqrt(max(0.0, score * (1.0 - score) / games))

    return (
        float(score),
        float(max(0.0, score - z * se)),
        float(min(1.0, score + z * se)),
    )


def seat_summary(results: list[EvaluationGameResult]) -> dict[str, Any]:
    games = len(results)
    wins = sum(r.outcome == "model" for r in results)
    losses = sum(r.outcome == "baseline" for r in results)
    draws = sum(r.outcome == "draw" for r in results)
    return {
        "games": games,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "match_score": (wins + 0.5 * draws) / games if games else None,
    }


def build_summary(
    *,
    successful: list[EvaluationGameResult],
    wall_seconds: float,
    gpu_server_stats: Optional[dict[str, Any]],
) -> dict[str, Any]:
    games = len(successful)
    model_wins = sum(r.outcome == "model" for r in successful)
    baseline_wins = sum(r.outcome == "baseline" for r in successful)
    draws = sum(r.outcome == "draw" for r in successful)

    model_as_p0 = [r for r in successful if r.model_player == 0]
    model_as_p1 = [r for r in successful if r.model_player == 1]

    interval = normal_score_interval(model_wins, draws, games)

    searches = sum(r.model_searches for r in successful)
    simulations = sum(r.model_simulations for r in successful)
    baseline_actions = sum(r.baseline_actions for r in successful)
    baseline_seconds = sum(r.baseline_decision_seconds for r in successful)

    return {
        "games": games,
        "opponent": successful[0].opponent if successful else None,
        "model_wins": model_wins,
        "baseline_wins": baseline_wins,
        "draws": draws,
        "model_win_rate": model_wins / games if games else None,
        "baseline_win_rate": baseline_wins / games if games else None,
        "draw_rate": draws / games if games else None,
        "model_match_score": interval[0] if interval else None,
        "model_match_score_approx_95ci": (
            [interval[1], interval[2]] if interval else None
        ),
        "model_as_player0": seat_summary(model_as_p0),
        "model_as_player1": seat_summary(model_as_p1),
        "average_positions": safe_mean(r.positions for r in successful),
        "average_final_turn_number": safe_mean(
            r.final_turn_number for r in successful
        ),
        "average_model_score": safe_mean(r.model_score for r in successful),
        "average_baseline_score": safe_mean(
            r.baseline_score for r in successful
        ),
        "average_score_difference": safe_mean(
            r.score_difference for r in successful
        ),
        "model_searches": searches,
        "model_average_simulations_per_search": (
            simulations / searches if searches else None
        ),
        "model_stop_reasons": merge_counters(
            r.model_stop_reasons for r in successful
        ),
        "baseline_actions": baseline_actions,
        "baseline_total_decision_seconds": float(baseline_seconds),
        "baseline_average_decision_ms": (
            baseline_seconds / baseline_actions * 1000.0
            if baseline_actions
            else None
        ),
        "wall_seconds": float(wall_seconds),
        "games_per_hour": (
            games / wall_seconds * 3600.0 if wall_seconds > 0 else None
        ),
        "gpu_server": gpu_server_stats,
        "model_inference_client": aggregate_client_stats(
            latest_worker_stats(successful)
        ),
    }


# ============================================================================
# SERVER / PROCESS LIFECYCLE
# ============================================================================


def drain_latest_stats(
    server,
    latest: Optional[dict[str, Any]],
) -> Optional[dict[str, Any]]:
    for event in server.drain_status_events():
        stats = getattr(event, "stats", None)
        if isinstance(stats, dict):
            latest = stats
        if getattr(event, "kind", None) == "fatal_error":
            raise RuntimeError(
                "GPU inference server fatal error:\n"
                + str(getattr(event, "message", ""))
            )
    return latest


def stop_server_and_get_stats(
    server: GPUInferenceServerProcess,
    latest: Optional[dict[str, Any]],
    timeout_s: float,
) -> Optional[dict[str, Any]]:
    if server is None:
        return latest

    try:
        latest = drain_latest_stats(server, latest)
    except Exception:
        pass

    server.close(
        timeout_s=timeout_s,
        terminate_if_needed=True,
    )

    final = server.final_stats
    return final if isinstance(final, dict) else latest


def wait_for_workers_ready(
    *,
    processes,
    result_queue,
    num_workers: int,
    timeout_s: float,
) -> None:
    deadline = time.perf_counter() + timeout_s
    ready: set[int] = set()

    while len(ready) < num_workers:
        dead = [
            (index, process.exitcode)
            for index, process in enumerate(processes)
            if process.exitcode is not None and not process.is_alive()
        ]
        if dead:
            raise RuntimeError(
                "Evaluation worker exited while waiting for READY: "
                f"{dead}"
            )

        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            raise TimeoutError("Timed out waiting for evaluation workers READY")

        try:
            message = result_queue.get(timeout=min(0.25, remaining))
        except Empty:
            continue

        if isinstance(message, EvaluationWorkerReady):
            ready.add(int(message.worker_id))
            continue

        if isinstance(message, EvaluationWorkerFatal):
            raise RuntimeError(
                f"Evaluation worker {message.worker_id} failed during startup: "
                f"{message.error_type}: {message.error}\n{message.traceback}"
            )

        raise RuntimeError(
            "Unexpected message before all workers were ready: "
            f"{type(message).__name__}"
        )


# ============================================================================
# MAIN EVALUATION
# ============================================================================


def run_parallel_evaluation(
    args,
) -> tuple[dict[str, Any], list[EvaluationGameResult]]:
    model = Path(args.model)

    if not model.exists():
        raise FileNotFoundError(f"Model checkpoint does not exist: {model}")
    if args.opponent not in SUPPORTED_OPPONENTS:
        raise ValueError(
            f"Unsupported opponent {args.opponent!r}; "
            f"choose from {SUPPORTED_OPPONENTS}"
        )
    if args.games < 1:
        raise ValueError("games must be >= 1")
    if args.workers < 1:
        raise ValueError("workers must be >= 1")
    if args.gpu_max_batch_size < 1:
        raise ValueError("gpu-max-batch-size must be >= 1")

    context = mp.get_context("spawn")

    ipc = create_inference_ipc(
        num_workers=args.workers,
        mp_context=context,
    )

    server = GPUInferenceServerProcess(
        config=GPUInferenceServerConfig(
            checkpoint_path=str(model),
            max_batch_size=min(args.gpu_max_batch_size, args.workers),
            batch_wait_ms=args.gpu_batch_wait_ms,
            device=args.device,
            stats_report_interval_batches=250,
        ),
        ipc=ipc,
        mp_context=context,
        process_name="SplendorBaselineEvalGPU",
    )

    worker_config = EvaluationWorkerConfig(
        opponent=args.opponent,
        search=EvaluationSearchConfig(
            simulations=args.simulations,
            min_simulations=args.min_simulations,
            check_interval=args.check_interval,
            target_visits_per_action=args.target_visits_per_action,
            single_action_simulations=args.single_action_simulations,
            stability_checks=args.stability_checks,
            c_puct=args.c_puct,
            max_game_steps=args.max_game_steps,
        ),
        h12_rollouts=args.h12_rollouts,
        h16_rollouts=args.h16_rollouts,
        inference_timeout_s=args.inference_timeout_s,
        request_put_timeout_s=args.request_put_timeout_s,
    )
    worker_config.validate()

    job_queue = context.Queue()
    result_queue = context.Queue()
    processes = []

    latest_server_stats = None
    server_stats = None
    results: list[EvaluationGameResult] = []

    server.start()

    try:
        ready = server.wait_until_ready(timeout_s=args.startup_timeout_s)
        print(f"GPU server READY (pid={ready.pid})")
        print("  checkpoint:", model)
        print("  model mode: eval() enforced by GPUInferenceServerProcess loader")

        for worker_id in range(args.workers):
            process = context.Process(
                target=evaluation_worker_main,
                kwargs={
                    "worker_id": worker_id,
                    "job_queue": job_queue,
                    "result_queue": result_queue,
                    "inference_request_queue": ipc.request_queue,
                    "inference_response_queue": ipc.response_queues[worker_id],
                    "config": worker_config,
                },
                name=f"SplendorBaselineEvalWorker-{worker_id}",
                daemon=False,
            )
            process.start()
            processes.append(process)

        wait_for_workers_ready(
            processes=processes,
            result_queue=result_queue,
            num_workers=args.workers,
            timeout_s=args.startup_timeout_s,
        )

        print(f"All {args.workers} evaluation workers READY")
        print()

        if args.games % 2:
            print(
                "WARNING: --games is odd, so the final seed is not seat-paired. "
                "Use an even game count for a fully paired evaluation."
            )
            print()

        # Paired-seat design:
        #   game 0: seed S,   model=P0
        #   game 1: seed S,   model=P1
        #   game 2: seed S+1, model=P0
        #   game 3: seed S+1, model=P1
        for game_id in range(args.games):
            pair_index = game_id // 2
            model_player = 0 if game_id % 2 == 0 else 1
            job_queue.put(
                EvaluationGameJob(
                    game_id=game_id,
                    seed=int(args.base_seed + pair_index),
                    model_player=model_player,
                )
            )

        started = time.perf_counter()
        last_result_time = started
        progress_every = max(1, args.games // 20)

        while len(results) < args.games:
            latest_server_stats = drain_latest_stats(
                server,
                latest_server_stats,
            )

            dead = [
                (index, process.exitcode)
                for index, process in enumerate(processes)
                if process.exitcode is not None and not process.is_alive()
            ]
            if dead:
                raise RuntimeError(
                    "One or more evaluation workers exited before completion: "
                    f"{dead}"
                )

            try:
                message = result_queue.get(timeout=0.5)
            except Empty:
                if (
                    time.perf_counter() - last_result_time
                    > args.result_stall_timeout_s
                ):
                    raise TimeoutError(
                        "No evaluation game completed within "
                        f"{args.result_stall_timeout_s:.1f}s"
                    )
                continue

            if isinstance(message, EvaluationWorkerFatal):
                raise RuntimeError(
                    f"Evaluation worker {message.worker_id} fatal error: "
                    f"{message.error_type}: {message.error}\n{message.traceback}"
                )

            if not isinstance(message, EvaluationGameResult):
                continue

            last_result_time = time.perf_counter()

            if not message.success:
                raise RuntimeError(
                    f"Evaluation game {message.game_id} failed on worker "
                    f"{message.worker_id}: {message.error_type}: "
                    f"{message.error}\n{message.traceback}"
                )

            results.append(message)

            if (
                len(results) == 1
                or len(results) % progress_every == 0
                or len(results) == args.games
            ):
                elapsed = time.perf_counter() - started
                model_wins = sum(r.outcome == "model" for r in results)
                baseline_wins = sum(r.outcome == "baseline" for r in results)
                draws = sum(r.outcome == "draw" for r in results)
                gph = (
                    len(results) / elapsed * 3600.0
                    if elapsed > 0
                    else 0.0
                )
                print(
                    f"Evaluation {len(results)}/{args.games} "
                    f"- Model {model_wins} / {args.opponent} {baseline_wins} "
                    f"/ D {draws} - {gph:.2f} games/hour"
                )

        wall_seconds = time.perf_counter() - started

        for _ in processes:
            job_queue.put(EvaluationWorkerShutdown())

        for process in processes:
            process.join(timeout=args.shutdown_timeout_s)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5.0)

        server_stats = stop_server_and_get_stats(
            server,
            latest_server_stats,
            args.shutdown_timeout_s,
        )

        results.sort(key=lambda r: r.game_id)

        summary = build_summary(
            successful=results,
            wall_seconds=wall_seconds,
            gpu_server_stats=server_stats,
        )

        return summary, results

    finally:
        for _ in processes:
            try:
                job_queue.put_nowait(EvaluationWorkerShutdown(reason="cleanup"))
            except Exception:
                pass

        for process in processes:
            if process.is_alive():
                process.join(timeout=1.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2.0)

        if server.is_alive():
            try:
                server.close(
                    timeout_s=args.shutdown_timeout_s,
                    terminate_if_needed=True,
                )
            except Exception:
                pass


# ============================================================================
# OUTPUT
# ============================================================================


def percent(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{100.0 * value:.2f}%"


def print_final_summary(args, summary: dict[str, Any]) -> None:
    print()
    print("=" * 78)
    print("PARALLEL MODEL 4 VS BASELINE EVALUATION COMPLETE")
    print("=" * 78)
    print("Model:", args.model)
    print("Opponent:", args.opponent)
    print("Games:", summary["games"])
    print()
    print(
        "Model wins:",
        summary["model_wins"],
        f"({percent(summary['model_win_rate'])})",
    )
    print(
        f"{args.opponent} wins:",
        summary["baseline_wins"],
        f"({percent(summary['baseline_win_rate'])})",
    )
    print("Draws:", summary["draws"], f"({percent(summary['draw_rate'])})")
    print(
        "Model match score (win=1, draw=.5):",
        percent(summary["model_match_score"]),
    )

    ci = summary.get("model_match_score_approx_95ci")
    if ci:
        print(
            "Approx model score 95% interval:",
            f"[{percent(ci[0])}, {percent(ci[1])}]",
        )

    print()
    print("Model as Player 0:", summary["model_as_player0"])
    print("Model as Player 1:", summary["model_as_player1"])
    print("Average positions:", summary["average_positions"])
    print("Average model score:", summary["average_model_score"])
    print("Average baseline score:", summary["average_baseline_score"])
    print("Average score difference:", summary["average_score_difference"])
    print()
    print(
        "Model average simulations/search:",
        summary["model_average_simulations_per_search"],
    )
    print("Model MCTS stop reasons:", summary["model_stop_reasons"])
    print(
        "Baseline average decision ms:",
        summary["baseline_average_decision_ms"],
    )
    print()
    print("Wall time:", f"{summary['wall_seconds']:.2f}s")
    print("Throughput:", f"{summary['games_per_hour']:.2f} games/hour")

    stats = summary.get("gpu_server") or {}
    print()
    print("GPU average batch size:", stats.get("average_batch_size"))
    print("GPU max observed batch size:", stats.get("max_observed_batch_size"))
    print(
        "GPU inference positions/sec:",
        stats.get("inference_positions_per_second"),
    )
    print("=" * 78)


def save_results_json(
    *,
    path: str,
    args,
    summary: dict[str, Any],
    results: list[EvaluationGameResult],
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model": str(args.model),
        "opponent": args.opponent,
        "config": {
            "games": args.games,
            "workers": args.workers,
            "gpu_max_batch_size": min(args.gpu_max_batch_size, args.workers),
            "gpu_batch_wait_ms": args.gpu_batch_wait_ms,
            "base_seed": args.base_seed,
            "simulations": args.simulations,
            "min_simulations": args.min_simulations,
            "check_interval": args.check_interval,
            "target_visits_per_action": args.target_visits_per_action,
            "single_action_simulations": args.single_action_simulations,
            "stability_checks": args.stability_checks,
            "c_puct": args.c_puct,
            "h12_rollouts": args.h12_rollouts,
            "h16_rollouts": args.h16_rollouts,
            "temperature": 0.0,
            "root_noise": False,
            "paired_seats": True,
            "model_mode": "eval",
            "replay_writes": False,
        },
        "summary": summary,
        "games": [asdict(result) for result in results],
    }

    temp = output.with_suffix(output.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
    os.replace(temp, output)


# ============================================================================
# CLI
# ============================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Parallel Model 4 checkpoint-vs-baseline evaluator"
    )

    parser.add_argument("--model", required=True, help="Model 4 checkpoint")
    parser.add_argument(
        "--opponent",
        choices=SUPPORTED_OPPONENTS,
        required=True,
        help="Fixed CPU baseline opponent",
    )
    parser.add_argument("--games", type=int, default=DEFAULT_NUM_GAMES)
    parser.add_argument("--workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument(
        "--gpu-max-batch-size",
        type=int,
        default=DEFAULT_GPU_MAX_BATCH_SIZE,
    )
    parser.add_argument(
        "--gpu-batch-wait-ms",
        type=float,
        default=DEFAULT_GPU_BATCH_WAIT_MS,
    )
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)

    parser.add_argument("--simulations", type=int, default=DEFAULT_SIMULATIONS)
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
    parser.add_argument("--c-puct", type=float, default=DEFAULT_C_PUCT)
    parser.add_argument(
        "--max-game-steps",
        type=int,
        default=DEFAULT_MAX_GAME_STEPS,
    )

    parser.add_argument(
        "--h12-rollouts",
        type=int,
        default=DEFAULT_H12_ROLLOUTS,
        help="H12 rollout count when --opponent h12",
    )
    parser.add_argument(
        "--h16-rollouts",
        type=int,
        default=DEFAULT_H16_ROLLOUTS,
        help="H16 rollout count when --opponent h16",
    )

    parser.add_argument(
        "--startup-timeout-s",
        type=float,
        default=DEFAULT_STARTUP_TIMEOUT_S,
    )
    parser.add_argument(
        "--result-stall-timeout-s",
        type=float,
        default=DEFAULT_RESULT_STALL_TIMEOUT_S,
    )
    parser.add_argument(
        "--shutdown-timeout-s",
        type=float,
        default=DEFAULT_SHUTDOWN_TIMEOUT_S,
    )
    parser.add_argument(
        "--inference-timeout-s",
        type=float,
        default=DEFAULT_INFERENCE_TIMEOUT_S,
    )
    parser.add_argument(
        "--request-put-timeout-s",
        type=float,
        default=DEFAULT_REQUEST_PUT_TIMEOUT_S,
    )
    parser.add_argument(
        "--output-json",
        default=(
            "splendor_v1/evaluation/results/"
            "latest_parallel_baseline_eval.json"
        ),
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()

    print("=" * 78)
    print("SPLENDOR V6 PARALLEL MODEL VS BASELINE EVALUATOR")
    print("=" * 78)
    print("Model:", args.model)
    print("Opponent:", args.opponent)
    print("Games:", args.games)
    print("Workers:", args.workers)
    print("GPU max batch size:", min(args.gpu_max_batch_size, args.workers))
    print("GPU batch wait ms:", args.gpu_batch_wait_ms)
    print("MCTS simulations hard cap:", args.simulations)
    print("MCTS minimum simulations:", args.min_simulations)
    if args.opponent == "h12":
        print("H12 rollouts:", args.h12_rollouts)
    if args.opponent == "h16":
        print("H16 rollouts:", args.h16_rollouts)
    print("Temperature: 0.0")
    print("Dirichlet root noise: OFF")
    print("Seat pairing: ON")
    print("Model mode: eval()")
    print("Replay writes: OFF")
    print("=" * 78)
    print()

    summary, results = run_parallel_evaluation(args)

    print_final_summary(args, summary)

    save_results_json(
        path=args.output_json,
        args=args,
        summary=summary,
        results=results,
    )

    print("Results JSON saved:", args.output_json)


if __name__ == "__main__":
    mp.freeze_support()
    main()
