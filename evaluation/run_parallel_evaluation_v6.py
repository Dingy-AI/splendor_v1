"""
Parallel Model 4 head-to-head evaluator for Splendor V6.

Purpose
-------
Evaluate two Model 4 checkpoints by playing many games in parallel while
reusing the existing V6 centralized GPU inference infrastructure.

Important evaluation rules
--------------------------
- Both checkpoints are loaded by GPUInferenceServerProcess.  The current V6
  GPU server calls model.eval() immediately after loading each checkpoint.
- No training occurs in this script.
- No replay samples are written.
- No Dirichlet root noise is used.
- Action selection is greedy from MCTS visit counts (temperature = 0).
- Seats are alternated in paired games.  A pair uses the same environment
  seed twice: once with Model A as player 0, once with Model A as player 1.
- Each agent uses its OWN MCTS and its OWN neural network for its move.  A
  Model A search never switches to Model B merely because a simulated leaf
  has a different current_player.
- Tree reuse is retained only while the same real player keeps control (for
  example forced discard / noble sub-decisions).  Trees are dropped when the
  real player changes, avoiding stale cross-agent tree state.

Architecture
------------
Main process
    -> GPU inference server A (checkpoint A, eval mode)
    -> GPU inference server B (checkpoint B, eval mode)
    -> N CPU game workers
         -> evaluator A -> server A
         -> evaluator B -> server B
         -> one complete Splendor game at a time

Run from the directory that CONTAINS splendor_v1, for example:

    python -m splendor_v1.evaluation.run_parallel_evaluation_v6 \
        --model-a splendor_v1/training_v6/data/model_10000_games.pt \
        --model-b splendor_v1/training_v6/data/model_5000_games.pt \
        --games 1000

If you place this file elsewhere, it can also be run directly as long as the
project root is on PYTHONPATH.
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

DEFAULT_NUM_GAMES = 1000
DEFAULT_NUM_WORKERS = 24
DEFAULT_GPU_MAX_BATCH_SIZE = 24
DEFAULT_GPU_BATCH_WAIT_MS = 1.0


# DEFAULT_BASE_SEED = 500_000
DEFAULT_BASE_SEED = 600_000
# DEFAULT_BASE_SEED = 700_000
# DEFAULT_BASE_SEED = 800_000
# DEFAULT_BASE_SEED = 900_000

DEFAULT_DEVICE = "cuda"
DEFAULT_MAX_GAME_STEPS = 300

DEFAULT_SIMULATIONS = 400
DEFAULT_MIN_SIMULATIONS = 80
DEFAULT_CHECK_INTERVAL = 20
DEFAULT_TARGET_VISITS_PER_ACTION = 20.0
DEFAULT_SINGLE_ACTION_SIMULATIONS = 4
DEFAULT_STABILITY_CHECKS = 3
DEFAULT_C_PUCT = 3.0

DEFAULT_STARTUP_TIMEOUT_S = 180.0
DEFAULT_RESULT_STALL_TIMEOUT_S = 1800.0
DEFAULT_SHUTDOWN_TIMEOUT_S = 30.0
DEFAULT_INFERENCE_TIMEOUT_S = 180.0
DEFAULT_REQUEST_PUT_TIMEOUT_S = 30.0


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
    search: EvaluationSearchConfig = field(
        default_factory=EvaluationSearchConfig
    )
    inference_timeout_s: float = DEFAULT_INFERENCE_TIMEOUT_S
    request_put_timeout_s: float = DEFAULT_REQUEST_PUT_TIMEOUT_S

    def validate(self) -> None:
        self.search.validate()
        if self.inference_timeout_s <= 0:
            raise ValueError("inference_timeout_s must be > 0")
        if self.request_put_timeout_s <= 0:
            raise ValueError("request_put_timeout_s must be > 0")


@dataclass(slots=True)
class EvaluationGameJob:
    game_id: int
    seed: int
    model_a_player: int


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
    model_a_player: int
    seconds: float

    outcome: Optional[str] = None  # "A", "B", "draw"
    winner_ids: Optional[list[int]] = None
    final_scores: Optional[list[Optional[int]]] = None
    model_a_score: Optional[int] = None
    model_b_score: Optional[int] = None
    positions: int = 0
    final_turn_number: Optional[int] = None

    model_a_searches: int = 0
    model_b_searches: int = 0
    model_a_simulations: int = 0
    model_b_simulations: int = 0
    model_a_stop_reasons: Optional[dict[str, int]] = None
    model_b_stop_reasons: Optional[dict[str, int]] = None

    evaluator_a_stats: Optional[dict[str, Any]] = None
    evaluator_b_stats: Optional[dict[str, Any]] = None

    error_type: Optional[str] = None
    error: Optional[str] = None
    traceback: Optional[str] = None


# ============================================================================
# MCTS / GAME HELPERS
# ============================================================================


def make_evaluation_mcts(
    *,
    evaluator: MultiprocessNeuralEvaluator,
    search: EvaluationSearchConfig,
) -> MCTS:
    """Build the same V5 adaptive PUCT search used by V6 self-play."""

    search.validate()

    return MCTS(
        simulations=search.simulations,
        rollout_type="neural",
        selection_type="puct",
        evaluator=evaluator,
        c_puct=search.c_puct,
        # Root noise is disabled in search() below.  These values therefore
        # never affect evaluation, but MCTS still expects valid parameters.
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
    """Temperature-0 action selection from root visit counts."""

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

    # np.argmax is deterministic: the first maximum wins a tie.
    return children[int(np.argmax(visits))].action


def find_selected_child(root, action):
    for child in getattr(root, "children", []):
        if child.action == action:
            return child
    return None


def apply_env_action(env: SplendorEnv, action) -> bool:
    """Apply one action and normalize Gym/Gymnasium termination formats."""

    result = env.step(action)

    if isinstance(result, tuple):
        if len(result) == 5:
            _, _, terminated, truncated, _ = result
            return bool(terminated or truncated)
        if len(result) == 4:
            _, _, terminated, _ = result
            return bool(terminated)

    # Last-resort compatibility with the project's environment API.
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


def classify_outcome(
    *,
    winner_ids: list[int],
    model_a_player: int,
) -> str:
    model_b_player = 1 - int(model_a_player)

    a_won = model_a_player in winner_ids
    b_won = model_b_player in winner_ids

    if a_won and not b_won:
        return "A"
    if b_won and not a_won:
        return "B"

    # Multiple winners (or an unusual terminal state with no unique winner)
    # are scored as a draw for head-to-head evaluation.
    return "draw"


def search_metadata(mcts: MCTS) -> dict[str, Any]:
    data = getattr(mcts, "last_search_metadata", None)
    return data if isinstance(data, dict) else {}


# ============================================================================
# ONE HEAD-TO-HEAD GAME
# ============================================================================


def run_evaluation_game(
    *,
    worker_id: int,
    evaluator_a: MultiprocessNeuralEvaluator,
    evaluator_b: MultiprocessNeuralEvaluator,
    job: EvaluationGameJob,
    config: EvaluationWorkerConfig,
) -> EvaluationGameResult:
    config.validate()

    started = time.perf_counter()

    try:
        env = SplendorEnv()
        env.reset(seed=int(job.seed))

        mcts_a = make_evaluation_mcts(
            evaluator=evaluator_a,
            search=config.search,
        )
        mcts_b = make_evaluation_mcts(
            evaluator=evaluator_b,
            search=config.search,
        )

        # Deterministic per-game MCTS RNG streams. Root noise is disabled,
        # but this also protects reproducibility if MCTS uses RNG elsewhere.
        seed_sequence = np.random.SeedSequence(int(job.seed))
        seed_a, seed_b = seed_sequence.spawn(2)
        mcts_a.rng = np.random.default_rng(seed_a)
        mcts_b.rng = np.random.default_rng(seed_b)

        model_a_player = int(job.model_a_player)
        if model_a_player not in (0, 1):
            raise ValueError("model_a_player must be 0 or 1")

        terminated = False
        step_index = 0

        # We only keep a root while the SAME real player retains control.
        # When the player changes, both agents start their next real move from
        # a clean root. This keeps the two agents' searches independent.
        root_a = None
        root_b = None

        searches = {"A": 0, "B": 0}
        simulations = {"A": 0, "B": 0}
        stop_reasons = {"A": Counter(), "B": Counter()}

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

            if current_player == model_a_player:
                label = "A"
                mcts = mcts_a
                current_root = root_a
            else:
                label = "B"
                mcts = mcts_b
                current_root = root_b

            search_result = mcts.search(
                env,
                state,
                root=current_root,
                return_root=True,
                # Evaluation must be deterministic / exploitative.
                add_root_noise=False,
                teacher_mode=False,
            )

            if (
                not isinstance(search_result, tuple)
                or len(search_result) != 2
            ):
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
                    "MCTS selected an illegal action during evaluation"
                )

            selected_child = find_selected_child(root, action)
            previous_player = current_player

            metadata = search_metadata(mcts)
            searches[label] += 1
            simulations[label] += int(metadata.get("actual_simulations", 0) or 0)
            reason = metadata.get("stop_reason")
            if reason is not None:
                stop_reasons[label][str(reason)] += 1

            terminated = apply_env_action(env, action)
            step_index += 1

            if terminated:
                root_a = None
                root_b = None
                break

            next_player = int(env.state.current_player)

            # Reuse only through forced sub-decisions owned by the same agent.
            if selected_child is not None and next_player == previous_player:
                selected_child.parent = None
                if label == "A":
                    root_a = selected_child
                    root_b = None
                else:
                    root_b = selected_child
                    root_a = None
            else:
                root_a = None
                root_b = None

        state = env.state
        winner_ids = [
            int(player_id)
            for player_id in (getattr(state, "winners", None) or [])
        ]
        scores = final_scores(state)
        model_b_player = 1 - model_a_player

        a_score = (
            int(scores[model_a_player])
            if len(scores) > model_a_player
            and scores[model_a_player] is not None
            else None
        )
        b_score = (
            int(scores[model_b_player])
            if len(scores) > model_b_player
            and scores[model_b_player] is not None
            else None
        )

        elapsed = time.perf_counter() - started

        return EvaluationGameResult(
            success=True,
            worker_id=int(worker_id),
            pid=os.getpid(),
            game_id=int(job.game_id),
            seed=int(job.seed),
            model_a_player=model_a_player,
            seconds=float(elapsed),
            outcome=classify_outcome(
                winner_ids=winner_ids,
                model_a_player=model_a_player,
            ),
            winner_ids=winner_ids,
            final_scores=scores,
            model_a_score=a_score,
            model_b_score=b_score,
            positions=int(step_index),
            final_turn_number=int(getattr(state, "turn_number", 0)),
            model_a_searches=int(searches["A"]),
            model_b_searches=int(searches["B"]),
            model_a_simulations=int(simulations["A"]),
            model_b_simulations=int(simulations["B"]),
            model_a_stop_reasons=dict(stop_reasons["A"]),
            model_b_stop_reasons=dict(stop_reasons["B"]),
            evaluator_a_stats=evaluator_a.stats_snapshot(),
            evaluator_b_stats=evaluator_b.stats_snapshot(),
        )

    except BaseException as exc:
        elapsed = time.perf_counter() - started

        return EvaluationGameResult(
            success=False,
            worker_id=int(worker_id),
            pid=os.getpid(),
            game_id=int(job.game_id),
            seed=int(job.seed),
            model_a_player=int(job.model_a_player),
            seconds=float(elapsed),
            evaluator_a_stats=evaluator_a.stats_snapshot(),
            evaluator_b_stats=evaluator_b.stats_snapshot(),
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
    inference_a_request_queue,
    inference_a_response_queue,
    inference_b_request_queue,
    inference_b_response_queue,
    config: EvaluationWorkerConfig,
) -> None:
    """Long-lived CPU worker. It never owns a model or initializes CUDA."""

    try:
        config.validate()

        evaluator_a = MultiprocessNeuralEvaluator(
            worker_id=int(worker_id),
            request_queue=inference_a_request_queue,
            response_queue=inference_a_response_queue,
            request_timeout_s=config.inference_timeout_s,
            request_put_timeout_s=config.request_put_timeout_s,
            return_policy_device="cpu",
        )

        evaluator_b = MultiprocessNeuralEvaluator(
            worker_id=int(worker_id),
            request_queue=inference_b_request_queue,
            response_queue=inference_b_response_queue,
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
                    evaluator_a=evaluator_a,
                    evaluator_b=evaluator_b,
                    job=job,
                    config=config,
                )
            )

    finally:
        evaluator_a.close()
        evaluator_b.close()


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
    attr_name: str,
) -> dict[int, dict[str, Any]]:
    latest: dict[int, dict[str, Any]] = {}
    for result in results:
        stats = getattr(result, attr_name, None)
        if stats:
            latest[int(result.worker_id)] = stats
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
        weighted_wait += float(stats.get("average_response_wait_ms", 0.0) or 0.0) * n
        weighted_round_trip += float(stats.get("average_round_trip_ms", 0.0) or 0.0) * n

    return {
        "requests_submitted": submitted,
        "responses_completed": completed,
        "responses_failed": failed,
        "average_response_wait_ms": (
            weighted_wait / weight if weight else None
        ),
        "average_round_trip_ms": (
            weighted_round_trip / weight if weight else None
        ),
    }


def normal_score_interval(
    wins: int,
    draws: int,
    games: int,
    z: float = 1.96,
) -> Optional[tuple[float, float, float]]:
    """
    Approximate interval for match score where win=1, draw=.5, loss=0.

    This is only a quick descriptive uncertainty estimate; game outcomes from
    paired seeds are not perfectly independent Bernoulli observations.
    """

    if games <= 0:
        return None

    score = (wins + 0.5 * draws) / games

    # Conservative Bernoulli-style standard error on the score fraction.
    se = math.sqrt(max(0.0, score * (1.0 - score) / games))
    return (
        float(score),
        float(max(0.0, score - z * se)),
        float(min(1.0, score + z * se)),
    )


def build_summary(
    *,
    successful: list[EvaluationGameResult],
    wall_seconds: float,
    server_a_stats: Optional[dict[str, Any]],
    server_b_stats: Optional[dict[str, Any]],
) -> dict[str, Any]:
    games = len(successful)
    a_wins = sum(r.outcome == "A" for r in successful)
    b_wins = sum(r.outcome == "B" for r in successful)
    draws = sum(r.outcome == "draw" for r in successful)

    a_as_p0 = [r for r in successful if r.model_a_player == 0]
    a_as_p1 = [r for r in successful if r.model_a_player == 1]

    interval = normal_score_interval(a_wins, draws, games)

    a_searches = sum(r.model_a_searches for r in successful)
    b_searches = sum(r.model_b_searches for r in successful)
    a_sims = sum(r.model_a_simulations for r in successful)
    b_sims = sum(r.model_b_simulations for r in successful)

    return {
        "games": games,
        "model_a_wins": a_wins,
        "model_b_wins": b_wins,
        "draws": draws,
        "model_a_win_rate": a_wins / games if games else None,
        "model_b_win_rate": b_wins / games if games else None,
        "draw_rate": draws / games if games else None,
        "model_a_match_score": interval[0] if interval else None,
        "model_a_match_score_approx_95ci": (
            [interval[1], interval[2]] if interval else None
        ),
        "model_a_as_player0": {
            "games": len(a_as_p0),
            "wins": sum(r.outcome == "A" for r in a_as_p0),
            "losses": sum(r.outcome == "B" for r in a_as_p0),
            "draws": sum(r.outcome == "draw" for r in a_as_p0),
        },
        "model_a_as_player1": {
            "games": len(a_as_p1),
            "wins": sum(r.outcome == "A" for r in a_as_p1),
            "losses": sum(r.outcome == "B" for r in a_as_p1),
            "draws": sum(r.outcome == "draw" for r in a_as_p1),
        },
        "average_positions": safe_mean(r.positions for r in successful),
        "average_final_turn_number": safe_mean(
            r.final_turn_number for r in successful
        ),
        "average_model_a_score": safe_mean(
            r.model_a_score for r in successful
        ),
        "average_model_b_score": safe_mean(
            r.model_b_score for r in successful
        ),
        "model_a_searches": a_searches,
        "model_b_searches": b_searches,
        "model_a_average_simulations_per_search": (
            a_sims / a_searches if a_searches else None
        ),
        "model_b_average_simulations_per_search": (
            b_sims / b_searches if b_searches else None
        ),
        "model_a_stop_reasons": merge_counters(
            r.model_a_stop_reasons for r in successful
        ),
        "model_b_stop_reasons": merge_counters(
            r.model_b_stop_reasons for r in successful
        ),
        "wall_seconds": float(wall_seconds),
        "games_per_hour": (
            games / wall_seconds * 3600.0 if wall_seconds > 0 else None
        ),
        "gpu_server_a": server_a_stats,
        "gpu_server_b": server_b_stats,
        "client_a": aggregate_client_stats(
            latest_worker_stats(successful, "evaluator_a_stats")
        ),
        "client_b": aggregate_client_stats(
            latest_worker_stats(successful, "evaluator_b_stats")
        ),
    }


# ============================================================================
# SERVER / PROCESS LIFECYCLE
# ============================================================================


def drain_latest_stats(server, latest: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
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

    # Capture any periodic telemetry before shutdown.
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


def run_parallel_evaluation(args) -> tuple[dict[str, Any], list[EvaluationGameResult]]:
    model_a = Path(args.model_a)
    model_b = Path(args.model_b)

    if not model_a.exists():
        raise FileNotFoundError(f"Model A checkpoint does not exist: {model_a}")
    if not model_b.exists():
        raise FileNotFoundError(f"Model B checkpoint does not exist: {model_b}")
    if args.games < 1:
        raise ValueError("games must be >= 1")
    if args.workers < 1:
        raise ValueError("workers must be >= 1")

    context = mp.get_context("spawn")

    ipc_a = create_inference_ipc(
        num_workers=args.workers,
        mp_context=context,
    )
    ipc_b = create_inference_ipc(
        num_workers=args.workers,
        mp_context=context,
    )

    server_a = GPUInferenceServerProcess(
        config=GPUInferenceServerConfig(
            checkpoint_path=str(model_a),
            max_batch_size=args.gpu_max_batch_size,
            batch_wait_ms=args.gpu_batch_wait_ms,
            device=args.device,
            stats_report_interval_batches=250,
        ),
        ipc=ipc_a,
        mp_context=context,
        process_name="SplendorEvalGPU_A",
    )

    server_b = GPUInferenceServerProcess(
        config=GPUInferenceServerConfig(
            checkpoint_path=str(model_b),
            max_batch_size=args.gpu_max_batch_size,
            batch_wait_ms=args.gpu_batch_wait_ms,
            device=args.device,
            stats_report_interval_batches=250,
        ),
        ipc=ipc_b,
        mp_context=context,
        process_name="SplendorEvalGPU_B",
    )

    worker_config = EvaluationWorkerConfig(
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
        inference_timeout_s=args.inference_timeout_s,
        request_put_timeout_s=args.request_put_timeout_s,
    )
    worker_config.validate()

    job_queue = context.Queue()
    result_queue = context.Queue()
    processes = []

    latest_a_stats = None
    latest_b_stats = None
    server_a_stats = None
    server_b_stats = None
    results: list[EvaluationGameResult] = []

    server_a.start()

    try:
        ready_a = server_a.wait_until_ready(timeout_s=args.startup_timeout_s)
        print(f"GPU server A READY (pid={ready_a.pid})")
        print("  checkpoint:", model_a)
        print("  model mode: eval() enforced by GPUInferenceServerProcess loader")

        server_b.start()
        ready_b = server_b.wait_until_ready(timeout_s=args.startup_timeout_s)
        print(f"GPU server B READY (pid={ready_b.pid})")
        print("  checkpoint:", model_b)
        print("  model mode: eval() enforced by GPUInferenceServerProcess loader")

        for worker_id in range(args.workers):
            process = context.Process(
                target=evaluation_worker_main,
                kwargs={
                    "worker_id": worker_id,
                    "job_queue": job_queue,
                    "result_queue": result_queue,
                    "inference_a_request_queue": ipc_a.request_queue,
                    "inference_a_response_queue": ipc_a.response_queues[worker_id],
                    "inference_b_request_queue": ipc_b.request_queue,
                    "inference_b_response_queue": ipc_b.response_queues[worker_id],
                    "config": worker_config,
                },
                name=f"SplendorEvalWorker-{worker_id}",
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

        # Paired-seat design:
        #   game 0: seed S, A=P0
        #   game 1: seed S, A=P1
        #   game 2: seed S+1, A=P0
        #   game 3: seed S+1, A=P1
        for game_id in range(args.games):
            pair_index = game_id // 2
            model_a_player = 0 if game_id % 2 == 0 else 1
            job_queue.put(
                EvaluationGameJob(
                    game_id=game_id,
                    seed=int(args.base_seed + pair_index),
                    model_a_player=model_a_player,
                )
            )

        started = time.perf_counter()
        last_result_time = started
        progress_every = max(1, args.games // 20)

        while len(results) < args.games:
            latest_a_stats = drain_latest_stats(server_a, latest_a_stats)
            latest_b_stats = drain_latest_stats(server_b, latest_b_stats)

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
                if time.perf_counter() - last_result_time > args.result_stall_timeout_s:
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
                a_wins = sum(r.outcome == "A" for r in results)
                b_wins = sum(r.outcome == "B" for r in results)
                draws = sum(r.outcome == "draw" for r in results)
                gph = len(results) / elapsed * 3600.0 if elapsed > 0 else 0.0
                print(
                    f"Evaluation {len(results)}/{args.games} "
                    f"- A {a_wins} / B {b_wins} / D {draws} "
                    f"- {gph:.2f} games/hour"
                )

        wall_seconds = time.perf_counter() - started

        # Normal worker shutdown.
        for _ in processes:
            job_queue.put(EvaluationWorkerShutdown())

        for process in processes:
            process.join(timeout=args.shutdown_timeout_s)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5.0)

        # Stop servers only after workers have stopped issuing requests.
        server_a_stats = stop_server_and_get_stats(
            server_a,
            latest_a_stats,
            args.shutdown_timeout_s,
        )
        server_b_stats = stop_server_and_get_stats(
            server_b,
            latest_b_stats,
            args.shutdown_timeout_s,
        )

        results.sort(key=lambda r: r.game_id)

        summary = build_summary(
            successful=results,
            wall_seconds=wall_seconds,
            server_a_stats=server_a_stats,
            server_b_stats=server_b_stats,
        )

        return summary, results

    finally:
        # Best-effort cleanup on exceptions.
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

        if server_a.is_alive():
            try:
                server_a.close(
                    timeout_s=args.shutdown_timeout_s,
                    terminate_if_needed=True,
                )
            except Exception:
                pass

        if server_b.is_alive():
            try:
                server_b.close(
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
    print("PARALLEL MODEL 4 EVALUATION COMPLETE")
    print("=" * 78)
    print("Model A:", args.model_a)
    print("Model B:", args.model_b)
    print("Games:", summary["games"])
    print()
    print("Model A wins:", summary["model_a_wins"], f"({percent(summary['model_a_win_rate'])})")
    print("Model B wins:", summary["model_b_wins"], f"({percent(summary['model_b_win_rate'])})")
    print("Draws:", summary["draws"], f"({percent(summary['draw_rate'])})")
    print("Model A match score (win=1, draw=.5):", percent(summary["model_a_match_score"]))

    ci = summary.get("model_a_match_score_approx_95ci")
    if ci:
        print("Approx A score 95% interval:", f"[{percent(ci[0])}, {percent(ci[1])}]")

    print()
    print("A as Player 0:", summary["model_a_as_player0"])
    print("A as Player 1:", summary["model_a_as_player1"])
    print("Average positions:", summary["average_positions"])
    print("Average A score:", summary["average_model_a_score"])
    print("Average B score:", summary["average_model_b_score"])
    print()
    print("A average simulations/search:", summary["model_a_average_simulations_per_search"])
    print("B average simulations/search:", summary["model_b_average_simulations_per_search"])
    print("A MCTS stop reasons:", summary["model_a_stop_reasons"])
    print("B MCTS stop reasons:", summary["model_b_stop_reasons"])
    print()
    print("Wall time:", f"{summary['wall_seconds']:.2f}s")
    print("Throughput:", f"{summary['games_per_hour']:.2f} games/hour")

    for label in ("a", "b"):
        stats = summary.get(f"gpu_server_{label}") or {}
        print()
        print(f"GPU server {label.upper()} average batch size:", stats.get("average_batch_size"))
        print(f"GPU server {label.upper()} max observed batch size:", stats.get("max_observed_batch_size"))
        print(f"GPU server {label.upper()} inference positions/sec:", stats.get("inference_positions_per_second"))

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
        "model_a": str(args.model_a),
        "model_b": str(args.model_b),
        "config": {
            "games": args.games,
            "workers": args.workers,
            "gpu_max_batch_size": args.gpu_max_batch_size,
            "gpu_batch_wait_ms": args.gpu_batch_wait_ms,
            "base_seed": args.base_seed,
            "simulations": args.simulations,
            "min_simulations": args.min_simulations,
            "check_interval": args.check_interval,
            "target_visits_per_action": args.target_visits_per_action,
            "single_action_simulations": args.single_action_simulations,
            "stability_checks": args.stability_checks,
            "c_puct": args.c_puct,
            "temperature": 0.0,
            "root_noise": False,
            "paired_seats": True,
            "model_mode": "eval",
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
        description="Parallel Model 4 checkpoint-vs-checkpoint evaluator"
    )

    parser.add_argument("--model-a", required=True, help="Model A checkpoint")
    parser.add_argument("--model-b", required=True, help="Model B checkpoint")
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
    parser.add_argument("--min-simulations", type=int, default=DEFAULT_MIN_SIMULATIONS)
    parser.add_argument("--check-interval", type=int, default=DEFAULT_CHECK_INTERVAL)
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
    parser.add_argument("--max-game-steps", type=int, default=DEFAULT_MAX_GAME_STEPS)

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
        default="splendor_v1/evaluation/results/latest_parallel_eval.json",
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()

    print("=" * 78)
    print("SPLENDOR V6 PARALLEL MODEL EVALUATOR")
    print("=" * 78)
    print("Model A:", args.model_a)
    print("Model B:", args.model_b)
    print("Games:", args.games)
    print("Workers:", args.workers)
    print("GPU max batch size:", args.gpu_max_batch_size)
    print("GPU batch wait ms:", args.gpu_batch_wait_ms)
    print("MCTS simulations hard cap:", args.simulations)
    print("MCTS minimum simulations:", args.min_simulations)
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
