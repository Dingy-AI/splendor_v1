"""
Production coordinator for multiprocess Splendor self-play.

Responsibilities
----------------

The coordinator runs in the MAIN process and owns:

    - the persistent ReplayBuffer
    - the centralized GPU inference server lifecycle
    - the long-lived CPU self-play worker pool
    - game job scheduling
    - completed-game validation
    - whole-game ReplayBuffer commits
    - progress / checkpoint callbacks
    - aggregate self-play diagnostics
    - clean shutdown

Workers NEVER mutate the persistent replay buffer.

Completed games flow like:

    worker
      |
      | SelfPlayGameResult(
      |     samples=[...],
      |     game_metadata={...},
      | )
      v
    MAIN coordinator
      |
      | replay_buffer.add_game(
      |     samples=...,
      |     game_metadata=...,
      | )
      v
    persistent replay buffer

Continuous scheduling
---------------------

Only up to num_workers games are in flight at once.

When one game finishes:

    commit whole game
        ->
    immediately submit next game

so faster CPU workers do not sit idle waiting for slower games.

Recommended training-loop lifecycle
-----------------------------------

For the current block-style training loop:

    coordinator = MultiprocessSelfPlayCoordinator(...)

    summary = coordinator.run_block(
        num_games=50,
        seed_start=50000,
    )

    # coordinator is now stopped
    # train Model 4
    # save next checkpoint
    # construct a new coordinator with the new checkpoint

This intentionally avoids hot-reloading model weights inside the GPU
server for now.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
import multiprocessing as mp
from queue import Empty
import statistics
import time
from typing import Any, Callable, Optional

from splendor_v1.mcts_batched.gpu_inference_server import (
    GPUInferenceServerConfig,
    GPUInferenceServerProcess,
    create_inference_ipc,
)

from splendor_v1.mcts_batched.multiprocess_self_play_worker import (
    MultiprocessSelfPlayPool,
    SelfPlayGameJob,
    SelfPlayGameResult,
    SelfPlayWorkerConfig,
    SelfPlayWorkerReady,
)


# ============================================================
# CONFIG
# ============================================================


@dataclass(slots=True)
class MultiprocessSelfPlayCoordinatorConfig:
    num_workers: int

    checkpoint_path: str

    worker_config: SelfPlayWorkerConfig = field(
        default_factory=SelfPlayWorkerConfig
    )

    device: str = "cuda"

    # None -> use num_workers. Because each synchronous worker has
    # at most one outstanding inference request, a larger value
    # cannot be filled without adding more in-flight game clients.
    max_batch_size: Optional[int] = None

    batch_wait_ms: float = 0.5

    server_stats_report_interval_batches: int = 1000

    startup_timeout_s: float = 180.0

    # Polling uses short waits internally; this is the maximum
    # time without a completed game before treating the block as
    # stalled.
    game_result_timeout_s: float = 600.0

    shutdown_timeout_s: float = 30.0

    # Production coordinator requires samples because it commits
    # complete games to the real replay buffer.
    require_worker_samples: bool = True

    def validate(self):
        if self.num_workers < 1:
            raise ValueError(
                "num_workers must be >= 1."
            )

        if not self.checkpoint_path:
            raise ValueError(
                "checkpoint_path must not be empty."
            )

        if self.max_batch_size is not None:
            if self.max_batch_size < 1:
                raise ValueError(
                    "max_batch_size must be >= 1."
                )

        if self.batch_wait_ms < 0:
            raise ValueError(
                "batch_wait_ms must be >= 0."
            )

        if self.startup_timeout_s <= 0:
            raise ValueError(
                "startup_timeout_s must be > 0."
            )

        if self.game_result_timeout_s <= 0:
            raise ValueError(
                "game_result_timeout_s must be > 0."
            )

        if self.shutdown_timeout_s <= 0:
            raise ValueError(
                "shutdown_timeout_s must be > 0."
            )

        self.worker_config.validate()

        if (
            self.require_worker_samples
            and not self.worker_config.return_samples
        ):
            raise ValueError(
                "Coordinator requires "
                "worker_config.return_samples=True."
            )

    @property
    def resolved_max_batch_size(
        self,
    ):
        if self.max_batch_size is None:
            return int(
                self.num_workers
            )

        return int(
            self.max_batch_size
        )


# ============================================================
# ERRORS
# ============================================================


class MultiprocessSelfPlayCoordinatorError(
    RuntimeError
):
    pass


class SelfPlayWorkerCrashed(
    MultiprocessSelfPlayCoordinatorError
):
    pass


class SelfPlayGameFailed(
    MultiprocessSelfPlayCoordinatorError
):
    pass


class ReplayCommitError(
    MultiprocessSelfPlayCoordinatorError
):
    pass


# ============================================================
# SUMMARY HELPERS
# ============================================================


def _weighted_mean(
    pairs,
):
    numerator = 0.0
    denominator = 0.0

    for value, weight in pairs:
        if value is None:
            continue

        weight = float(
            weight
        )

        if weight <= 0:
            continue

        numerator += (
            float(value)
            * weight
        )

        denominator += weight

    if denominator <= 0:
        return None

    return float(
        numerator / denominator
    )


def _merge_stop_reasons(
    results,
):
    merged = {}

    for result in results:
        summary = (
            result.search_summary
            or {}
        )

        reasons = summary.get(
            "stop_reasons",
            {},
        )

        for reason, count in reasons.items():
            reason = str(
                reason
            )

            merged[reason] = (
                merged.get(
                    reason,
                    0,
                )
                + int(
                    count
                )
            )

    return merged


def _latest_worker_evaluator_stats(
    results,
):
    """
    Evaluator stats are cumulative within each long-lived worker.
    Keep only that worker's latest snapshot.
    """

    latest = {}

    for result in results:
        worker_id = int(
            result.worker_id
        )

        latest[worker_id] = (
            result.evaluator_stats
            or {}
        )

    return latest


def _aggregate_worker_evaluator_stats(
    latest_stats,
):
    if not latest_stats:
        return {}

    submitted = sum(
        int(
            stats.get(
                "requests_submitted",
                0,
            )
        )
        for stats
        in latest_stats.values()
    )

    completed = sum(
        int(
            stats.get(
                "responses_completed",
                0,
            )
        )
        for stats
        in latest_stats.values()
    )

    failed = sum(
        int(
            stats.get(
                "responses_failed",
                0,
            )
        )
        for stats
        in latest_stats.values()
    )

    unexpected = sum(
        int(
            stats.get(
                "unexpected_responses",
                0,
            )
        )
        for stats
        in latest_stats.values()
    )

    average_round_trip_ms = (
        _weighted_mean(
            [
                (
                    stats.get(
                        "average_round_trip_ms"
                    ),
                    stats.get(
                        "responses_completed",
                        0,
                    ),
                )
                for stats
                in latest_stats.values()
            ]
        )
    )

    average_response_wait_ms = (
        _weighted_mean(
            [
                (
                    stats.get(
                        "average_response_wait_ms"
                    ),
                    stats.get(
                        "responses_completed",
                        0,
                    ),
                )
                for stats
                in latest_stats.values()
            ]
        )
    )

    max_round_trip_ms = max(
        [
            float(
                stats.get(
                    "max_round_trip_ms",
                    0.0,
                )
            )
            for stats
            in latest_stats.values()
        ],
        default=0.0,
    )

    return {
        "requests_submitted":
            int(
                submitted
            ),

        "responses_completed":
            int(
                completed
            ),

        "responses_failed":
            int(
                failed
            ),

        "unexpected_responses":
            int(
                unexpected
            ),

        "average_round_trip_ms":
            average_round_trip_ms,

        "average_response_wait_ms":
            average_response_wait_ms,

        "max_round_trip_ms":
            float(
                max_round_trip_ms
            ),

        "per_worker":
            copy.deepcopy(
                latest_stats
            ),
    }


# ============================================================
# COORDINATOR
# ============================================================


class MultiprocessSelfPlayCoordinator:
    """
    Main-process owner of one self-play block.
    """

    def __init__(
        self,
        *,
        replay_buffer,
        config,
        mp_context=None,
    ):
        config.validate()

        if replay_buffer is None:
            raise ValueError(
                "replay_buffer is required."
            )

        if not hasattr(
            replay_buffer,
            "add_game",
        ):
            raise TypeError(
                "Persistent replay_buffer must expose "
                "add_game(samples=..., game_metadata=...)."
            )

        if mp_context is None:
            mp_context = mp.get_context(
                "spawn"
            )

        self.replay_buffer = (
            replay_buffer
        )

        self.config = config
        self.mp_context = (
            mp_context
        )

        self.ipc = None
        self.server = None
        self.pool = None

        self._started = False
        self._closed = False

        self._worker_pids = {}

        self._last_gpu_server_stats = None
        self._last_run_summary = None

    # --------------------------------------------------------
    # Lifecycle
    # --------------------------------------------------------

    @property
    def started(self):
        return bool(
            self._started
        )

    @property
    def closed(self):
        return bool(
            self._closed
        )

    @property
    def worker_pids(self):
        return dict(
            self._worker_pids
        )

    @property
    def last_gpu_server_stats(self):
        return copy.deepcopy(
            self._last_gpu_server_stats
        )

    @property
    def last_run_summary(self):
        return copy.deepcopy(
            self._last_run_summary
        )

    def start(
        self,
    ):
        if self._closed:
            raise RuntimeError(
                "Coordinator instances cannot be "
                "restarted after close(). Create a "
                "new coordinator for the next model "
                "checkpoint."
            )

        if self._started:
            return

        self.ipc = (
            create_inference_ipc(
                num_workers=(
                    self.config.num_workers
                ),
                mp_context=(
                    self.mp_context
                ),
            )
        )

        self.server = (
            GPUInferenceServerProcess(
                config=(
                    GPUInferenceServerConfig(
                        checkpoint_path=(
                            self.config
                            .checkpoint_path
                        ),
                        max_batch_size=(
                            self.config
                            .resolved_max_batch_size
                        ),
                        batch_wait_ms=(
                            self.config
                            .batch_wait_ms
                        ),
                        device=(
                            self.config.device
                        ),
                        stats_report_interval_batches=(
                            self.config
                            .server_stats_report_interval_batches
                        ),
                    )
                ),
                ipc=self.ipc,
                mp_context=(
                    self.mp_context
                ),
            )
        )

        self.pool = (
            MultiprocessSelfPlayPool(
                num_workers=(
                    self.config.num_workers
                ),
                inference_ipc=(
                    self.ipc
                ),
                config=(
                    self.config
                    .worker_config
                ),
                mp_context=(
                    self.mp_context
                ),
            )
        )

        try:
            self.server.start()

            self.server.wait_until_ready(
                timeout_s=(
                    self.config
                    .startup_timeout_s
                )
            )

            self.pool.start()

            self._wait_for_all_workers_ready()

            self._started = True

        except BaseException:
            self.close()

            raise

    def _wait_for_all_workers_ready(
        self,
    ):
        deadline = (
            time.perf_counter()
            + self.config.startup_timeout_s
        )

        worker_pids = {}

        while (
            len(worker_pids)
            < self.config.num_workers
        ):
            remaining = (
                deadline
                - time.perf_counter()
            )

            if remaining <= 0:
                raise TimeoutError(
                    "Timed out waiting for all "
                    "self-play workers to become READY."
                )

            try:
                message = self.pool.get_result(
                    timeout=min(
                        remaining,
                        1.0,
                    )
                )

            except Empty:
                self._raise_if_worker_died(
                    context=(
                        "while waiting for READY"
                    )
                )

                continue

            if isinstance(
                message,
                SelfPlayWorkerReady,
            ):
                worker_id = int(
                    message.worker_id
                )

                worker_pids[
                    worker_id
                ] = int(
                    message.pid
                )

            elif isinstance(
                message,
                SelfPlayGameResult,
            ):
                raise MultiprocessSelfPlayCoordinatorError(
                    "Received a game result before "
                    "all workers reported READY."
                )

        if len(
            set(
                worker_pids.values()
            )
        ) != self.config.num_workers:
            raise MultiprocessSelfPlayCoordinatorError(
                "Self-play workers did not start "
                "with distinct process IDs."
            )

        if (
            self.server.pid is not None
            and int(
                self.server.pid
            )
            in set(
                worker_pids.values()
            )
        ):
            raise MultiprocessSelfPlayCoordinatorError(
                "GPU inference server PID matches "
                "a CPU worker PID."
            )

        self._worker_pids = (
            worker_pids
        )

    def close(
        self,
    ):
        if self._closed:
            return

        # Stop CPU workers first so they stop generating new neural
        # requests. Then stop the GPU server.
        if self.pool is not None:
            try:
                self.pool.close(
                    timeout_s=(
                        self.config
                        .shutdown_timeout_s
                    ),
                    terminate_if_needed=True,
                )

            except Exception:
                pass

        if self.server is not None:
            try:
                self.server.close(
                    timeout_s=(
                        self.config
                        .shutdown_timeout_s
                    ),
                    terminate_if_needed=True,
                )

                self._last_gpu_server_stats = (
                    self.server.final_stats
                )

            except Exception:
                pass

        self._started = False
        self._closed = True

    def __enter__(
        self,
    ):
        self.start()

        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback_object,
    ):
        self.close()

    # --------------------------------------------------------
    # Health
    # --------------------------------------------------------

    def _raise_if_worker_died(
        self,
        *,
        context,
    ):
        if self.pool is None:
            return

        dead = []

        for worker_id, process in enumerate(
            self.pool.processes
        ):
            if process.exitcode is not None:
                dead.append(
                    (
                        int(
                            worker_id
                        ),
                        int(
                            process.exitcode
                        ),
                    )
                )

        if dead:
            raise SelfPlayWorkerCrashed(
                "One or more self-play workers "
                f"exited {context}: {dead}"
            )

        if (
            self.server is not None
            and not self.server.is_alive()
        ):
            raise SelfPlayWorkerCrashed(
                "GPU inference server exited "
                f"{context}. exitcode="
                f"{self.server.exitcode}"
            )

    # --------------------------------------------------------
    # Persistent replay commit
    # --------------------------------------------------------

    def _commit_completed_game(
        self,
        result,
    ):
        if result.samples is None:
            raise ReplayCommitError(
                "Completed worker game returned "
                "samples=None. Production "
                "coordination requires "
                "return_samples=True."
            )

        if len(
            result.samples
        ) == 0:
            raise ReplayCommitError(
                "Completed worker game returned "
                "an empty sample list."
            )

        if result.game_metadata is None:
            raise ReplayCommitError(
                "Completed worker game returned "
                "no game_metadata."
            )

        metadata = copy.deepcopy(
            result.game_metadata
        )

        # Coordinator-specific provenance is additive. Do not
        # overwrite ReplayGenerator's existing metadata.
        metadata.setdefault(
            "multiprocess_coordinator",
            True,
        )

        metadata.setdefault(
            "source_worker_id",
            int(
                result.worker_id
            ),
        )

        metadata.setdefault(
            "source_worker_pid",
            int(
                result.pid
            ),
        )

        metadata.setdefault(
            "source_job_game_id",
            int(
                result.game_id
            ),
        )

        # V5/V6 resume logic uses game_index as the authoritative
        # attempted-game index. Preserve a worker-provided value and
        # fall back to the coordinator job id.
        metadata.setdefault(
            "game_index",
            int(
                result.game_id
            ),
        )

        metadata.setdefault(
            "source_seed",
            int(
                result.seed
            ),
        )

        try:
            persistent_game_id = (
                self.replay_buffer.add_game(
                    samples=(
                        result.samples
                    ),
                    game_metadata=(
                        metadata
                    ),
                )
            )

        except Exception as exc:
            raise ReplayCommitError(
                "Persistent ReplayBuffer.add_game "
                "failed for "
                f"job_game_id={result.game_id}, "
                f"seed={result.seed}: {exc}"
            ) from exc

        return persistent_game_id

    # --------------------------------------------------------
    # Continuous scheduling
    # --------------------------------------------------------

    def generate_games(
        self,
        *,
        num_games,
        seed_start,
        game_id_start=0,
        extra_game_metadata=None,
        checkpoint_every_games=None,
        checkpoint_callback=None,
        progress_callback=None,

        # Optional callback used by V6 to preserve deterministic
        # seed/split/metadata assignment:
        #
        #   job_factory(
        #       offset,
        #       default_game_id,
        #       default_seed,
        #   ) -> SelfPlayGameJob
        job_factory=None,
    ):
        """
        Generate and commit a block of completed self-play games.

        Parameters
        ----------
        num_games:
            Number of SUCCESSFULLY COMMITTED games requested.

        seed_start:
            Seed assigned to the first game. Subsequent jobs use
            seed_start + offset.

        game_id_start:
            Coordinator job ID assigned to the first game.

        extra_game_metadata:
            Optional metadata copied into every worker game. Per-game
            coordinator provenance is added separately.

        checkpoint_every_games:
            Optional positive integer. If provided, invoke
            checkpoint_callback every N committed games.

        checkpoint_callback:
            Callable:
                callback(
                    replay_buffer,
                    progress_dict,
                )

            The coordinator intentionally does not know your replay
            serialization format.

        progress_callback:
            Optional callable invoked after every committed game:
                callback(progress_dict)

        Returns
        -------
        dict
            Aggregate block diagnostics.
        """

        if num_games < 1:
            raise ValueError(
                "num_games must be >= 1."
            )

        if (
            checkpoint_every_games
            is not None
        ):
            checkpoint_every_games = int(
                checkpoint_every_games
            )

            if checkpoint_every_games < 1:
                raise ValueError(
                    "checkpoint_every_games "
                    "must be >= 1."
                )

            if checkpoint_callback is None:
                raise ValueError(
                    "checkpoint_callback is required "
                    "when checkpoint_every_games is set."
                )

        if not self._started:
            self.start()

        if self._closed:
            raise RuntimeError(
                "Coordinator is closed."
            )

        block_start = (
            time.perf_counter()
        )

        requested = int(
            num_games
        )

        next_offset = 0

        submitted = 0
        completed = 0
        committed = 0

        results = []
        persistent_game_ids = []

        in_flight = 0

        shared_metadata = (
            copy.deepcopy(
                extra_game_metadata
            )
            if extra_game_metadata
            else {}
        )

        # ----------------------------------------------------
        # Local submission helper
        # ----------------------------------------------------

        def submit_next_job():
            nonlocal next_offset
            nonlocal submitted
            nonlocal in_flight

            if next_offset >= requested:
                return False

            default_game_id = (
                int(
                    game_id_start
                )
                + next_offset
            )

            default_seed = (
                int(
                    seed_start
                )
                + next_offset
            )

            if job_factory is None:
                job = SelfPlayGameJob(
                    game_id=(
                        default_game_id
                    ),
                    seed=(
                        default_seed
                    ),
                    extra_game_metadata=(
                        copy.deepcopy(
                            shared_metadata
                        )
                    ),
                )

            else:
                job = job_factory(
                    int(
                        next_offset
                    ),
                    int(
                        default_game_id
                    ),
                    int(
                        default_seed
                    ),
                )

                if not isinstance(
                    job,
                    SelfPlayGameJob,
                ):
                    raise TypeError(
                        "job_factory must return "
                        "SelfPlayGameJob."
                    )

            self.pool.submit(
                job
            )

            next_offset += 1
            submitted += 1
            in_flight += 1

            return True

        # Keep only one job per worker in flight. When a worker
        # completes a game, immediately release the next job.
        initial_jobs = min(
            self.config.num_workers,
            requested,
        )

        for _ in range(
            initial_jobs
        ):
            submit_next_job()

        last_result_time = (
            time.perf_counter()
        )

        while committed < requested:
            elapsed_without_result = (
                time.perf_counter()
                - last_result_time
            )

            remaining_stall_budget = (
                self.config
                .game_result_timeout_s
                - elapsed_without_result
            )

            if remaining_stall_budget <= 0:
                raise TimeoutError(
                    "No completed self-play game was "
                    "received within "
                    f"{self.config.game_result_timeout_s}s."
                )

            try:
                message = self.pool.get_result(
                    timeout=min(
                        remaining_stall_budget,
                        1.0,
                    )
                )

            except Empty:
                self._raise_if_worker_died(
                    context=(
                        "while generating games"
                    )
                )

                continue

            if isinstance(
                message,
                SelfPlayWorkerReady,
            ):
                # All READY events should have been consumed by
                # start(), but treating an extra event as harmless
                # makes the coordinator robust to delayed queue
                # delivery.
                continue

            if not isinstance(
                message,
                SelfPlayGameResult,
            ):
                continue

            last_result_time = (
                time.perf_counter()
            )

            completed += 1
            in_flight -= 1

            if not message.success:
                raise SelfPlayGameFailed(
                    "Self-play game failed: "
                    f"worker={message.worker_id}, "
                    f"pid={message.pid}, "
                    f"game_id={message.game_id}, "
                    f"seed={message.seed}, "
                    f"error_type={message.error_type}, "
                    f"error={message.error}\n"
                    f"{message.traceback or ''}"
                )

            persistent_game_id = (
                self._commit_completed_game(
                    message
                )
            )

            results.append(
                message
            )

            persistent_game_ids.append(
                persistent_game_id
            )

            committed += 1

            # The completed worker is now available. Put exactly
            # one new job into the shared job queue if work remains.
            submit_next_job()

            progress = (
                self._build_progress_snapshot(
                    requested=requested,
                    submitted=submitted,
                    completed=completed,
                    committed=committed,
                    in_flight=in_flight,
                    block_start=block_start,
                    latest_result=message,
                    persistent_game_id=(
                        persistent_game_id
                    ),
                    results=results,
                )
            )

            if progress_callback is not None:
                progress_callback(
                    copy.deepcopy(
                        progress
                    )
                )

            if (
                checkpoint_every_games
                is not None
                and committed
                % checkpoint_every_games
                == 0
            ):
                checkpoint_callback(
                    self.replay_buffer,
                    copy.deepcopy(
                        progress
                    ),
                )

        summary = (
            self._build_final_summary(
                requested=requested,
                submitted=submitted,
                completed=completed,
                committed=committed,
                block_start=block_start,
                results=results,
                persistent_game_ids=(
                    persistent_game_ids
                ),
            )
        )

        self._last_run_summary = (
            copy.deepcopy(
                summary
            )
        )

        return summary

    # --------------------------------------------------------
    # Progress / summaries
    # --------------------------------------------------------

    def _build_progress_snapshot(
        self,
        *,
        requested,
        submitted,
        completed,
        committed,
        in_flight,
        block_start,
        latest_result,
        persistent_game_id,
        results,
    ):
        wall_seconds = (
            time.perf_counter()
            - block_start
        )

        positions = sum(
            int(
                result.num_positions
            )
            for result
            in results
        )

        return {
            "requested_games":
                int(
                    requested
                ),

            "submitted_games":
                int(
                    submitted
                ),

            "completed_worker_games":
                int(
                    completed
                ),

            "committed_games":
                int(
                    committed
                ),

            "in_flight_games":
                int(
                    in_flight
                ),

            "committed_positions":
                int(
                    positions
                ),

            "wall_seconds":
                float(
                    wall_seconds
                ),

            "games_per_hour":
                float(
                    committed
                    / wall_seconds
                    * 3600.0
                    if wall_seconds > 0
                    else 0.0
                ),

            "positions_per_hour":
                float(
                    positions
                    / wall_seconds
                    * 3600.0
                    if wall_seconds > 0
                    else 0.0
                ),

            "latest_job_game_id":
                int(
                    latest_result.game_id
                ),

            "latest_seed":
                int(
                    latest_result.seed
                ),

            "latest_worker_id":
                int(
                    latest_result.worker_id
                ),

            "latest_game_seconds":
                float(
                    latest_result.seconds
                ),

            "latest_positions":
                int(
                    latest_result.num_positions
                ),

            "latest_persistent_game_id":
                persistent_game_id,
        }

    def _build_final_summary(
        self,
        *,
        requested,
        submitted,
        completed,
        committed,
        block_start,
        results,
        persistent_game_ids,
    ):
        wall_seconds = (
            time.perf_counter()
            - block_start
        )

        game_seconds = [
            float(
                result.seconds
            )
            for result
            in results
        ]

        positions = [
            int(
                result.num_positions
            )
            for result
            in results
        ]

        total_positions = sum(
            positions
        )

        search_sample_counts = [
            int(
                (
                    result.search_summary
                    or {}
                ).get(
                    "search_samples_with_actual_simulations",
                    0,
                )
            )
            for result
            in results
        ]

        average_actual_simulations = (
            _weighted_mean(
                [
                    (
                        (
                            result.search_summary
                            or {}
                        ).get(
                            "average_actual_simulations"
                        ),
                        count,
                    )
                    for result, count
                    in zip(
                        results,
                        search_sample_counts,
                    )
                ]
            )
        )

        simulation_savings_fraction = (
            _weighted_mean(
                [
                    (
                        (
                            result.search_summary
                            or {}
                        ).get(
                            "simulation_savings_fraction"
                        ),
                        count,
                    )
                    for result, count
                    in zip(
                        results,
                        search_sample_counts,
                    )
                ]
            )
        )

        average_legal_actions = (
            _weighted_mean(
                [
                    (
                        (
                            result.search_summary
                            or {}
                        ).get(
                            "average_legal_actions"
                        ),
                        count,
                    )
                    for result, count
                    in zip(
                        results,
                        search_sample_counts,
                    )
                ]
            )
        )

        latest_worker_stats = (
            _latest_worker_evaluator_stats(
                results
            )
        )

        evaluator_summary = (
            _aggregate_worker_evaluator_stats(
                latest_worker_stats
            )
        )

        # Aggregate deterministic whole-game train/validation
        # assignments for this completed self-play block.
        split_games = {}
        split_positions = {}

        for result in results:
            metadata = (
                result.game_metadata
                or {}
            )

            split_name = str(
                metadata.get(
                    "split",
                    "unknown",
                )
            )

            split_games[
                split_name
            ] = (
                split_games.get(
                    split_name,
                    0,
                )
                + 1
            )

            split_positions[
                split_name
            ] = (
                split_positions.get(
                    split_name,
                    0,
                )
                + int(
                    result.num_positions
                )
            )

        return {
            "requested_games":
                int(
                    requested
                ),

            "submitted_games":
                int(
                    submitted
                ),

            "completed_worker_games":
                int(
                    completed
                ),

            "committed_games":
                int(
                    committed
                ),

            "failed_games":
                0,

            "wall_seconds":
                float(
                    wall_seconds
                ),

            "games_per_hour":
                float(
                    committed
                    / wall_seconds
                    * 3600.0
                    if wall_seconds > 0
                    else 0.0
                ),

            "total_positions":
                int(
                    total_positions
                ),

            "split_games":
                dict(
                    split_games
                ),

            "split_positions":
                dict(
                    split_positions
                ),

            "positions_per_hour":
                float(
                    total_positions
                    / wall_seconds
                    * 3600.0
                    if wall_seconds > 0
                    else 0.0
                ),

            "average_positions_per_game":
                float(
                    statistics.mean(
                        positions
                    )
                    if positions
                    else 0.0
                ),

            "average_individual_game_seconds":
                float(
                    statistics.mean(
                        game_seconds
                    )
                    if game_seconds
                    else 0.0
                ),

            "median_individual_game_seconds":
                float(
                    statistics.median(
                        game_seconds
                    )
                    if game_seconds
                    else 0.0
                ),

            "min_individual_game_seconds":
                float(
                    min(
                        game_seconds
                    )
                    if game_seconds
                    else 0.0
                ),

            "max_individual_game_seconds":
                float(
                    max(
                        game_seconds
                    )
                    if game_seconds
                    else 0.0
                ),

            "average_actual_simulations":
                average_actual_simulations,

            "simulation_savings_fraction":
                simulation_savings_fraction,

            "average_legal_actions":
                average_legal_actions,

            "stop_reasons":
                _merge_stop_reasons(
                    results
                ),

            "worker_evaluator":
                evaluator_summary,

            "worker_pids":
                self.worker_pids,

            "persistent_game_ids":
                list(
                    persistent_game_ids
                ),

            # GPU server is still running after generate_games().
            # run_block() closes it and fills this field.
            "gpu_server":
                None,
        }

    # --------------------------------------------------------
    # Recommended block-style API
    # --------------------------------------------------------

    def run_block(
        self,
        *,
        num_games,
        seed_start,
        game_id_start=0,
        extra_game_metadata=None,
        checkpoint_every_games=None,
        checkpoint_callback=None,
        progress_callback=None,
        job_factory=None,
    ):
        """
        Start -> generate/commit games -> cleanly stop.

        This is the recommended API for the current training loop,
        because the next training phase will produce a new model
        checkpoint anyway.
        """

        try:
            self.start()

            summary = self.generate_games(
                num_games=num_games,
                seed_start=seed_start,
                game_id_start=(
                    game_id_start
                ),
                extra_game_metadata=(
                    extra_game_metadata
                ),
                checkpoint_every_games=(
                    checkpoint_every_games
                ),
                checkpoint_callback=(
                    checkpoint_callback
                ),
                progress_callback=(
                    progress_callback
                ),
                job_factory=(
                    job_factory
                ),
            )

            return_summary = (
                copy.deepcopy(
                    summary
                )
            )

        finally:
            self.close()

        return_summary[
            "gpu_server"
        ] = copy.deepcopy(
            self._last_gpu_server_stats
        )

        self._last_run_summary = (
            copy.deepcopy(
                return_summary
            )
        )

        return return_summary
