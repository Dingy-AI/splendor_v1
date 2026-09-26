"""
Reusable spawned self-play worker for centralized GPU inference.

Each worker process owns CPU-side game/search state:

    - SplendorEnv
    - V5 MCTS
    - MCTS tree reuse
    - RNG streams
    - ModelReplayGenerator
    - private scratch ReplayBuffer

The worker does NOT own Model 4 or CUDA.

Every neural leaf evaluation goes through:

    MultiprocessNeuralEvaluator
        -> shared request_queue
        -> centralized GPU inference process
        -> this worker's response_queue

The main process owns the persistent replay buffer. Workers may return
their completed game's samples, but never mutate a shared ReplayBuffer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from queue import Empty
import statistics
import time
import traceback
from typing import Any, Optional

import numpy as np

from splendor_v1.env.env import SplendorEnv

from splendor_v1.training_v2.replay_buffer import (
    ReplayBuffer,
)

from splendor_v1.training_v2.state_serializer import (
    serialize_state,
)

from splendor_v1.training_v5.model_replay_generator_v5_pruning import (
    ModelReplayGenerator,
)

from splendor_v1.mcts_batched.mcts_v5_direct import (
    MCTS,
)

from splendor_v1.mcts_batched.multiprocess_neural_evaluator import (
    MultiprocessNeuralEvaluator,
)


# ============================================================
# MESSAGES / CONFIG
# ============================================================


@dataclass(slots=True)
class SelfPlaySearchConfig:
    simulations: int = 400
    min_simulations: int = 80
    check_interval: int = 20
    target_visits_per_action: float = 20.0
    single_action_simulations: int = 4
    stability_checks: int = 3

    c_puct: float = 3.0
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25

    max_game_steps: int = 300

    def validate(self):
        if self.simulations < 1:
            raise ValueError(
                "simulations must be >= 1."
            )

        if self.min_simulations < 1:
            raise ValueError(
                "min_simulations must be >= 1."
            )

        if self.check_interval < 1:
            raise ValueError(
                "check_interval must be >= 1."
            )

        if self.target_visits_per_action <= 0:
            raise ValueError(
                "target_visits_per_action must be > 0."
            )

        if self.single_action_simulations < 1:
            raise ValueError(
                "single_action_simulations must be >= 1."
            )

        if self.stability_checks < 1:
            raise ValueError(
                "stability_checks must be >= 1."
            )

        if self.max_game_steps < 1:
            raise ValueError(
                "max_game_steps must be >= 1."
            )


@dataclass(slots=True)
class SelfPlayWorkerConfig:
    search: SelfPlaySearchConfig = field(
        default_factory=SelfPlaySearchConfig
    )

    replay_capacity: int = 10_000

    request_timeout_s: float = 120.0
    request_put_timeout_s: float = 30.0

    model_generation: int = 4
    model_checkpoint_label: str = (
        "multiprocess_self_play"
    )

    split: str = "train"

    # Returning full samples is useful for the production coordinator.
    # Smoke tests can disable it to keep IPC payloads small.
    return_samples: bool = True

    def validate(self):
        self.search.validate()

        if self.replay_capacity < 1:
            raise ValueError(
                "replay_capacity must be >= 1."
            )

        if self.request_timeout_s <= 0:
            raise ValueError(
                "request_timeout_s must be > 0."
            )

        if self.request_put_timeout_s <= 0:
            raise ValueError(
                "request_put_timeout_s must be > 0."
            )


@dataclass(slots=True)
class SelfPlayGameJob:
    game_id: int
    seed: int

    # Optional per-game override. V6 uses this to preserve V5's
    # deterministic whole-game train/validation split.
    split: Optional[str] = None

    extra_game_metadata: Optional[dict[str, Any]] = None


@dataclass(slots=True)
class SelfPlayWorkerShutdown:
    reason: str = "shutdown requested"


@dataclass(slots=True)
class SelfPlayWorkerReady:
    worker_id: int
    pid: int


@dataclass(slots=True)
class SelfPlayGameResult:
    success: bool
    worker_id: int
    pid: int
    game_id: int
    seed: int

    seconds: float

    num_positions: int = 0

    search_summary: Optional[dict[str, Any]] = None
    evaluator_stats: Optional[dict[str, Any]] = None

    # Metadata returned by ReplayGenerator.commit_game().
    # The main coordinator uses this together with samples to call
    # the persistent ReplayBuffer.add_game(...) API.
    game_metadata: Optional[dict[str, Any]] = None

    samples: Optional[list[Any]] = None

    error_type: Optional[str] = None
    error: Optional[str] = None
    traceback: Optional[str] = None


# ============================================================
# SEARCH / GENERATOR BUILDERS
# ============================================================


def make_multiprocess_mcts(
    *,
    evaluator,
    search_config,
):
    search_config.validate()

    return MCTS(
        simulations=(
            search_config.simulations
        ),
        rollout_type="neural",
        selection_type="puct",
        evaluator=evaluator,
        c_puct=(
            search_config.c_puct
        ),
        dirichlet_alpha=(
            search_config.dirichlet_alpha
        ),
        dirichlet_epsilon=(
            search_config.dirichlet_epsilon
        ),
        adaptive_simulations=True,
        min_simulations=(
            search_config.min_simulations
        ),
        check_interval=(
            search_config.check_interval
        ),
        target_visits_per_action=(
            search_config.target_visits_per_action
        ),
        single_action_simulations=(
            search_config.single_action_simulations
        ),
        stability_checks=(
            search_config.stability_checks
        ),
    )


def make_model_replay_generator(
    *,
    env,
    mcts,
    replay_buffer,
    max_game_steps,
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
        max_game_steps=(
            max_game_steps
        ),
    )


# ============================================================
# REPLAY / SEARCH TELEMETRY
# ============================================================


def replay_samples(
    replay_buffer,
):
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

    average_actual = (
        float(
            statistics.mean(
                actual_simulations
            )
        )
        if actual_simulations
        else None
    )

    median_actual = (
        float(
            statistics.median(
                actual_simulations
            )
        )
        if actual_simulations
        else None
    )

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

    def optional_mean(
        values,
    ):
        if not values:
            return None

        return float(
            statistics.mean(
                values
            )
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
            average_actual,

        "median_actual_simulations":
            median_actual,

        "simulation_savings_fraction":
            savings_fraction,

        "average_initial_root_visits":
            optional_mean(
                initial_root_visits
            ),

        "average_final_root_visits":
            optional_mean(
                final_root_visits
            ),

        "average_legal_actions":
            optional_mean(
                legal_counts
            ),

        "stop_reasons":
            stop_reasons,
    }


# ============================================================
# ONE REAL SELF-PLAY GAME
# ============================================================


def run_self_play_game(
    *,
    worker_id,
    evaluator,
    job,
    config,
):
    """
    Run one complete real ModelReplayGenerator self-play game.

    Everything except neural inference is private to this process.
    """

    config.validate()

    env = SplendorEnv()

    mcts = make_multiprocess_mcts(
        evaluator=evaluator,
        search_config=(
            config.search
        ),
    )

    scratch_replay = ReplayBuffer(
        capacity=(
            config.replay_capacity
        )
    )

    generator = (
        make_model_replay_generator(
            env=env,
            mcts=mcts,
            replay_buffer=(
                scratch_replay
            ),
            max_game_steps=(
                config.search.max_game_steps
            ),
        )
    )

    started = (
        time.perf_counter()
    )

    try:
        metadata = {
            "multiprocess_self_play":
                True,

            "worker_id":
                int(
                    worker_id
                ),

            "worker_pid":
                int(
                    os.getpid()
                ),
        }

        if job.extra_game_metadata:
            metadata.update(
                job.extra_game_metadata
            )

        game_split = (
            job.split
            if job.split is not None
            else config.split
        )

        generator_result = (
            generator.generate_game(
                seed=int(
                    job.seed
                ),
                split=(
                    game_split
                ),
                model_generation=(
                    config.model_generation
                ),
                model_checkpoint=(
                    config.model_checkpoint_label
                ),
                extra_game_metadata=(
                    metadata
                ),
            )
        )

        elapsed = (
            time.perf_counter()
            - started
        )

        samples = replay_samples(
            scratch_replay
        )

        search_summary = (
            summarize_search_samples(
                scratch_replay
            )
        )

        num_positions = int(
            generator_result.get(
                "num_positions",
                len(samples),
            )
        )

        return SelfPlayGameResult(
            success=True,
            worker_id=int(
                worker_id
            ),
            pid=int(
                os.getpid()
            ),
            game_id=int(
                job.game_id
            ),
            seed=int(
                job.seed
            ),
            seconds=float(
                elapsed
            ),
            num_positions=(
                num_positions
            ),
            search_summary=(
                search_summary
            ),
            evaluator_stats=(
                evaluator.stats_snapshot()
            ),
            game_metadata=(
                generator_result.get(
                    "game_metadata"
                )
            ),
            samples=(
                samples
                if config.return_samples
                else None
            ),
        )

    except BaseException as exc:
        elapsed = (
            time.perf_counter()
            - started
        )

        return SelfPlayGameResult(
            success=False,
            worker_id=int(
                worker_id
            ),
            pid=int(
                os.getpid()
            ),
            game_id=int(
                job.game_id
            ),
            seed=int(
                job.seed
            ),
            seconds=float(
                elapsed
            ),
            evaluator_stats=(
                evaluator.stats_snapshot()
            ),
            error_type=(
                type(exc).__name__
            ),
            error=str(
                exc
            ),
            traceback=(
                traceback.format_exc()
            ),
        )


# ============================================================
# LONG-LIVED WORKER PROCESS
# ============================================================


def self_play_worker_main(
    *,
    worker_id,
    job_queue,
    result_queue,
    inference_request_queue,
    inference_response_queue,
    config,
):
    """
    Long-lived worker process.

    The worker creates exactly one MultiprocessNeuralEvaluator and
    then processes game jobs sequentially until shutdown.

    Multiple worker processes execute this function concurrently,
    giving true CPU parallelism across independent games.
    """

    config.validate()

    evaluator = (
        MultiprocessNeuralEvaluator(
            worker_id=int(
                worker_id
            ),
            request_queue=(
                inference_request_queue
            ),
            response_queue=(
                inference_response_queue
            ),
            request_timeout_s=(
                config.request_timeout_s
            ),
            request_put_timeout_s=(
                config.request_put_timeout_s
            ),
            return_policy_device="cpu",
        )
    )

    result_queue.put(
        SelfPlayWorkerReady(
            worker_id=int(
                worker_id
            ),
            pid=int(
                os.getpid()
            ),
        )
    )

    while True:
        job = job_queue.get()

        if isinstance(
            job,
            SelfPlayWorkerShutdown,
        ):
            break

        if not isinstance(
            job,
            SelfPlayGameJob,
        ):
            continue

        result = run_self_play_game(
            worker_id=(
                worker_id
            ),
            evaluator=evaluator,
            job=job,
            config=config,
        )

        result_queue.put(
            result
        )

    evaluator.close()


# ============================================================
# PARENT-SIDE WORKER POOL HELPER
# ============================================================


class MultiprocessSelfPlayPool:
    """
    Parent-side helper for starting/stopping CPU self-play workers.

    The GPU inference server is intentionally NOT owned here.
    The coordinator should start one GPUInferenceServerProcess first,
    then pass its request/response queues to this pool.
    """

    def __init__(
        self,
        *,
        num_workers,
        inference_ipc,
        config,
        mp_context,
    ):
        if num_workers < 1:
            raise ValueError(
                "num_workers must be >= 1."
            )

        if (
            len(
                inference_ipc.response_queues
            )
            < num_workers
        ):
            raise ValueError(
                "InferenceIPC has fewer response "
                "queues than self-play workers."
            )

        config.validate()

        self.num_workers = int(
            num_workers
        )

        self.inference_ipc = (
            inference_ipc
        )

        self.config = config
        self.mp_context = (
            mp_context
        )

        self.job_queue = (
            mp_context.Queue()
        )

        self.result_queue = (
            mp_context.Queue()
        )

        self.processes = []

    def start(self):
        if self.processes:
            raise RuntimeError(
                "Worker pool already started."
            )

        for worker_id in range(
            self.num_workers
        ):
            process = (
                self.mp_context.Process(
                    target=(
                        self_play_worker_main
                    ),
                    kwargs={
                        "worker_id":
                            worker_id,

                        "job_queue":
                            self.job_queue,

                        "result_queue":
                            self.result_queue,

                        "inference_request_queue":
                            (
                                self.inference_ipc
                                .request_queue
                            ),

                        "inference_response_queue":
                            (
                                self.inference_ipc
                                .response_queues[
                                    worker_id
                                ]
                            ),

                        "config":
                            self.config,
                    },
                    name=(
                        "SplendorSelfPlayWorker"
                        f"-{worker_id}"
                    ),
                    daemon=False,
                )
            )

            process.start()

            self.processes.append(
                process
            )

    def submit(
        self,
        job,
    ):
        if not isinstance(
            job,
            SelfPlayGameJob,
        ):
            raise TypeError(
                "submit expects SelfPlayGameJob."
            )

        self.job_queue.put(
            job
        )

    def submit_many(
        self,
        jobs,
    ):
        for job in jobs:
            self.submit(
                job
            )

    def get_result(
        self,
        timeout=None,
    ):
        if timeout is None:
            return self.result_queue.get()

        return self.result_queue.get(
            timeout=timeout
        )

    def request_shutdown(
        self,
    ):
        for _ in range(
            self.num_workers
        ):
            self.job_queue.put(
                SelfPlayWorkerShutdown()
            )

    def close(
        self,
        *,
        timeout_s=30.0,
        terminate_if_needed=True,
    ):
        if not self.processes:
            return

        self.request_shutdown()

        deadline = (
            time.perf_counter()
            + float(
                timeout_s
            )
        )

        for process in self.processes:
            remaining = max(
                0.0,
                deadline
                - time.perf_counter(),
            )

            process.join(
                timeout=remaining
            )

        if terminate_if_needed:
            for process in self.processes:
                if process.is_alive():
                    process.terminate()

            for process in self.processes:
                process.join(
                    timeout=5.0
                )

    def worker_exitcodes(
        self,
    ):
        return [
            process.exitcode
            for process
            in self.processes
        ]
