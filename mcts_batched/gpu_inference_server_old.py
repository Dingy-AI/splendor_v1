"""
Centralized GPU inference server process for Splendor Model 4.

Architecture
------------

    Game/MCTS process 1 ----\
    Game/MCTS process 2 -----+--> request_queue
    Game/MCTS process N ----/         |
                                      v
                            GPUInferenceServer
                                      |
                              dynamic microbatch
                                      |
                                   Model 4
                                      |
                  +-------------------+-------------------+
                  v                   v                   v
          response_queue[0]   response_queue[1]   response_queue[N]

The GPU server is the ONLY process that owns Model 4 on CUDA.

Windows
-------
Create/start multiprocessing components under:

    if __name__ == "__main__":
        multiprocessing.freeze_support()
        ...

Model 4 and CUDA are initialized inside the child process.
"""

from __future__ import annotations

from dataclasses import dataclass
import multiprocessing as mp
import os
from pathlib import Path
from queue import Empty
import time
import traceback
from typing import Any, Optional

import numpy as np

from splendor_v1.mcts_batched.inference_protocol import (
    InferenceRequest,
    InferenceResponse,
    InferenceServerStatus,
    InferenceShutdown,
)


# ============================================================
# CONFIG
# ============================================================


@dataclass(frozen=True, slots=True)
class GPUInferenceServerConfig:
    checkpoint_path: str

    max_batch_size: int = 64
    batch_wait_ms: float = 0.5

    device: str = "cuda"

    # 0 disables periodic stats events.
    stats_report_interval_batches: int = 1000

    # None leaves torch's default unchanged.
    float32_matmul_precision: Optional[str] = None

    def validate(self):
        if self.max_batch_size < 1:
            raise ValueError(
                "max_batch_size must be >= 1."
            )

        if self.batch_wait_ms < 0:
            raise ValueError(
                "batch_wait_ms must be >= 0."
            )

        if self.stats_report_interval_batches < 0:
            raise ValueError(
                "stats_report_interval_batches must be >= 0."
            )

        if not self.checkpoint_path:
            raise ValueError(
                "checkpoint_path must not be empty."
            )


# ============================================================
# IPC
# ============================================================


@dataclass(slots=True)
class InferenceIPC:
    """
    Queue bundle shared by parent, workers, and GPU server.

    One response queue per game worker prevents workers from
    competing for a single shared response queue.
    """

    request_queue: Any
    response_queues: list[Any]
    status_queue: Any


def create_inference_ipc(
    num_workers,
    *,
    mp_context=None,
    request_queue_maxsize=0,
    response_queue_maxsize=0,
    status_queue_maxsize=0,
):
    if num_workers < 1:
        raise ValueError(
            "num_workers must be >= 1."
        )

    if mp_context is None:
        mp_context = mp.get_context(
            "spawn"
        )

    return InferenceIPC(
        request_queue=mp_context.Queue(
            maxsize=request_queue_maxsize
        ),
        response_queues=[
            mp_context.Queue(
                maxsize=response_queue_maxsize
            )
            for _ in range(
                num_workers
            )
        ],
        status_queue=mp_context.Queue(
            maxsize=status_queue_maxsize
        ),
    )


# ============================================================
# MODEL LOADING -- CHILD ONLY
# ============================================================


def _extract_model_state_dict(
    checkpoint,
):
    import torch

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


def _load_model_in_child(
    checkpoint_path,
    device_string,
    float32_matmul_precision,
):
    """
    Import torch/model and initialize CUDA only inside child.
    """

    import torch

    from splendor_v1.network.model_4_legal_scorer import (
        SplendorNetwork,
    )

    if float32_matmul_precision is not None:
        torch.set_float32_matmul_precision(
            float32_matmul_precision
        )

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

    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested but "
                "torch.cuda.is_available() is False."
            )

        torch.cuda.set_device(
            device
        )

    checkpoint_file = Path(
        checkpoint_path
    )

    if not checkpoint_file.exists():
        raise FileNotFoundError(
            "Inference checkpoint does not exist: "
            f"{checkpoint_file}"
        )

    # Load CPU first. Move the completed model to CUDA afterward.
    checkpoint = torch.load(
        checkpoint_file,
        map_location="cpu",
    )

    model = SplendorNetwork()

    model.load_state_dict(
        _extract_model_state_dict(
            checkpoint
        ),
        strict=True,
    )

    model.to(
        device
    )

    model.eval()

    return (
        model,
        device,
        torch,
    )


# ============================================================
# TELEMETRY
# ============================================================


class _ServerStats:
    def __init__(
        self,
        max_batch_size,
        batch_wait_ms,
    ):
        self.max_batch_size = int(
            max_batch_size
        )

        self.batch_wait_ms = float(
            batch_wait_ms
        )

        self.started_perf = (
            time.perf_counter()
        )

        self.requests_received = 0
        self.requests_completed = 0
        self.requests_failed = 0

        self.batches_completed = 0
        self.batch_items_total = 0
        self.max_observed_batch_size = 0

        self.collection_seconds_total = 0.0
        self.batch_build_seconds_total = 0.0
        self.inference_seconds_total = 0.0
        self.response_route_seconds_total = 0.0

        self.request_queue_delay_seconds_total = 0.0
        self.request_queue_delay_samples = 0

        self.batch_size_histogram = {}

    def record_request_received(
        self,
        request,
    ):
        self.requests_received += 1

        submitted_ns = int(
            getattr(
                request,
                "submitted_perf_counter_ns",
                0,
            )
            or 0
        )

        if submitted_ns > 0:
            delay_ns = max(
                0,
                (
                    time.perf_counter_ns()
                    - submitted_ns
                ),
            )

            self.request_queue_delay_seconds_total += (
                delay_ns
                / 1_000_000_000.0
            )

            self.request_queue_delay_samples += 1

    def record_batch(
        self,
        batch_size,
        build_seconds,
        inference_seconds,
        response_route_seconds,
    ):
        self.batches_completed += 1

        self.batch_items_total += int(
            batch_size
        )

        self.max_observed_batch_size = max(
            self.max_observed_batch_size,
            int(batch_size),
        )

        self.batch_build_seconds_total += float(
            build_seconds
        )

        self.inference_seconds_total += float(
            inference_seconds
        )

        self.response_route_seconds_total += float(
            response_route_seconds
        )

        key = str(
            int(batch_size)
        )

        self.batch_size_histogram[key] = (
            self.batch_size_histogram.get(
                key,
                0,
            )
            + 1
        )

    def snapshot(self):
        batches = int(
            self.batches_completed
        )

        completed = int(
            self.requests_completed
        )

        uptime = max(
            0.0,
            time.perf_counter()
            - self.started_perf,
        )

        average_batch_size = (
            self.batch_items_total
            / batches
            if batches > 0
            else 0.0
        )

        return {
            "requests_received":
                int(
                    self.requests_received
                ),

            "requests_completed":
                completed,

            "requests_failed":
                int(
                    self.requests_failed
                ),

            "batches_completed":
                batches,

            "average_batch_size":
                float(
                    average_batch_size
                ),

            "max_observed_batch_size":
                int(
                    self.max_observed_batch_size
                ),

            "configured_max_batch_size":
                int(
                    self.max_batch_size
                ),

            "batch_fill_fraction":
                float(
                    average_batch_size
                    / self.max_batch_size
                    if self.max_batch_size > 0
                    else 0.0
                ),

            "configured_batch_wait_ms":
                float(
                    self.batch_wait_ms
                ),

            "average_collection_wait_ms":
                float(
                    self.collection_seconds_total
                    / batches
                    * 1000.0
                    if batches > 0
                    else 0.0
                ),

            "average_request_queue_delay_ms":
                float(
                    self.request_queue_delay_seconds_total
                    / self.request_queue_delay_samples
                    * 1000.0
                    if self.request_queue_delay_samples > 0
                    else 0.0
                ),

            "average_batch_build_ms":
                float(
                    self.batch_build_seconds_total
                    / batches
                    * 1000.0
                    if batches > 0
                    else 0.0
                ),

            "average_batch_inference_ms":
                float(
                    self.inference_seconds_total
                    / batches
                    * 1000.0
                    if batches > 0
                    else 0.0
                ),

            "average_response_route_ms":
                float(
                    self.response_route_seconds_total
                    / batches
                    * 1000.0
                    if batches > 0
                    else 0.0
                ),

            "total_batch_inference_seconds":
                float(
                    self.inference_seconds_total
                ),

            "inference_positions_per_second":
                float(
                    completed
                    / self.inference_seconds_total
                    if self.inference_seconds_total > 0
                    else 0.0
                ),

            "server_uptime_seconds":
                float(
                    uptime
                ),

            "end_to_end_positions_per_second":
                float(
                    completed / uptime
                    if uptime > 0
                    else 0.0
                ),

            "batch_size_histogram":
                dict(
                    self.batch_size_histogram
                ),
        }


# ============================================================
# STATUS
# ============================================================


def _emit_status(
    status_queue,
    kind,
    *,
    message="",
    stats=None,
):
    try:
        status_queue.put_nowait(
            InferenceServerStatus(
                kind=kind,
                pid=os.getpid(),
                message=message,
                stats=stats,
            )
        )

    except Exception:
        # Telemetry should never crash inference.
        pass


# ============================================================
# VALIDATION / ERROR ROUTING
# ============================================================


def _validate_request(
    request,
    num_workers,
):
    worker_id = int(
        request.worker_id
    )

    if not (
        0 <= worker_id
        < num_workers
    ):
        raise ValueError(
            f"worker_id {worker_id} outside "
            f"[0, {num_workers - 1}]."
        )

    observation = np.asarray(
        request.observation
    )

    if observation.ndim != 1:
        raise ValueError(
            "observation must be 1D, "
            f"got {observation.shape}."
        )

    legal_action_ids = [
        int(action_id)
        for action_id
        in request.legal_action_ids
    ]

    if not legal_action_ids:
        raise ValueError(
            "Request contains no legal actions."
        )

    return (
        worker_id,
        observation,
        legal_action_ids,
    )


def _send_error_response(
    response_queues,
    request,
    error_text,
):
    worker_id = int(
        getattr(
            request,
            "worker_id",
            -1,
        )
    )

    if not (
        0 <= worker_id
        < len(response_queues)
    ):
        return

    response_queues[
        worker_id
    ].put(
        InferenceResponse(
            worker_id=worker_id,
            request_id=int(
                getattr(
                    request,
                    "request_id",
                    -1,
                )
            ),
            error=error_text,
        )
    )


# ============================================================
# DYNAMIC BATCH COLLECTION
# ============================================================


def _collect_batch(
    *,
    first_request,
    request_queue,
    max_batch_size,
    batch_wait_seconds,
    stats,
):
    batch = [
        first_request
    ]

    stats.record_request_received(
        first_request
    )

    saw_shutdown = False

    if len(batch) >= max_batch_size:
        return (
            batch,
            saw_shutdown,
        )

    if batch_wait_seconds <= 0:
        while len(batch) < max_batch_size:
            try:
                item = (
                    request_queue
                    .get_nowait()
                )

            except Empty:
                break

            if isinstance(
                item,
                InferenceShutdown,
            ):
                saw_shutdown = True
                break

            if isinstance(
                item,
                InferenceRequest,
            ):
                stats.record_request_received(
                    item
                )

                batch.append(
                    item
                )

        return (
            batch,
            saw_shutdown,
        )

    deadline = (
        time.perf_counter()
        + batch_wait_seconds
    )

    while len(batch) < max_batch_size:
        remaining = (
            deadline
            - time.perf_counter()
        )

        if remaining <= 0:
            break

        try:
            item = request_queue.get(
                timeout=remaining
            )

        except Empty:
            break

        if isinstance(
            item,
            InferenceShutdown,
        ):
            saw_shutdown = True
            break

        if isinstance(
            item,
            InferenceRequest,
        ):
            stats.record_request_received(
                item
            )

            batch.append(
                item
            )

    return (
        batch,
        saw_shutdown,
    )


# ============================================================
# ONE MODEL BATCH
# ============================================================


def _process_batch(
    *,
    batch,
    model,
    device,
    torch,
    response_queues,
    stats,
):
    valid = []

    # Bad requests fail individually instead of poisoning the batch.
    for request in batch:
        try:
            (
                worker_id,
                observation,
                legal_action_ids,
            ) = _validate_request(
                request,
                len(response_queues),
            )

            valid.append(
                (
                    request,
                    worker_id,
                    observation,
                    legal_action_ids,
                )
            )

        except Exception as exc:
            stats.requests_failed += 1

            _send_error_response(
                response_queues,
                request,
                (
                    f"{type(exc).__name__}: "
                    f"{exc}"
                ),
            )

    if not valid:
        return False

    build_start = (
        time.perf_counter()
    )

    observations_np = np.stack(
        [
            item[2]
            for item in valid
        ],
        axis=0,
    )

    model_weight = (
        model.action_embedding.weight
    )

    observations = torch.as_tensor(
        observations_np,
        dtype=model_weight.dtype,
        device=device,
    )

    max_legal_actions = max(
        len(
            item[3]
        )
        for item in valid
    )

    # Padded positions are masked out before scoring. Use action ID
    # 0 for the unused slots, matching the already-validated
    # threaded BatchedNeuralEvaluator implementation.
    legal_action_ids = torch.zeros(
        (
            len(valid),
            max_legal_actions,
        ),
        dtype=torch.long,
        device=device,
    )

    legal_action_mask = torch.zeros(
        (
            len(valid),
            max_legal_actions,
        ),
        dtype=torch.bool,
        device=device,
    )

    legal_counts = []

    for row, item in enumerate(
        valid
    ):
        ids = item[3]

        count = len(
            ids
        )

        legal_counts.append(
            count
        )

        legal_action_ids[
            row,
            :count,
        ] = torch.as_tensor(
            ids,
            dtype=torch.long,
            device=device,
        )

        legal_action_mask[
            row,
            :count,
        ] = True

    build_seconds = (
        time.perf_counter()
        - build_start
    )

    # --------------------------------------------------------
    # One GPU forward
    # --------------------------------------------------------

    if device.type == "cuda":
        torch.cuda.synchronize(
            device
        )

    inference_start = (
        time.perf_counter()
    )

    with torch.inference_mode():
        (
            legal_logits,
            wdl_logits,
        ) = model.forward_legal(
            observations,
            legal_action_ids,
            legal_action_mask,
        )

        wdl_probs = torch.softmax(
            wdl_logits,
            dim=-1,
        )

        # WDL order = LOSS, DRAW, WIN.
        values = (
            wdl_probs[..., 2]
            - wdl_probs[..., 0]
        )

    if device.type == "cuda":
        torch.cuda.synchronize(
            device
        )

    inference_seconds = (
        time.perf_counter()
        - inference_start
    )

    # One CPU transfer per batched output tensor.
    legal_logits_cpu = (
        legal_logits
        .detach()
        .float()
        .cpu()
    )

    values_cpu = (
        values
        .detach()
        .float()
        .cpu()
    )

    # --------------------------------------------------------
    # Route one result to each worker
    # --------------------------------------------------------

    route_start = (
        time.perf_counter()
    )

    for row, item in enumerate(
        valid
    ):
        (
            request,
            worker_id,
            _observation,
            _ids,
        ) = item

        count = legal_counts[
            row
        ]

        legal_probs = (
            torch.softmax(
                legal_logits_cpu[
                    row,
                    :count,
                ],
                dim=0,
            )
            .numpy()
            .astype(
                np.float32,
                copy=False,
            )
        )

        value = float(
            values_cpu[
                row
            ].item()
        )

        response_queues[
            worker_id
        ].put(
            InferenceResponse(
                worker_id=worker_id,
                request_id=int(
                    request.request_id
                ),
                legal_probs=(
                    legal_probs
                ),
                value=value,
                error=None,
            )
        )

        stats.requests_completed += 1

    response_route_seconds = (
        time.perf_counter()
        - route_start
    )

    stats.record_batch(
        batch_size=len(valid),
        build_seconds=(
            build_seconds
        ),
        inference_seconds=(
            inference_seconds
        ),
        response_route_seconds=(
            response_route_seconds
        ),
    )

    return True


# ============================================================
# CHILD PROCESS ENTRY POINT
# ============================================================


def _gpu_inference_server_main(
    config,
    request_queue,
    response_queues,
    status_queue,
):
    """
    Top-level target function required for Windows spawn.
    """

    config.validate()

    stats = _ServerStats(
        max_batch_size=(
            config.max_batch_size
        ),
        batch_wait_ms=(
            config.batch_wait_ms
        ),
    )

    _emit_status(
        status_queue,
        "starting",
        message=(
            "GPU inference server process started."
        ),
    )

    try:
        (
            model,
            device,
            torch,
        ) = _load_model_in_child(
            checkpoint_path=(
                config.checkpoint_path
            ),
            device_string=(
                config.device
            ),
            float32_matmul_precision=(
                config.float32_matmul_precision
            ),
        )

        if device.type == "cuda":
            torch.cuda.synchronize(
                device
            )

        _emit_status(
            status_queue,
            "ready",
            message=(
                "Model 4 loaded and server is ready."
            ),
            stats=stats.snapshot(),
        )

        batch_wait_seconds = (
            config.batch_wait_ms
            / 1000.0
        )

        shutdown_requested = False

        while not shutdown_requested:
            item = (
                request_queue.get()
            )

            if isinstance(
                item,
                InferenceShutdown,
            ):
                break

            if not isinstance(
                item,
                InferenceRequest,
            ):
                continue

            collection_start = (
                time.perf_counter()
            )

            (
                batch,
                saw_shutdown,
            ) = _collect_batch(
                first_request=item,
                request_queue=(
                    request_queue
                ),
                max_batch_size=(
                    config.max_batch_size
                ),
                batch_wait_seconds=(
                    batch_wait_seconds
                ),
                stats=stats,
            )

            collection_seconds = (
                time.perf_counter()
                - collection_start
            )

            batches_before = (
                stats.batches_completed
            )

            try:
                batch_recorded = (
                    _process_batch(
                        batch=batch,
                        model=model,
                        device=device,
                        torch=torch,
                        response_queues=(
                            response_queues
                        ),
                        stats=stats,
                    )
                )

            except Exception as exc:
                batch_recorded = False

                error_text = (
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )

                for request in batch:
                    stats.requests_failed += 1

                    _send_error_response(
                        response_queues,
                        request,
                        error_text,
                    )

            if (
                batch_recorded
                and stats.batches_completed
                > batches_before
            ):
                stats.collection_seconds_total += (
                    collection_seconds
                )

            shutdown_requested = (
                saw_shutdown
            )

            interval = int(
                config.stats_report_interval_batches
            )

            if (
                interval > 0
                and stats.batches_completed > 0
                and (
                    stats.batches_completed
                    % interval
                    == 0
                )
            ):
                _emit_status(
                    status_queue,
                    "stats",
                    message=(
                        "Periodic inference telemetry."
                    ),
                    stats=stats.snapshot(),
                )

        _emit_status(
            status_queue,
            "stopped",
            message=(
                "GPU inference server stopped cleanly."
            ),
            stats=stats.snapshot(),
        )

    except BaseException as exc:
        _emit_status(
            status_queue,
            "fatal_error",
            message=(
                f"{type(exc).__name__}: "
                f"{exc}\n"
                f"{traceback.format_exc()}"
            ),
            stats=stats.snapshot(),
        )

        raise


# ============================================================
# PARENT-SIDE LIFECYCLE WRAPPER
# ============================================================


class GPUInferenceServerProcess:
    """
    Parent-side lifecycle wrapper.

    This object does not load Model 4 or initialize CUDA in the
    parent process.
    """

    def __init__(
        self,
        *,
        config,
        ipc,
        mp_context=None,
        process_name=(
            "SplendorGPUInferenceServer"
        ),
    ):
        config.validate()

        if not ipc.response_queues:
            raise ValueError(
                "At least one response queue is required."
            )

        if mp_context is None:
            mp_context = mp.get_context(
                "spawn"
            )

        self.config = config
        self.ipc = ipc
        self.mp_context = mp_context
        self.process_name = str(
            process_name
        )

        self._process = None
        self._ready_status = None
        self._final_status = None

    @property
    def pid(self):
        if self._process is None:
            return None

        return self._process.pid

    @property
    def exitcode(self):
        if self._process is None:
            return None

        return self._process.exitcode

    def is_alive(self):
        return bool(
            self._process is not None
            and self._process.is_alive()
        )

    def start(self):
        if (
            self._process is not None
            and self._process.is_alive()
        ):
            return

        if self._process is not None:
            raise RuntimeError(
                "Server instances cannot be restarted."
            )

        self._process = (
            self.mp_context.Process(
                target=(
                    _gpu_inference_server_main
                ),
                args=(
                    self.config,
                    self.ipc.request_queue,
                    self.ipc.response_queues,
                    self.ipc.status_queue,
                ),
                name=self.process_name,
                daemon=False,
            )
        )

        self._process.start()

    def wait_until_ready(
        self,
        timeout_s=120.0,
    ):
        if self._process is None:
            raise RuntimeError(
                "Server has not been started."
            )

        deadline = (
            time.perf_counter()
            + float(timeout_s)
        )

        while True:
            if (
                self._process.exitcode
                is not None
                and not self._process.is_alive()
            ):
                raise RuntimeError(
                    "GPU inference server exited "
                    "before READY. "
                    f"exitcode={self._process.exitcode}"
                )

            remaining = (
                deadline
                - time.perf_counter()
            )

            if remaining <= 0:
                raise TimeoutError(
                    "Timed out waiting for "
                    "GPU inference server READY."
                )

            try:
                status = (
                    self.ipc.status_queue.get(
                        timeout=min(
                            remaining,
                            0.25,
                        )
                    )
                )

            except Empty:
                continue

            if not isinstance(
                status,
                InferenceServerStatus,
            ):
                continue

            if status.kind == "ready":
                self._ready_status = (
                    status
                )

                return status

            if status.kind == "fatal_error":
                self._final_status = (
                    status
                )

                raise RuntimeError(
                    "GPU inference server failed "
                    "during startup:\n"
                    f"{status.message}"
                )

    def request_shutdown(
        self,
        reason="parent shutdown",
    ):
        if (
            self._process is None
            or not self._process.is_alive()
        ):
            return

        self.ipc.request_queue.put(
            InferenceShutdown(
                reason=str(
                    reason
                )
            )
        )

    def join(
        self,
        timeout_s=None,
    ):
        if self._process is None:
            return None

        self._process.join(
            timeout=timeout_s
        )

        return self._process.exitcode

    def close(
        self,
        *,
        timeout_s=30.0,
        terminate_if_needed=True,
    ):
        if self._process is None:
            return

        if self._process.is_alive():
            self.request_shutdown()

            self._process.join(
                timeout=timeout_s
            )

        if (
            self._process.is_alive()
            and terminate_if_needed
        ):
            self._process.terminate()

            self._process.join(
                timeout=5.0
            )

        self._drain_final_status()

    def _drain_final_status(self):
        while True:
            try:
                status = (
                    self.ipc.status_queue
                    .get_nowait()
                )

            except Empty:
                break

            if isinstance(
                status,
                InferenceServerStatus,
            ):
                if status.kind in (
                    "stopped",
                    "fatal_error",
                ):
                    self._final_status = (
                        status
                    )

    def drain_status_events(self):
        events = []

        while True:
            try:
                item = (
                    self.ipc.status_queue
                    .get_nowait()
                )

            except Empty:
                break

            if isinstance(
                item,
                InferenceServerStatus,
            ):
                events.append(
                    item
                )

                if item.kind in (
                    "stopped",
                    "fatal_error",
                ):
                    self._final_status = (
                        item
                    )

        return events

    @property
    def ready_status(self):
        return self._ready_status

    @property
    def final_status(self):
        self._drain_final_status()

        return self._final_status

    @property
    def final_stats(self):
        status = self.final_status

        if status is None:
            return None

        return status.stats

    def __enter__(self):
        self.start()
        self.wait_until_ready()

        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback_object,
    ):
        self.close()
