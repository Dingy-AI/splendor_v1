"""
Dynamic batched neural evaluator for Model 4.

This evaluator preserves the same public contract as
DirectNeuralEvaluator:

    evaluate(env, state, legal_actions=None)
        -> (legal_probs, value_number)

The difference is that inference is not executed immediately.
Requests from multiple callers are placed into a shared queue.
A single background worker collects a dynamic microbatch, performs
one Model 4 forward pass, and routes each result back to the caller.

This is the first batching layer. It is intended for multiple
concurrent callers within one Python process. Later, the same public
contract can be placed behind a multiprocessing client/server layer
for true cross-process self-play.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from queue import Empty, Queue
import threading
import time
from typing import Any, Optional

import numpy as np
import torch

from splendor_v1.mcts_batched.direct_neural_evaluator import (
    wdl_logits_to_value,
)


# ============================================================
# INTERNAL REQUEST OBJECT
# ============================================================


@dataclass
class _InferenceRequest:
    """
    One pending Model 4 inference request.

    The caller owns:
        observation
        legal_action_ids

    The batch worker fills:
        legal_probs
        value_number
        exception

    completion_event wakes the waiting caller.
    """

    observation: np.ndarray
    legal_action_ids: list[int]

    completion_event: threading.Event = field(
        default_factory=threading.Event
    )

    legal_probs: Optional[torch.Tensor] = None
    value_number: Optional[float] = None
    exception: Optional[BaseException] = None


# Unique sentinel used to stop the worker.
_STOP = object()


# ============================================================
# BATCHED MODEL 4 EVALUATOR
# ============================================================


class BatchedNeuralEvaluator:
    """
    Dynamic in-process neural batching for Model 4.

    Multiple callers may invoke ``evaluate(...)`` concurrently.

    Each call:

        1. encodes the state
        2. converts legal actions to canonical IDs
        3. submits a request to the shared queue
        4. waits for its own result

    The background inference worker:

        1. waits for the first request
        2. gathers more requests up to ``max_batch_size``
        3. stops gathering when ``batch_wait_ms`` expires
        4. pads legal-action candidate IDs
        5. performs one batched Model 4 forward pass
        6. routes each policy/value result to the right caller

    Important
    ---------
    This class is thread-safe for concurrent callers in the SAME
    Python process.

    It is not yet a multiprocessing IPC evaluator. That will be a
    separate layer built on the same evaluate(...) contract.

    Parameters
    ----------
    model:
        Model 4 instance already moved to the desired device.

    max_batch_size:
        Maximum number of requests combined into one GPU forward.

    batch_wait_ms:
        Maximum amount of time after the first request arrives that
        the worker waits for more requests.

        A small value such as 0.25-1.0 ms is appropriate for dynamic
        microbatching.

    request_timeout_s:
        Optional maximum time a caller waits for its response.
        None means wait indefinitely.

    start_worker:
        Start the background inference thread immediately.
    """

    def __init__(
        self,
        model,
        max_batch_size=8,
        batch_wait_ms=0.5,
        request_timeout_s=None,
        start_worker=True,
    ):
        if max_batch_size < 1:
            raise ValueError(
                "max_batch_size must be >= 1."
            )

        if batch_wait_ms < 0:
            raise ValueError(
                "batch_wait_ms must be >= 0."
            )

        if (
            request_timeout_s is not None
            and request_timeout_s <= 0
        ):
            raise ValueError(
                "request_timeout_s must be > 0 "
                "or None."
            )

        self.model = model

        self.max_batch_size = int(
            max_batch_size
        )

        self.batch_wait_ms = float(
            batch_wait_ms
        )

        self.batch_wait_seconds = (
            self.batch_wait_ms / 1000.0
        )

        self.request_timeout_s = (
            request_timeout_s
        )

        self._request_queue = Queue()

        self._worker_thread = None

        self._lifecycle_lock = (
            threading.Lock()
        )

        self._closed = False

        # ----------------------------------------------------
        # Telemetry
        # ----------------------------------------------------

        self._stats_lock = (
            threading.Lock()
        )

        self._requests_submitted = 0
        self._requests_completed = 0
        self._batches_completed = 0
        self._batch_items_total = 0
        self._max_observed_batch_size = 0

        self._queue_wait_seconds_total = 0.0
        self._batch_inference_seconds_total = 0.0
        self._batch_build_seconds_total = 0.0

        if start_worker:
            self.start()

    # --------------------------------------------------------
    # Model metadata
    # --------------------------------------------------------

    @property
    def device(self):
        """
        Device used by Model 4.
        """

        return (
            self.model
            .action_embedding
            .weight
            .device
        )

    @property
    def dtype(self):
        """
        Floating-point dtype used by Model 4.
        """

        return (
            self.model
            .action_embedding
            .weight
            .dtype
        )

    # --------------------------------------------------------
    # Lifecycle
    # --------------------------------------------------------

    def start(self):
        """
        Start the single background inference worker.
        """

        with self._lifecycle_lock:

            if self._closed:
                raise RuntimeError(
                    "Cannot restart a closed "
                    "BatchedNeuralEvaluator."
                )

            if (
                self._worker_thread
                is not None
                and self._worker_thread.is_alive()
            ):
                return

            self._worker_thread = (
                threading.Thread(
                    target=self._worker_loop,
                    name=(
                        "BatchedNeuralEvaluator"
                        "-InferenceWorker"
                    ),
                    daemon=True,
                )
            )

            self._worker_thread.start()

    def close(
        self,
        join_timeout_s=5.0,
    ):
        """
        Stop the inference worker.

        Existing requests already taken by the worker are allowed to
        finish. Requests still waiting in the queue are failed during
        shutdown.
        """

        with self._lifecycle_lock:

            if self._closed:
                return

            self._closed = True

            self._request_queue.put(
                _STOP
            )

            worker = (
                self._worker_thread
            )

        if (
            worker is not None
            and worker.is_alive()
            and threading.current_thread()
            is not worker
        ):
            worker.join(
                timeout=join_timeout_s
            )

        self._fail_remaining_requests(
            RuntimeError(
                "BatchedNeuralEvaluator "
                "closed before request "
                "completion."
            )
        )

    def __enter__(self):
        self.start()
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ):
        self.close()

    # --------------------------------------------------------
    # Public evaluation API
    # --------------------------------------------------------

    def evaluate(
        self,
        env,
        state,
        legal_actions=None,
    ):
        """
        Queue one state for dynamically batched Model 4 inference.

        Returns
        -------
        (legal_probs, value_number)

        legal_probs:
            Tensor with shape [num_legal_actions].
            Ordering exactly matches ``legal_actions``.

        value_number:
            Python float in [-1, +1], computed as

                P(WIN) - P(LOSS)
        """

        if self._closed:
            raise RuntimeError(
                "BatchedNeuralEvaluator "
                "is closed."
            )

        if (
            self._worker_thread is None
            or not self._worker_thread.is_alive()
        ):
            self.start()

        if legal_actions is None:

            legal_actions = (
                env._legal_actions(
                    state
                )
            )

        if not legal_actions:

            raise ValueError(
                "BatchedNeuralEvaluator.evaluate "
                "received no legal actions."
            )

        observation = (
            env.observation_encoder.encoder(
                state
            )
        )

        observation = np.asarray(
            observation
        )

        legal_action_ids = [
            int(
                env.action_to_id(
                    action
                )
            )
            for action in legal_actions
        ]

        request = _InferenceRequest(
            observation=observation,
            legal_action_ids=(
                legal_action_ids
            ),
        )

        with self._stats_lock:
            self._requests_submitted += 1

        self._request_queue.put(
            request
        )

        completed = (
            request
            .completion_event
            .wait(
                timeout=(
                    self.request_timeout_s
                )
            )
        )

        if not completed:
            raise TimeoutError(
                "Timed out waiting for batched "
                "neural inference result."
            )

        if request.exception is not None:
            raise RuntimeError(
                "Batched neural inference failed."
            ) from request.exception

        if request.legal_probs is None:
            raise RuntimeError(
                "Batched neural inference "
                "completed without policy output."
            )

        if request.value_number is None:
            raise RuntimeError(
                "Batched neural inference "
                "completed without value output."
            )

        return (
            request.legal_probs,
            request.value_number,
        )

    def __call__(
        self,
        env,
        state,
        legal_actions=None,
    ):
        """
        Convenience alias for evaluate(...).
        """

        return self.evaluate(
            env=env,
            state=state,
            legal_actions=legal_actions,
        )

    # --------------------------------------------------------
    # Worker loop
    # --------------------------------------------------------

    def _worker_loop(self):
        """
        Own all normal model inference for this evaluator.
        """

        while True:

            first_item = (
                self._request_queue.get()
            )

            if first_item is _STOP:
                break

            if not isinstance(
                first_item,
                _InferenceRequest,
            ):
                continue

            batch = [
                first_item
            ]

            gather_start = (
                time.perf_counter()
            )

            self._collect_more_requests(
                batch
            )

            gather_elapsed = (
                time.perf_counter()
                - gather_start
            )

            try:
                self._process_batch(
                    batch
                )

            except BaseException as exc:

                for request in batch:
                    request.exception = exc
                    request.completion_event.set()

            finally:

                with self._stats_lock:

                    self._queue_wait_seconds_total += (
                        gather_elapsed
                    )

        # Shutdown: fail anything that was queued behind the sentinel.
        self._fail_remaining_requests(
            RuntimeError(
                "BatchedNeuralEvaluator "
                "worker stopped."
            )
        )

    def _collect_more_requests(
        self,
        batch,
    ):
        """
        Dynamically fill one microbatch.

        After the first request arrives, gather until either:

            - max_batch_size is reached
            - batch_wait_ms expires
            - queue is currently empty when wait_ms == 0
        """

        if (
            len(batch)
            >= self.max_batch_size
        ):
            return

        if self.batch_wait_seconds <= 0:

            while (
                len(batch)
                < self.max_batch_size
            ):
                try:
                    item = (
                        self._request_queue
                        .get_nowait()
                    )

                except Empty:
                    break

                if item is _STOP:
                    # Reinsert the stop marker so the outer loop
                    # handles shutdown after this batch finishes.
                    self._request_queue.put(
                        _STOP
                    )
                    break

                if isinstance(
                    item,
                    _InferenceRequest,
                ):
                    batch.append(
                        item
                    )

            return

        deadline = (
            time.perf_counter()
            + self.batch_wait_seconds
        )

        while (
            len(batch)
            < self.max_batch_size
        ):
            remaining = (
                deadline
                - time.perf_counter()
            )

            if remaining <= 0:
                break

            try:
                item = (
                    self._request_queue.get(
                        timeout=remaining
                    )
                )

            except Empty:
                break

            if item is _STOP:
                self._request_queue.put(
                    _STOP
                )
                break

            if isinstance(
                item,
                _InferenceRequest,
            ):
                batch.append(
                    item
                )

    # --------------------------------------------------------
    # Batch construction / inference
    # --------------------------------------------------------

    def _process_batch(
        self,
        batch,
    ):
        """
        Build padded Model 4 tensors, run one GPU forward, and
        distribute each result.
        """

        if not batch:
            return

        build_start = (
            time.perf_counter()
        )

        batch_size = len(
            batch
        )

        max_legal_actions = max(
            len(
                request.legal_action_ids
            )
            for request in batch
        )

        if max_legal_actions < 1:
            raise RuntimeError(
                "Cannot build an inference batch "
                "with zero legal actions."
            )

        observations_np = np.stack(
            [
                request.observation
                for request in batch
            ],
            axis=0,
        )

        observations = torch.as_tensor(
            observations_np,
            dtype=self.dtype,
            device=self.device,
        )

        legal_action_ids = torch.zeros(
            (
                batch_size,
                max_legal_actions,
            ),
            dtype=torch.long,
            device=self.device,
        )

        legal_action_mask = torch.zeros(
            (
                batch_size,
                max_legal_actions,
            ),
            dtype=torch.bool,
            device=self.device,
        )

        legal_counts = []

        for row, request in enumerate(
            batch
        ):
            count = len(
                request.legal_action_ids
            )

            legal_counts.append(
                count
            )

            ids = torch.as_tensor(
                request.legal_action_ids,
                dtype=torch.long,
                device=self.device,
            )

            legal_action_ids[
                row,
                :count,
            ] = ids

            legal_action_mask[
                row,
                :count,
            ] = True

        build_elapsed = (
            time.perf_counter()
            - build_start
        )

        inference_start = (
            time.perf_counter()
        )

        with torch.inference_mode():

            (
                legal_logits,
                wdl_logits,
            ) = self.model(
                observations,
                legal_action_ids,
                legal_action_mask,
            )

            values = (
                wdl_logits_to_value(
                    wdl_logits
                )
            )

        # Synchronize here so batch telemetry measures completed
        # GPU inference rather than only kernel launch time.
        if self.device.type == "cuda":
            torch.cuda.synchronize(
                self.device
            )

        inference_elapsed = (
            time.perf_counter()
            - inference_start
        )

        # ----------------------------------------------------
        # Route results
        # ----------------------------------------------------

        for row, request in enumerate(
            batch
        ):
            count = legal_counts[
                row
            ]

            row_logits = (
                legal_logits[
                    row,
                    :count,
                ]
            )

            legal_probs = torch.softmax(
                row_logits,
                dim=0,
            )

            value_number = float(
                values[row].item()
            )

            request.legal_probs = (
                legal_probs
            )

            request.value_number = (
                value_number
            )

            request.completion_event.set()

        with self._stats_lock:

            self._requests_completed += (
                batch_size
            )

            self._batches_completed += 1

            self._batch_items_total += (
                batch_size
            )

            self._max_observed_batch_size = max(
                self._max_observed_batch_size,
                batch_size,
            )

            self._batch_build_seconds_total += (
                build_elapsed
            )

            self._batch_inference_seconds_total += (
                inference_elapsed
            )

    # --------------------------------------------------------
    # Shutdown helpers
    # --------------------------------------------------------

    def _fail_remaining_requests(
        self,
        exception,
    ):
        while True:

            try:
                item = (
                    self._request_queue
                    .get_nowait()
                )

            except Empty:
                break

            if isinstance(
                item,
                _InferenceRequest,
            ):
                item.exception = exception
                item.completion_event.set()

    # --------------------------------------------------------
    # Telemetry
    # --------------------------------------------------------

    def stats_snapshot(self):
        """
        Return a thread-safe snapshot of batching telemetry.
        """

        with self._stats_lock:

            batches = (
                self._batches_completed
            )

            completed = (
                self._requests_completed
            )

            average_batch_size = (
                self._batch_items_total
                / batches
                if batches > 0
                else 0.0
            )

            batch_fill_fraction = (
                average_batch_size
                / self.max_batch_size
                if self.max_batch_size > 0
                else 0.0
            )

            average_batch_inference_ms = (
                (
                    self._batch_inference_seconds_total
                    / batches
                    * 1000.0
                )
                if batches > 0
                else 0.0
            )

            average_batch_build_ms = (
                (
                    self._batch_build_seconds_total
                    / batches
                    * 1000.0
                )
                if batches > 0
                else 0.0
            )

            average_collection_wait_ms = (
                (
                    self._queue_wait_seconds_total
                    / batches
                    * 1000.0
                )
                if batches > 0
                else 0.0
            )

            positions_per_second = (
                (
                    completed
                    / self._batch_inference_seconds_total
                )
                if (
                    self._batch_inference_seconds_total
                    > 0
                )
                else 0.0
            )

            return {
                "requests_submitted":
                    int(
                        self._requests_submitted
                    ),

                "requests_completed":
                    int(
                        self._requests_completed
                    ),

                "batches_completed":
                    int(
                        self._batches_completed
                    ),

                "average_batch_size":
                    float(
                        average_batch_size
                    ),

                "max_observed_batch_size":
                    int(
                        self._max_observed_batch_size
                    ),

                "configured_max_batch_size":
                    int(
                        self.max_batch_size
                    ),

                "batch_fill_fraction":
                    float(
                        batch_fill_fraction
                    ),

                "configured_batch_wait_ms":
                    float(
                        self.batch_wait_ms
                    ),

                "average_collection_wait_ms":
                    float(
                        average_collection_wait_ms
                    ),

                "average_batch_build_ms":
                    float(
                        average_batch_build_ms
                    ),

                "average_batch_inference_ms":
                    float(
                        average_batch_inference_ms
                    ),

                "total_batch_inference_seconds":
                    float(
                        self._batch_inference_seconds_total
                    ),

                "inference_positions_per_second":
                    float(
                        positions_per_second
                    ),
            }

    def reset_stats(self):
        """
        Reset batching telemetry without restarting the worker.
        """

        with self._stats_lock:

            self._requests_submitted = 0
            self._requests_completed = 0
            self._batches_completed = 0
            self._batch_items_total = 0
            self._max_observed_batch_size = 0

            self._queue_wait_seconds_total = 0.0
            self._batch_inference_seconds_total = 0.0
            self._batch_build_seconds_total = 0.0
