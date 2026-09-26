"""
CPU-side evaluator client for the centralized GPU inference server.

Public contract
---------------
This preserves the evaluator interface already used by MCTS:

    legal_probs, value = evaluator.evaluate(
        env=env,
        state=state,
        legal_actions=legal_actions,
    )

The client does NOT own Model 4 and does NOT use CUDA.

Instead it:

    1. encodes the Splendor state locally
    2. converts legal actions to canonical action IDs
    3. sends an InferenceRequest to the centralized GPU process
    4. waits on this worker's dedicated response queue
    5. verifies worker_id/request_id
    6. returns a CPU torch.float32 policy tensor + Python float value

Intended usage
--------------
Create one MultiprocessNeuralEvaluator per game/MCTS worker process.

Each game worker should have a unique worker_id and its corresponding
response_queue from InferenceIPC.response_queues[worker_id].

This class is designed for synchronous MCTS: one outstanding neural
request per worker at a time.
"""

from __future__ import annotations

from dataclasses import dataclass
from queue import Empty, Full
import time
from typing import Any, Optional

import numpy as np
import torch

from splendor_v1.mcts_batched.inference_protocol import (
    InferenceRequest,
    InferenceResponse,
)


# ============================================================
# ERRORS
# ============================================================


class MultiprocessNeuralEvaluatorError(
    RuntimeError
):
    """Base error for evaluator IPC failures."""


class InferenceRequestTimeout(
    MultiprocessNeuralEvaluatorError
):
    """Timed out waiting for the GPU server response."""


class InferenceRequestQueueFull(
    MultiprocessNeuralEvaluatorError
):
    """Timed out trying to submit a request to the server."""


class InferenceProtocolError(
    MultiprocessNeuralEvaluatorError
):
    """Received an invalid or mismatched inference response."""


class RemoteInferenceError(
    MultiprocessNeuralEvaluatorError
):
    """GPU server returned an error for this request."""


# ============================================================
# CLIENT TELEMETRY
# ============================================================


@dataclass(slots=True)
class _ClientStats:
    requests_submitted: int = 0
    responses_completed: int = 0
    responses_failed: int = 0

    request_put_seconds_total: float = 0.0
    response_wait_seconds_total: float = 0.0
    total_round_trip_seconds: float = 0.0

    max_round_trip_seconds: float = 0.0

    unexpected_responses: int = 0


# ============================================================
# MULTIPROCESS EVALUATOR
# ============================================================


class MultiprocessNeuralEvaluator:
    """
    Synchronous MCTS evaluator backed by a centralized GPU process.

    Parameters
    ----------
    worker_id:
        Unique zero-based worker index.

    request_queue:
        Shared multiprocessing request queue consumed by the GPU
        inference server.

    response_queue:
        Dedicated multiprocessing response queue for this worker.

    request_timeout_s:
        Maximum time to wait for a matching GPU response.
        None waits indefinitely, though a finite timeout is strongly
        recommended for self-play robustness.

    request_put_timeout_s:
        Maximum time to block while submitting to a bounded request
        queue. None blocks indefinitely.

    observation_dtype:
        NumPy dtype used before IPC. float32 is recommended because
        it cuts payload size and matches normal Model 4 inference.

    return_policy_device:
        Device for the returned policy tensor.

        Default "cpu" is intentional: game/MCTS workers should not
        initialize CUDA. Do not set this to "cuda" in multiprocess
        self-play workers.

    strict_response_order:
        With synchronous MCTS there should be one outstanding request
        per worker, so any nonmatching response is a protocol error.

        Set False only if a future client allows multiple outstanding
        requests per worker; unexpected responses will then be stashed
        until their request_id is requested.
    """

    def __init__(
        self,
        *,
        worker_id: int,
        request_queue: Any,
        response_queue: Any,
        request_timeout_s: Optional[float] = 120.0,
        request_put_timeout_s: Optional[float] = 30.0,
        observation_dtype=np.float32,
        return_policy_device: str = "cpu",
        strict_response_order: bool = True,
    ):
        worker_id = int(
            worker_id
        )

        if worker_id < 0:
            raise ValueError(
                "worker_id must be >= 0."
            )

        if (
            request_timeout_s is not None
            and request_timeout_s <= 0
        ):
            raise ValueError(
                "request_timeout_s must be > 0 "
                "or None."
            )

        if (
            request_put_timeout_s is not None
            and request_put_timeout_s <= 0
        ):
            raise ValueError(
                "request_put_timeout_s must be > 0 "
                "or None."
            )

        self.worker_id = (
            worker_id
        )

        self.request_queue = (
            request_queue
        )

        self.response_queue = (
            response_queue
        )

        self.request_timeout_s = (
            request_timeout_s
        )

        self.request_put_timeout_s = (
            request_put_timeout_s
        )

        self.observation_dtype = (
            np.dtype(
                observation_dtype
            )
        )

        self.return_policy_device = (
            torch.device(
                return_policy_device
            )
        )

        if (
            self.return_policy_device.type
            != "cpu"
        ):
            raise ValueError(
                "MultiprocessNeuralEvaluator "
                "must return policy tensors on CPU "
                "for centralized-GPU self-play. "
                "Use return_policy_device='cpu'."
            )

        self.strict_response_order = bool(
            strict_response_order
        )

        # Worker-local monotonically increasing sequence.
        # request_id only needs to be unique within this worker
        # because responses are routed through a dedicated queue.
        self._next_request_id = 0

        # Only used when strict_response_order=False.
        self._pending_responses = {}

        self._stats = (
            _ClientStats()
        )

    # --------------------------------------------------------
    # Public evaluator API
    # --------------------------------------------------------

    def evaluate(
        self,
        env,
        state,
        legal_actions=None,
    ):
        """
        Evaluate one state through the centralized GPU server.

        Returns
        -------
        legal_probs:
            CPU torch.float32 tensor with shape
            [num_legal_actions].

            Its ordering exactly matches legal_actions.

        value:
            Python float in [-1, +1],
            P(WIN) - P(LOSS).
        """

        if legal_actions is None:
            legal_actions = (
                env._legal_actions(
                    state
                )
            )

        if not legal_actions:
            raise ValueError(
                "MultiprocessNeuralEvaluator.evaluate "
                "received no legal actions."
            )

        observation = (
            env.observation_encoder.encoder(
                state
            )
        )

        observation = np.asarray(
            observation,
            dtype=self.observation_dtype,
        )

        if observation.ndim != 1:
            raise ValueError(
                "Encoded observation must be 1D. "
                f"Got shape {observation.shape}."
            )

        # Make the IPC payload independent of a view into any
        # environment-owned array and ensure efficient pickling.
        observation = np.ascontiguousarray(
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

        return self.evaluate_encoded(
            observation=observation,
            legal_action_ids=(
                legal_action_ids
            ),
        )

    def __call__(
        self,
        env,
        state,
        legal_actions=None,
    ):
        return self.evaluate(
            env=env,
            state=state,
            legal_actions=legal_actions,
        )

    # --------------------------------------------------------
    # Encoded API
    # --------------------------------------------------------

    def evaluate_encoded(
        self,
        *,
        observation,
        legal_action_ids,
    ):
        """
        Lower-level IPC method.

        This is useful for smoke tests and future workers that already
        have an encoded observation.
        """

        observation = np.asarray(
            observation,
            dtype=self.observation_dtype,
        )

        if observation.ndim != 1:
            raise ValueError(
                "observation must be 1D. "
                f"Got shape {observation.shape}."
            )

        observation = np.ascontiguousarray(
            observation
        )

        legal_action_ids = [
            int(action_id)
            for action_id
            in legal_action_ids
        ]

        if not legal_action_ids:
            raise ValueError(
                "legal_action_ids must not be empty."
            )

        request_id = (
            self._allocate_request_id()
        )

        request = InferenceRequest(
            worker_id=(
                self.worker_id
            ),
            request_id=request_id,
            observation=observation,
            legal_action_ids=(
                legal_action_ids
            ),
            submitted_perf_counter_ns=(
                time.perf_counter_ns()
            ),
        )

        round_trip_start = (
            time.perf_counter()
        )

        self._submit_request(
            request
        )

        response = (
            self._wait_for_response(
                request_id=request_id
            )
        )

        round_trip_seconds = (
            time.perf_counter()
            - round_trip_start
        )

        self._record_round_trip(
            round_trip_seconds
        )

        legal_probs = (
            self._validate_and_convert_response(
                response=response,
                request_id=request_id,
                expected_legal_count=(
                    len(
                        legal_action_ids
                    )
                ),
            )
        )

        return (
            legal_probs,
            float(
                response.value
            ),
        )

    # --------------------------------------------------------
    # Request ID
    # --------------------------------------------------------

    def _allocate_request_id(
        self,
    ):
        request_id = int(
            self._next_request_id
        )

        self._next_request_id += 1

        return request_id

    # --------------------------------------------------------
    # Queue submit
    # --------------------------------------------------------

    def _submit_request(
        self,
        request,
    ):
        put_start = (
            time.perf_counter()
        )

        try:
            if (
                self.request_put_timeout_s
                is None
            ):
                self.request_queue.put(
                    request
                )

            else:
                self.request_queue.put(
                    request,
                    timeout=(
                        self.request_put_timeout_s
                    ),
                )

        except Full as exc:
            self._stats.responses_failed += 1

            raise InferenceRequestQueueFull(
                "Timed out submitting inference "
                f"request worker={self.worker_id} "
                f"request={request.request_id}."
            ) from exc

        put_seconds = (
            time.perf_counter()
            - put_start
        )

        self._stats.requests_submitted += 1

        self._stats.request_put_seconds_total += (
            put_seconds
        )

    # --------------------------------------------------------
    # Response wait / matching
    # --------------------------------------------------------

    def _wait_for_response(
        self,
        *,
        request_id,
    ):
        # Future asynchronous clients can leave responses in this
        # stash. Synchronous MCTS normally never uses it.
        if request_id in self._pending_responses:
            return self._pending_responses.pop(
                request_id
            )

        wait_start = (
            time.perf_counter()
        )

        deadline = None

        if self.request_timeout_s is not None:
            deadline = (
                wait_start
                + float(
                    self.request_timeout_s
                )
            )

        while True:
            try:
                if deadline is None:
                    response = (
                        self.response_queue.get()
                    )

                else:
                    remaining = (
                        deadline
                        - time.perf_counter()
                    )

                    if remaining <= 0:
                        raise Empty

                    response = (
                        self.response_queue.get(
                            timeout=remaining
                        )
                    )

            except Empty as exc:
                self._stats.responses_failed += 1

                raise InferenceRequestTimeout(
                    "Timed out waiting for GPU "
                    "inference response "
                    f"worker={self.worker_id} "
                    f"request={request_id} "
                    f"timeout="
                    f"{self.request_timeout_s}s."
                ) from exc

            if not isinstance(
                response,
                InferenceResponse,
            ):
                self._stats.unexpected_responses += 1

                if self.strict_response_order:
                    self._stats.responses_failed += 1

                    raise InferenceProtocolError(
                        "Worker received a non-"
                        "InferenceResponse object: "
                        f"{type(response).__name__}."
                    )

                continue

            if (
                int(response.worker_id)
                != self.worker_id
            ):
                self._stats.unexpected_responses += 1

                self._stats.responses_failed += 1

                raise InferenceProtocolError(
                    "Response was routed to the wrong "
                    "worker queue: "
                    f"expected worker={self.worker_id}, "
                    f"got worker={response.worker_id}."
                )

            response_request_id = int(
                response.request_id
            )

            if (
                response_request_id
                == request_id
            ):
                wait_seconds = (
                    time.perf_counter()
                    - wait_start
                )

                self._stats.response_wait_seconds_total += (
                    wait_seconds
                )

                return response

            self._stats.unexpected_responses += 1

            if self.strict_response_order:
                self._stats.responses_failed += 1

                raise InferenceProtocolError(
                    "Unexpected request_id on dedicated "
                    "worker response queue: "
                    f"expected {request_id}, "
                    f"got {response_request_id}."
                )

            # Future multi-outstanding-request mode.
            self._pending_responses[
                response_request_id
            ] = response

    # --------------------------------------------------------
    # Response validation
    # --------------------------------------------------------

    def _validate_and_convert_response(
        self,
        *,
        response,
        request_id,
        expected_legal_count,
    ):
        if response.error is not None:
            self._stats.responses_failed += 1

            raise RemoteInferenceError(
                "GPU inference server returned an "
                f"error for worker={self.worker_id} "
                f"request={request_id}: "
                f"{response.error}"
            )

        if response.legal_probs is None:
            self._stats.responses_failed += 1

            raise InferenceProtocolError(
                "Successful inference response "
                "contained no legal_probs."
            )

        if response.value is None:
            self._stats.responses_failed += 1

            raise InferenceProtocolError(
                "Successful inference response "
                "contained no value."
            )

        probs_np = np.asarray(
            response.legal_probs,
            dtype=np.float32,
        )

        if probs_np.ndim != 1:
            self._stats.responses_failed += 1

            raise InferenceProtocolError(
                "legal_probs must be 1D. "
                f"Got shape {probs_np.shape}."
            )

        if (
            len(probs_np)
            != expected_legal_count
        ):
            self._stats.responses_failed += 1

            raise InferenceProtocolError(
                "Policy length does not match "
                "submitted legal actions: "
                f"expected {expected_legal_count}, "
                f"got {len(probs_np)}."
            )

        if not np.all(
            np.isfinite(
                probs_np
            )
        ):
            self._stats.responses_failed += 1

            raise InferenceProtocolError(
                "legal_probs contains NaN or Inf."
            )

        value = float(
            response.value
        )

        if not np.isfinite(
            value
        ):
            self._stats.responses_failed += 1

            raise InferenceProtocolError(
                "value contains NaN or Inf."
            )

        # Server uses softmax. Keep a loose protocol guard rather
        # than renormalizing and silently hiding a server bug.
        probability_sum = float(
            probs_np.sum()
        )

        if not np.isclose(
            probability_sum,
            1.0,
            atol=1e-4,
            rtol=1e-4,
        ):
            self._stats.responses_failed += 1

            raise InferenceProtocolError(
                "legal_probs does not sum to 1 "
                "within tolerance: "
                f"sum={probability_sum}."
            )

        if np.any(
            probs_np < -1e-7
        ):
            self._stats.responses_failed += 1

            raise InferenceProtocolError(
                "legal_probs contains negative "
                "probabilities."
            )

        if not (
            -1.0001
            <= value
            <= 1.0001
        ):
            self._stats.responses_failed += 1

            raise InferenceProtocolError(
                "WDL scalar value is outside "
                "expected [-1, +1] range: "
                f"{value}."
            )

        probs_np = np.ascontiguousarray(
            probs_np
        )

        legal_probs = torch.from_numpy(
            probs_np
        ).to(
            device=(
                self.return_policy_device
            ),
            dtype=torch.float32,
        )

        self._stats.responses_completed += 1

        return legal_probs

    # --------------------------------------------------------
    # Telemetry
    # --------------------------------------------------------

    def _record_round_trip(
        self,
        seconds,
    ):
        seconds = float(
            seconds
        )

        self._stats.total_round_trip_seconds += (
            seconds
        )

        self._stats.max_round_trip_seconds = max(
            self._stats.max_round_trip_seconds,
            seconds,
        )

    def stats_snapshot(
        self,
    ):
        submitted = int(
            self._stats.requests_submitted
        )

        completed = int(
            self._stats.responses_completed
        )

        return {
            "worker_id":
                int(
                    self.worker_id
                ),

            "requests_submitted":
                submitted,

            "responses_completed":
                completed,

            "responses_failed":
                int(
                    self._stats.responses_failed
                ),

            "unexpected_responses":
                int(
                    self._stats.unexpected_responses
                ),

            "average_request_put_ms":
                float(
                    self._stats.request_put_seconds_total
                    / submitted
                    * 1000.0
                    if submitted > 0
                    else 0.0
                ),

            "average_response_wait_ms":
                float(
                    self._stats.response_wait_seconds_total
                    / completed
                    * 1000.0
                    if completed > 0
                    else 0.0
                ),

            "average_round_trip_ms":
                float(
                    self._stats.total_round_trip_seconds
                    / completed
                    * 1000.0
                    if completed > 0
                    else 0.0
                ),

            "max_round_trip_ms":
                float(
                    self._stats.max_round_trip_seconds
                    * 1000.0
                ),

            "next_request_id":
                int(
                    self._next_request_id
                ),
        }

    def reset_stats(
        self,
    ):
        self._stats = (
            _ClientStats()
        )

    # --------------------------------------------------------
    # Lifecycle
    # --------------------------------------------------------

    def close(
        self,
    ):
        """
        No-op by design.

        The parent coordinator owns multiprocessing queues and the
        GPU server lifecycle. Individual clients must not close or
        cancel shared queue feeder threads.
        """

        return None

    def __enter__(
        self,
    ):
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback_object,
    ):
        self.close()
