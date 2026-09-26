"""
Multiprocessing IPC messages for centralized neural inference.

Only simple CPU-side data crosses the process boundary:

    worker -> GPU server:
        observation
        legal_action_ids

    GPU server -> worker:
        legal_probs
        scalar value

No Splendor State, Action, Node, MCTS tree, PyTorch model, or CUDA
tensor is sent through multiprocessing queues.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


@dataclass(slots=True)
class InferenceRequest:
    """
    One neural evaluation request from a game/MCTS worker.

    worker_id:
        Integer index used to select that worker's response queue.

    request_id:
        Worker-owned request ID copied into the response.

    observation:
        Encoded Splendor observation on CPU.

    legal_action_ids:
        Canonical action IDs in exactly the same order as the
        worker's legal_actions list.

    submitted_perf_counter_ns:
        Optional perf_counter_ns timestamp used only for telemetry.
    """

    worker_id: int
    request_id: int
    observation: np.ndarray
    legal_action_ids: list[int]
    submitted_perf_counter_ns: int = 0


@dataclass(slots=True)
class InferenceResponse:
    """
    Response routed back to the originating worker.

    On success:
        error is None
        legal_probs has shape [num_legal_actions]
        value is a Python float in [-1, +1]

    On failure:
        error contains a readable message.
    """

    worker_id: int
    request_id: int
    legal_probs: Optional[np.ndarray] = None
    value: Optional[float] = None
    error: Optional[str] = None


@dataclass(slots=True)
class InferenceShutdown:
    """Graceful shutdown message."""

    reason: str = "shutdown requested"


@dataclass(slots=True)
class InferenceServerStatus:
    """
    Status/telemetry event emitted by the server.

    kind:
        starting
        ready
        stats
        stopped
        fatal_error
    """

    kind: str
    pid: int
    message: str = ""
    stats: Optional[dict[str, Any]] = None
