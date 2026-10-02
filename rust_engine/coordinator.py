"""Concurrent native searches with one shared in-process Model 4 inference owner.

Workers are threads, not spawned Python processes. Native selection/backup/move
execution releases the GIL. Requests use an in-process queue; the main thread
owns batched PyTorch evaluation and whole-game replay commits.
"""
import copy
import os
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass
from queue import Empty, Queue

from splendor_v1.mcts_batched.multiprocess_self_play_coordinator import (
    MultiprocessSelfPlayCoordinator, MultiprocessSelfPlayCoordinatorConfig,
)
from splendor_v1.mcts_batched.multiprocess_self_play_worker import SelfPlayGameJob
from splendor_v1.rust_engine.mcts import Model4Evaluator
from splendor_v1.rust_engine.self_play import EXECUTION, run_native_game
from splendor_v1.rust_engine.failures import FailurePolicy


class NativeSelfPlayError(RuntimeError):
    pass


@dataclass
class _Request:
    observation: object
    legal_ids: object
    response: Future


class _ThreadEvaluator:
    def __init__(self, requests, stop, timeout):
        self.requests = requests
        self.stop = stop
        self.timeout = timeout

    def evaluate(self, observation, legal_ids):
        if self.stop.is_set():
            raise NativeSelfPlayError("Native self-play stopped")
        response = Future()
        self.requests.put(_Request(observation, legal_ids, response), timeout=self.timeout)
        deadline = time.perf_counter() + self.timeout
        while not self.stop.is_set():
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                raise NativeSelfPlayError("Timed out waiting for a native neural evaluation")
            try:
                return response.result(timeout=min(0.05, remaining))
            except TimeoutError:
                if response.done():
                    raise
                continue
        raise NativeSelfPlayError("Native self-play stopped")


class NativeSelfPlayCoordinator:
    """The V6 coordinator's run_block interface with native threaded workers.

    Accepts the existing coordinator/worker configuration to retain current
    settings. Spawn delays/IPC startup/shutdown options have no effect here.
    A supplied evaluator is useful for deterministic tests and must implement
    evaluate_batch(requests). Otherwise the inference snapshot is loaded once.
    """

    def __init__(self, *, replay_buffer, config, mp_context=None, evaluator=None,
                 max_rejected_games=0, failure_dir="splendor_v1/training_v6/data/native_failures"):
        del mp_context
        config.validate()
        if not config.worker_config.return_samples:
            raise ValueError("Native coordinator requires completed-game samples")
        self.replay_buffer = replay_buffer
        self.config = config
        self.evaluator = evaluator
        self._owns_evaluator = evaluator is None
        self._pool = None
        self._closed = False
        self._stop = threading.Event()
        self._requests = Queue(maxsize=max(2, config.num_workers * 2))
        self.last_run_summary = None
        self.failures = FailurePolicy(max_rejected_games, failure_dir)

    @property
    def worker_pids(self):
        return [os.getpid()]  # One process; native searches use multiple threads.

    def _load_evaluator(self):
        if self.evaluator is not None:
            return
        import torch
        from splendor_v1.network.model_4_legal_scorer import SplendorNetwork
        checkpoint = torch.load(self.config.checkpoint_path, map_location="cpu", weights_only=True)
        weights = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
        model = SplendorNetwork()
        model.load_state_dict(weights)
        model.to(self.config.device).eval()
        self.evaluator = Model4Evaluator(model)

    def close(self):
        self._stop.set()
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=True)
            self._pool = None
        while True:
            try:
                request = self._requests.get_nowait()
            except Empty:
                break
            if not request.response.done():
                request.response.set_exception(NativeSelfPlayError("Native coordinator closed"))
        if self._owns_evaluator:
            self.evaluator = None
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        self._closed = True

    def run_block(self, *, num_games, seed_start, game_id_start=0,
                  extra_game_metadata=None, checkpoint_every_games=None,
                  checkpoint_callback=None, progress_callback=None, job_factory=None):
        if self._closed or self._pool is not None:
            raise NativeSelfPlayError("A coordinator can run only one block")
        if num_games < 1 or seed_start < 0 or game_id_start < 0:
            raise ValueError("num_games must be positive; seed and game IDs nonnegative")
        if checkpoint_every_games is not None and checkpoint_every_games < 1:
            raise ValueError("checkpoint_every_games must be positive")
        block_start = time.perf_counter()
        self._load_evaluator()
        self._pool = ThreadPoolExecutor(max_workers=self.config.num_workers,
                                        thread_name_prefix="SplendorRust")
        self.replay_buffer.metadata["self_play_execution"] = EXECUTION
        clients = [_ThreadEvaluator(self._requests, self._stop,
                                    self.config.worker_config.request_timeout_s)
                   for _ in range(self.config.num_workers)]
        active = {}
        results, persistent_ids = [], []
        submitted = batches = inference_positions = max_observed = 0
        inference_seconds = 0.0
        job_ids = set()
        last_completion = time.perf_counter()

        def submit(slot):
            nonlocal submitted
            offset = submitted
            game_id, seed = game_id_start + offset, seed_start + offset
            job = job_factory(offset, game_id, seed) if job_factory else SelfPlayGameJob(
                game_id=game_id, seed=seed, extra_game_metadata=copy.deepcopy(extra_game_metadata))
            if not isinstance(job, SelfPlayGameJob) or job.game_id < 0 or job.seed < 0:
                raise ValueError("job_factory must return a valid SelfPlayGameJob")
            if job.game_id in job_ids:
                raise ValueError("job_factory returned a duplicate game ID")
            job_ids.add(job.game_id)
            future = self._pool.submit(run_native_game, worker_id=slot,
                evaluator=clients[slot], job=job, config=self.config.worker_config)
            active[future] = (slot, job)
            submitted += 1

        def telemetry():
            return dict(backend="rust_threads", average_batch_size=inference_positions / batches if batches else 0.0,
                max_observed_batch_size=max_observed, configured_max_batch_size=self.config.resolved_max_batch_size,
                inference_positions_per_second=inference_positions / inference_seconds if inference_seconds else 0.0,
                total_batches=batches, total_inference_positions=inference_positions,
                total_inference_seconds=inference_seconds)

        try:
            for slot in range(min(self.config.num_workers, num_games)):
                submit(slot)
            while active:
                # Only this owner thread touches the shared replay buffer.
                completed = [future for future in active if future.done()]
                for future in completed:
                    slot, job = active.pop(future)
                    try:
                        result = future.result()
                    except Exception as exc:
                        recover, diagnostic = self.failures.handle(exc)
                        if not recover:
                            raise NativeSelfPlayError(f"Native game failed: job={job.game_id}, seed={job.seed}: "
                                                      f"{exc} {diagnostic}") from exc
                        last_completion = time.perf_counter()
                        if len(results) + len(active) < num_games:
                            submit(slot)
                        continue
                    result.game_metadata["multiprocess_coordinator"] = False
                    persistent_id = MultiprocessSelfPlayCoordinator._commit_completed_game(self, result)
                    # Replay owns the arrays now; don't retain evicted trajectories in block statistics.
                    result.samples = None
                    persistent_ids.append(persistent_id)
                    results.append(result)
                    last_completion = time.perf_counter()
                    committed = len(results)
                    wall = last_completion - block_start
                    if progress_callback is not None:
                        progress_callback(dict(requested_games=num_games, submitted_games=submitted,
                            committed_games=committed, in_flight_games=len(active),
                            latest_positions=result.num_positions, games_per_hour=committed / wall * 3600,
                            wall_seconds=wall, gpu_server=telemetry()))
                    if checkpoint_callback is not None and checkpoint_every_games is not None and committed % checkpoint_every_games == 0:
                        checkpoint_callback(dict(committed_games=committed, requested_games=num_games,
                            persistent_game_ids=list(persistent_ids), wall_seconds=wall))
                    if len(results) + len(active) < num_games:
                        submit(slot)
                if not active:
                    break
                if time.perf_counter() - last_completion > self.config.game_result_timeout_s:
                    raise NativeSelfPlayError("Native self-play stalled without a completed game")
                try:
                    first = self._requests.get(timeout=0.005)
                except Empty:
                    continue
                batch = [first]
                deadline = time.perf_counter() + self.config.batch_wait_ms / 1000.0
                while len(batch) < self.config.resolved_max_batch_size:
                    try:
                        batch.append(self._requests.get_nowait())
                        continue
                    except Empty:
                        remaining = deadline - time.perf_counter()
                        if remaining <= 0:
                            break
                        try:
                            batch.append(self._requests.get(timeout=remaining))
                        except Empty:
                            break
                evaluation_start = time.perf_counter()
                try:
                    responses = self.evaluator.evaluate_batch([(r.observation, r.legal_ids) for r in batch])
                    if len(responses) != len(batch):
                        raise NativeSelfPlayError("Inference returned the wrong batch size")
                except Exception as exc:
                    for request in batch:
                        request.response.set_exception(exc)
                    raise
                inference_seconds += time.perf_counter() - evaluation_start
                batches += 1
                inference_positions += len(batch)
                max_observed = max(max_observed, len(batch))
                for request, response in zip(batch, responses):
                    request.response.set_result(response)
            summary = MultiprocessSelfPlayCoordinator._build_final_summary(self,
                requested=num_games, submitted=submitted, completed=len(results), committed=len(results),
                block_start=block_start, results=results, persistent_game_ids=persistent_ids)
            summary.update(execution_backend="rust", self_play_execution=EXECUTION,
                failed_games=len(self.failures.records), rejected_games=list(self.failures.records),
                gpu_server=telemetry(), worker_evaluator={"neural_requests": inference_positions},
                native_worker_threads=self.config.num_workers)
            self.last_run_summary = summary
            return summary
        finally:
            self.close()


# Reuse V6's configuration type; keep model/search/training settings in one place.
NativeSelfPlayCoordinatorConfig = MultiprocessSelfPlayCoordinatorConfig
