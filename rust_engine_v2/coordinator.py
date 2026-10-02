"""V6-compatible coordinator: Rust owns all trees and gathers neural batches.

num_workers now means concurrent game slots, not threads. The owner loop only
handles real-decision replay boundaries and one PyTorch evaluation per batch.
"""
import copy
from dataclasses import asdict
import os
import time

from splendor_v1.mcts_batched.multiprocess_self_play_coordinator import (
    MultiprocessSelfPlayCoordinator, MultiprocessSelfPlayCoordinatorConfig,
)
from splendor_v1.mcts_batched.multiprocess_self_play_worker import SelfPlayGameJob
from splendor_v1.rust_engine_v2.arena import decode_batch, response_bytes
from splendor_v1.rust_engine_v2.inference import (
    InferenceOptions, PackedModel4Evaluator, load_model,
)
from splendor_v1.rust_engine_v2.self_play import GameSession
from splendor_v1.rust_engine_v2.self_play_settings import EXECUTION
from splendor_v1.rust_engine.failures import FailurePolicy


class NativeSelfPlayError(RuntimeError):
    pass


class NativeSelfPlayCoordinator:
    def __init__(self, *, replay_buffer, config, mp_context=None, evaluator=None,
                 options=None, verbose=True, heartbeat_s=10.0,
                 max_rejected_games=0, failure_dir="splendor_v1/training_v6/data/native_failures"):
        del mp_context
        config.validate()
        if not config.worker_config.return_samples:
            raise ValueError("V2 requires completed-game replay samples")
        if heartbeat_s <= 0:
            raise ValueError("heartbeat_s must be positive")
        self.config, self.replay_buffer = config, replay_buffer
        self.options = options or InferenceOptions()
        self.options.validate()
        self.evaluator, self._owns_evaluator = evaluator, evaluator is None
        self.verbose, self.heartbeat_s = verbose, heartbeat_s
        self.last_run_summary = None
        self._closed = False
        self.arena = None
        self.failures = FailurePolicy(max_rejected_games, failure_dir)

    @property
    def worker_pids(self):
        return [os.getpid()]

    def _log(self, message):
        if self.verbose: print(message, flush=True)

    def _load_evaluator(self):
        if self.evaluator is not None: return
        self._log(f"V2: loading checkpoint on {self.config.device}; precision={self.options.precision}, "
                  f"compile={self.options.compile_mode}")
        self.evaluator = PackedModel4Evaluator(
            load_model(self.config.checkpoint_path, self.config.device), self.options)

    def close(self):
        self.arena = None
        if self._owns_evaluator: self.evaluator = None
        self._closed = True

    def run_block(self, *, num_games, seed_start, game_id_start=0,
                  extra_game_metadata=None, checkpoint_every_games=None,
                  checkpoint_callback=None, progress_callback=None, job_factory=None):
        from splendor_rust_v2 import RustArena
        if self._closed or self.arena is not None:
            raise NativeSelfPlayError("A coordinator can run only one block")
        if num_games < 1 or seed_start < 0 or game_id_start < 0:
            raise ValueError("Invalid game count, seed, or game ID")
        if checkpoint_every_games is not None and checkpoint_every_games < 1:
            raise ValueError("checkpoint_every_games must be positive")
        block_start = time.perf_counter()
        active, results, persistent_ids, job_ids = {}, [], [], set()
        submitted = batches = neural_rows = max_batch = decisions = 0
        scheduler_s = boundary_s = evaluation_s = load_s = 0.0
        last_heartbeat = block_start
        metadata = asdict(self.options)

        def submit(slot):
            nonlocal submitted
            offset = submitted
            gid, seed = game_id_start + offset, seed_start + offset
            job = job_factory(offset, gid, seed) if job_factory else SelfPlayGameJob(
                gid, seed, extra_game_metadata=copy.deepcopy(extra_game_metadata))
            if not isinstance(job, SelfPlayGameJob) or job.game_id < 0 or job.seed < 0:
                raise ValueError("job_factory must return a valid SelfPlayGameJob")
            if job.game_id in job_ids: raise ValueError("job_factory returned duplicate game IDs")
            job_ids.add(job.game_id)
            try:
                active[slot] = GameSession(self.arena, slot, job, self.config.worker_config, metadata)
            except Exception as exc:
                _, diagnostic = self.failures.handle(exc)
                raise NativeSelfPlayError(f"V2 game failed at initialization: job={job.game_id}, "
                                          f"seed={job.seed}: {exc} {diagnostic}") from exc
            submitted += 1

        def telemetry():
            info = self.evaluator.profile() if hasattr(self.evaluator, "profile") else {}
            info.update(backend="rust_arena_v2", total_batches=batches,
                total_inference_positions=neural_rows, max_observed_batch_size=max_batch,
                configured_max_batch_size=self.config.resolved_max_batch_size,
                average_batch_size=neural_rows / batches if batches else 0.0,
                total_inference_seconds=evaluation_s,
                inference_positions_per_second=neural_rows / evaluation_s if evaluation_s else 0.0)
            return info

        try:
            self._load_evaluator()
            checkpoint_sha256 = getattr(getattr(self.evaluator, "model", None), "checkpoint_sha256", None)
            if checkpoint_sha256 is not None: metadata["checkpoint_sha256"] = checkpoint_sha256
            load_s = time.perf_counter() - block_start
            if hasattr(self.evaluator, "reset_profile"): self.evaluator.reset_profile()
            self.arena = RustArena(min(num_games, self.config.num_workers))
            self.replay_buffer.metadata["self_play_execution"] = EXECUTION
            self._log(f"V2: starting {min(num_games, self.config.num_workers)} native game slots; "
                      f"maximum inference batch={self.config.resolved_max_batch_size}")
            boundary_start = time.perf_counter()
            for slot in range(min(num_games, self.config.num_workers)): submit(slot)
            boundary_s += time.perf_counter() - boundary_start
            while active:
                iteration_start = time.perf_counter()
                started = time.perf_counter()
                token, slots, ready, observations, actions, mask = decode_batch(
                    self.arena.gather(self.config.resolved_max_batch_size))
                scheduler_s += time.perf_counter() - started
                boundary_start = time.perf_counter()
                for slot in ready:
                    session = active[slot]
                    try:
                        result = session.finish_decision()
                    except Exception as exc:
                        recover, diagnostic = self.failures.handle(exc)
                        if not recover:
                            raise NativeSelfPlayError(f"V2 game failed: job={session.job.game_id}, "
                                                      f"seed={session.job.seed}: {exc} {diagnostic}") from exc
                        del active[slot]
                        self.arena.remove(slot)
                        if len(results) + len(active) < num_games: submit(slot)
                        continue
                    decisions += 1
                    if result is None: continue
                    persistent_id = MultiprocessSelfPlayCoordinator._commit_completed_game(self, result)
                    result.samples = None
                    results.append(result); persistent_ids.append(persistent_id)
                    del active[slot]
                    self.arena.remove(slot)
                    last_completion = time.perf_counter()
                    committed, wall = len(results), last_completion - block_start
                    if progress_callback:
                        progress_callback(dict(requested_games=num_games, submitted_games=submitted,
                            committed_games=committed, in_flight_games=len(active),
                            latest_positions=result.num_positions, games_per_hour=committed / wall * 3600,
                            wall_seconds=wall, gpu_server=telemetry()))
                    if checkpoint_callback and checkpoint_every_games and committed % checkpoint_every_games == 0:
                        checkpoint_callback(dict(committed_games=committed, requested_games=num_games,
                            persistent_game_ids=list(persistent_ids), wall_seconds=wall))
                    if len(results) + len(active) < num_games: submit(slot)
                boundary_s += time.perf_counter() - boundary_start
                if slots:
                    started = time.perf_counter()
                    output = self.evaluator.evaluate_packed(observations, actions, mask)
                    payload, double = response_bytes(output, len(slots), actions.shape[1])
                    evaluation_s += time.perf_counter() - started
                    started = time.perf_counter()
                    self.arena.respond(token, payload, double)
                    scheduler_s += time.perf_counter() - started
                    batches += 1; neural_rows += len(slots); max_batch = max(max_batch, len(slots))
                now = time.perf_counter()
                if active and now - iteration_start > self.config.game_result_timeout_s:
                    raise NativeSelfPlayError("A V2 scheduler/evaluation iteration exceeded the progress timeout")
                if active and not slots and not ready:
                    raise NativeSelfPlayError("V2 arena made no progress with active games")
                if now - last_heartbeat >= self.heartbeat_s:
                    self._log(f"V2: {len(results)}/{num_games} completed, {len(active)} active games, "
                              f"{decisions} decisions, {neural_rows:,} evaluations, "
                              f"average batch={neural_rows / batches if batches else 0:.1f}, "
                              f"elapsed={now - block_start:.1f}s")
                    last_heartbeat = now
            summary = MultiprocessSelfPlayCoordinator._build_final_summary(self,
                requested=num_games, submitted=submitted, completed=len(results), committed=len(results),
                block_start=block_start, results=results, persistent_game_ids=persistent_ids)
            summary.update(execution_backend="rust_v2", self_play_execution=EXECUTION,
                failed_games=len(self.failures.records), rejected_games=list(self.failures.records),
                native_game_slots=self.config.num_workers, native_worker_threads=0,
                inference_options=metadata, gpu_server=telemetry(),
                checkpoint_sha256=checkpoint_sha256,
                worker_evaluator={"neural_requests": neural_rows},
                pipeline_profile=dict(checkpoint_load_seconds=load_s, native_scheduler_seconds=scheduler_s,
                    replay_boundary_seconds=boundary_s, evaluator_seconds=evaluation_s,
                    completed_decisions=decisions))
            self.last_run_summary = summary
            return summary
        finally:
            self.close()


NativeSelfPlayCoordinatorConfig = MultiprocessSelfPlayCoordinatorConfig
