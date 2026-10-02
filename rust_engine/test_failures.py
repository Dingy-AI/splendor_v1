"""Shared dead-end evidence, bounded replacement, and mismatch rejection."""
import copy
import json
import pytest

from splendor_v1.rust_engine.failures import EmptyActionError, FailurePolicy, empty_action_error
from splendor_v1.rust_engine.replay_failure import replay_failure
from splendor_v1.rust_engine.test_self_play import ArraysEvaluator, worker_config
from splendor_v1.rust_engine.coordinator import NativeSelfPlayCoordinator as V1
from splendor_v1.rust_engine.coordinator import NativeSelfPlayCoordinatorConfig
from splendor_v1.mcts_batched.multiprocess_self_play_worker import SelfPlayGameJob
from splendor_v1.training_v2.replay_buffer import ReplayBuffer
from splendor_v1.env.env import SplendorEnv
from splendor_v1.training_v2.state_serializer import serialize_state


def backend_classes(backend):
    if backend == "v1":
        return V1, ArraysEvaluator()
    pytest.importorskip("splendor_rust_v2")
    from splendor_v1.rust_engine_v2.coordinator import NativeSelfPlayCoordinator as V2
    from splendor_v1.rust_engine_v2.test_arena import PackedArraysEvaluator
    return V2, PackedArraysEvaluator()


@pytest.mark.parametrize("backend", ["v1", "v2"])
@pytest.mark.parametrize("workers", [1, 2])
def test_shared_dead_end_is_saved_reproduced_and_replaced(tmp_path, backend, workers):
    buffer = ReplayBuffer(5000)
    config = NativeSelfPlayCoordinatorConfig(num_workers=workers, checkpoint_path="unused",
        worker_config=worker_config(), max_batch_size=1, device="cpu")
    cls, evaluator = backend_classes(backend)
    owner = cls(replay_buffer=buffer, config=config, evaluator=evaluator,
                max_rejected_games=1, failure_dir=tmp_path)
    jobs = []
    def factory(offset, gid, seed):
        jobs.append((offset, gid, seed))
        # First job is the known deterministic shared dead-end; replacements finish.
        return SelfPlayGameJob(gid, 42 if offset == 0 else offset - 1,
            extra_game_metadata={"game_index": gid})
    progress = []
    summary = owner.run_block(num_games=2, seed_start=19417, game_id_start=9417,
                             job_factory=factory, progress_callback=progress.append)
    assert summary["requested_games"] == summary["committed_games"] == 2
    assert summary["submitted_games"] == 3 and summary["failed_games"] == 1
    assert jobs == [(0, 9417, 19417), (1, 9418, 19418), (2, 9419, 19419)]
    assert sorted(g["game_index"] for g in buffer.games.values()) == [9418, 9419]
    assert len(progress) == 2 and not any(g["seed"] == 42 for g in buffer.games.values())
    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1
    data = json.loads(files[0].read_text())
    assert data["shared_dead_end"] and data["job_id"] == 9417 and data["seed"] == 42
    assert data["python_legal_action_ids"] == data["native_legal_action_ids"] == []
    assert data["step_index"] == len(data["action_ids"]) > 0
    assert replay_failure(data)["shared_nonterminal_dead_end"]
    assert data["trajectory_verification"]["shared_nonterminal_dead_end"]
    modified = copy.deepcopy(data); modified["action_ids"][0] = 1138
    with pytest.raises(RuntimeError): replay_failure(modified)
    policy = FailurePolicy(100, tmp_path / "mismatch")
    allowed, message = policy.handle(EmptyActionError(modified))
    assert not allowed and "mismatch" in message
    assert "trajectory_verification_error" in modified


@pytest.mark.parametrize("backend", ["v1", "v2"])
@pytest.mark.parametrize("limit", [0, 1])
def test_repeated_dead_ends_stop_at_limit_without_partial_replay(tmp_path, backend, limit):
    buffer = ReplayBuffer(5000)
    config = NativeSelfPlayCoordinatorConfig(num_workers=1, checkpoint_path="unused",
        worker_config=worker_config(), device="cpu")
    cls, evaluator = backend_classes(backend)
    owner = cls(replay_buffer=buffer, config=config, evaluator=evaluator,
                max_rejected_games=limit, failure_dir=tmp_path)
    with pytest.raises(RuntimeError, match="Diagnostic:.*limit exceeded"):
        owner.run_block(num_games=1, seed_start=42,
                        job_factory=lambda offset, gid, seed: SelfPlayGameJob(gid, 42))
    assert len(list(tmp_path.glob("*.json"))) == limit + 1
    assert not buffer.buffer and not buffer.games and owner._closed


def test_python_native_empty_action_mismatch_is_never_recoverable(tmp_path):
    env = SplendorEnv(); env.reset(seed=0)
    error = empty_action_error(env, serialize_state(env.state), 0)
    error.add_context(SelfPlayGameJob(0, 0), worker_config(), [], "rust")
    assert error.diagnostic["python_legal_action_ids"] and not error.diagnostic["shared_dead_end"]
    policy = FailurePolicy(100, tmp_path)
    allowed, message = policy.handle(error)
    assert not allowed and "mismatch" in message
    assert len(list(tmp_path.glob("*.json"))) == 1
    assert policy.handle(RuntimeError("inference failure")) == (False, "")


@pytest.mark.parametrize("backend", ["v1", "v2"])
def test_recovery_does_not_swallow_step_cap(tmp_path, backend):
    buffer = ReplayBuffer(5000)
    config = NativeSelfPlayCoordinatorConfig(num_workers=1, checkpoint_path="unused",
        worker_config=worker_config(max_steps=1), device="cpu")
    cls, evaluator = backend_classes(backend)
    owner = cls(replay_buffer=buffer, config=config, evaluator=evaluator,
                max_rejected_games=100, failure_dir=tmp_path)
    with pytest.raises(RuntimeError, match="maximum game length"):
        owner.run_block(num_games=1, seed_start=0)
    assert not buffer.buffer and not buffer.games and not list(tmp_path.glob("*.json"))
