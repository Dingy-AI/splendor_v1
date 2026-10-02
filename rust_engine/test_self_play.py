"""Complete-game replay parity, threaded batching, failure handling, and training."""
import numpy as np
import pytest
import torch
from dataclasses import replace
from pathlib import Path

from splendor_v1.env.env import SplendorEnv
from splendor_v1.mcts_batched.multiprocess_self_play_worker import (
    SelfPlayGameJob, SelfPlaySearchConfig, SelfPlayWorkerConfig,
    make_model_replay_generator, make_multiprocess_mcts,
)
from splendor_v1.rust_engine.coordinator import (
    NativeSelfPlayCoordinator, NativeSelfPlayCoordinatorConfig, NativeSelfPlayError,
)
from splendor_v1.rust_engine.self_play import run_native_game, EXECUTION
from splendor_v1.rust_engine.test_search_parity import DeterministicEvaluator
from splendor_v1.training_v2.replay_buffer import ReplayBuffer


class ArraysEvaluator:
    def __init__(self):
        self.deterministic = DeterministicEvaluator()
        self.batch_sizes = []

    def evaluate(self, observation, ids):
        return self.deterministic.evaluate_arrays(observation, ids)

    def evaluate_batch(self, requests):
        self.batch_sizes.append(len(requests))
        return [self.evaluate(*request) for request in requests]


def worker_config(simulations=40, max_steps=300):
    return SelfPlayWorkerConfig(search=SelfPlaySearchConfig(simulations=simulations,
        min_simulations=12, check_interval=5, target_visits_per_action=0.5,
        max_game_steps=max_steps), model_checkpoint_label="parity-test")


def compare_samples(expected, actual):
    assert expected.keys() == actual.keys()
    for key, value in expected.items():
        if isinstance(value, np.ndarray):
            assert value.dtype == actual[key].dtype, key
            np.testing.assert_array_equal(value, actual[key], err_msg=key)
        elif isinstance(value, float):
            assert actual[key] == pytest.approx(value, rel=0, abs=1e-12), key
        else:
            assert actual[key] == value, key


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_entire_replay_matches_python(seed):
    config = worker_config()
    reference_buffer, native_buffer = ReplayBuffer(5000), ReplayBuffer(5000)
    env = SplendorEnv()
    evaluator = DeterministicEvaluator()
    mcts = make_multiprocess_mcts(evaluator=evaluator, search_config=config.search)
    generator = make_model_replay_generator(env=env, mcts=mcts, replay_buffer=reference_buffer,
                                           max_game_steps=config.search.max_game_steps)
    expected = generator.generate_game(seed=seed, split="train", model_generation=4,
                                       model_checkpoint=config.model_checkpoint_label)
    actual = run_native_game(worker_id=0, evaluator=ArraysEvaluator(),
        job=SelfPlayGameJob(game_id=seed, seed=seed, split="train"), config=config)
    native_buffer.add_game(actual.samples, actual.game_metadata)
    assert len(reference_buffer) == len(native_buffer) == actual.num_positions
    for py, native in zip(reference_buffer.buffer, native_buffer.buffer):
        compare_samples(py, native)
    for key in ["winner_ids", "final_scores", "final_turn_number", "seed", "split", "search"]:
        assert actual.game_metadata[key] == expected["game_metadata"][key], key


def coordinator(replay, evaluator, *, workers=3, config=None, batch_size=None):
    cfg = NativeSelfPlayCoordinatorConfig(num_workers=workers,
        checkpoint_path="unused-for-deterministic-evaluator", worker_config=config or worker_config(),
        device="cpu", max_batch_size=batch_size or workers, batch_wait_ms=2,
        game_result_timeout_s=30)
    return NativeSelfPlayCoordinator(replay_buffer=replay, config=cfg, evaluator=evaluator)


def test_threaded_batching_splits_and_completion_order():
    buffer = ReplayBuffer(5000)
    evaluator = ArraysEvaluator()
    progress, checkpoints = [], []
    jobs = lambda offset, gid, seed: SelfPlayGameJob(gid, seed,
        split="train" if offset % 2 == 0 else "val", extra_game_metadata={"game_index": gid})
    owner = coordinator(buffer, evaluator)
    summary = owner.run_block(num_games=4, seed_start=0, game_id_start=100, job_factory=jobs,
        progress_callback=progress.append, checkpoint_every_games=2, checkpoint_callback=checkpoints.append)
    assert summary["committed_games"] == summary["submitted_games"] == 4
    assert summary["split_games"] == {"train": 2, "val": 2}
    assert len(checkpoints) == 2 and len(progress) == 4
    assert len(buffer.games) == 4
    assert max(evaluator.batch_sizes) == 3
    assert summary["gpu_server"]["max_observed_batch_size"] == 3
    assert summary["gpu_server"]["total_inference_positions"] == sum(evaluator.batch_sizes)
    assert summary["self_play_execution"] == EXECUTION
    assert sorted(g["game_index"] for g in buffer.games.values()) == [100, 101, 102, 103]
    for gid, game in buffer.games.items():
        assert game["self_play_execution"] == EXECUTION
        assert game["multiprocess_self_play"] is False
        assert game["multiprocess_coordinator"] is False
        assert game["winner_ids"] and game["num_positions"] > 0
        rows = [s for s in buffer.buffer if s["game_id"] == gid]
        assert len(rows) == game["num_positions"]
        assert rows[-1]["terminated_after_action"]
        assert not any(s["terminated_after_action"] for s in rows[:-1])
    # Independent game RNG streams make batching/worker scheduling irrelevant for a deterministic evaluator.
    for seed in range(4):
        expected = run_native_game(worker_id=0, evaluator=ArraysEvaluator(),
            job=SelfPlayGameJob(seed, seed), config=worker_config())
        game = next(g for g in buffer.games.values() if g["seed"] == seed)
        rows = [s for s in buffer.buffer if s["game_id"] == game["game_id"]]
        for a, b in zip(expected.samples, rows):
            b = dict(b)
            b.pop("game_id")
            compare_samples(a, b)


def test_batch_cap_and_single_worker():
    evaluator = ArraysEvaluator()
    owner = coordinator(ReplayBuffer(5000), evaluator, workers=2, batch_size=1)
    summary = owner.run_block(num_games=2, seed_start=0)
    assert summary["committed_games"] == 2
    assert set(evaluator.batch_sizes) == {1}


def test_failure_never_commits_partial_game():
    buffer = ReplayBuffer(1000)
    owner = coordinator(buffer, ArraysEvaluator(), workers=1, config=worker_config(max_steps=1))
    with pytest.raises(NativeSelfPlayError, match="seed=42.*maximum game length"):
        owner.run_block(num_games=1, seed_start=42)
    assert not buffer.buffer and not buffer.games
    with pytest.raises(NativeSelfPlayError):
        owner.run_block(num_games=1, seed_start=0)


def test_existing_dead_end_game_is_rejected_by_both_backends():
    config = worker_config()
    buffer = ReplayBuffer(5000)
    mcts = make_multiprocess_mcts(evaluator=DeterministicEvaluator(), search_config=config.search)
    generator = make_model_replay_generator(env=SplendorEnv(), mcts=mcts, replay_buffer=buffer,
                                           max_game_steps=config.search.max_game_steps)
    with pytest.raises(RuntimeError, match="No legal actions"):
        generator.generate_game(seed=42)
    with pytest.raises(RuntimeError, match="No legal actions"):
        run_native_game(worker_id=0, evaluator=ArraysEvaluator(),
                        job=SelfPlayGameJob(0, 42), config=config)
    assert not buffer.buffer and not buffer.games


@pytest.mark.parametrize("failure", ["exception", "wrong_batch", "invalid_value"])
def test_evaluator_failure_shuts_down_workers(failure):
    class BadEvaluator:
        def evaluate_batch(self, requests):
            if failure == "exception": raise RuntimeError("test inference failure")
            if failure == "wrong_batch": return []
            return [(np.ones(len(ids)) / len(ids), float("nan")) for _, ids in requests]
    buffer = ReplayBuffer(1000)
    owner = coordinator(buffer, BadEvaluator(), workers=2)
    with pytest.raises((NativeSelfPlayError, RuntimeError)):
        owner.run_block(num_games=2, seed_start=0)
    assert not buffer.buffer and not buffer.games
    assert owner._pool is None


def test_training_launcher_preserves_configuration(monkeypatch):
    from splendor_v1.rust_engine import run_training
    from splendor_v1.training_v6 import run_training_v6 as v6
    original = v6.MultiprocessSelfPlayCoordinator
    before = (v6.SIMULATIONS, v6.NUM_SELF_PLAY_WORKERS, v6.RESUME_CHECKPOINT_PATH, v6.BATCH_SIZE)
    called = []
    def main():
        cfg = NativeSelfPlayCoordinatorConfig(num_workers=1, checkpoint_path="unused", device="cpu")
        owner = v6.MultiprocessSelfPlayCoordinator(replay_buffer=ReplayBuffer(10), config=cfg)
        assert isinstance(owner, NativeSelfPlayCoordinator)
        assert owner.failures.limit == 10
        called.append(True)
    monkeypatch.setattr(v6, "main", main)
    run_training.main([])
    assert called == [True]
    assert original is v6.MultiprocessSelfPlayCoordinator
    assert before == (v6.SIMULATIONS, v6.NUM_SELF_PLAY_WORKERS, v6.RESUME_CHECKPOINT_PATH, v6.BATCH_SIZE)


def test_native_samples_train_and_roundtrip(tmp_path):
    from splendor_v1.training_v5.train_v5 import build_training_batch, train_network
    from splendor_v1.network.model_4_legal_scorer import SplendorNetwork
    torch.set_num_threads(1)
    buffer = ReplayBuffer(5000)
    result = run_native_game(worker_id=0, evaluator=ArraysEvaluator(),
        job=SelfPlayGameJob(0, 0, split="train"), config=worker_config())
    buffer.add_game(result.samples, result.game_metadata)
    batch = build_training_batch(buffer.buffer[:16], buffer)
    assert batch["observations"].shape == (16, 258)
    assert torch.allclose(batch["target_policy"].sum(dim=1), torch.ones(16))
    assert batch["target_wdl"].min() >= 0 and batch["target_wdl"].max() <= 2
    model = SplendorNetwork()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    before = model.action_embedding.weight.detach().clone()
    metrics = train_network(model=model, replay_buffer=buffer, optimizer=optimizer,
        batch_size=16, training_steps=1, split="train")
    assert np.isfinite(metrics["average_total_loss"])
    assert not torch.equal(before, model.action_embedding.weight)
    path = tmp_path / "native-replay.pkl"
    buffer.save(str(path))
    from splendor_v1.training_v6.run_training_v6 import load_rich_replay_buffer
    restored = load_rich_replay_buffer(str(path))
    assert len(restored) == len(buffer)
    assert restored.games == buffer.games
    compare_samples(restored.buffer[0], buffer.buffer[0])


def test_full_v6_iteration_with_native_backend(tmp_path, monkeypatch):
    from splendor_v1.training_v6 import run_training_v6 as v6
    from splendor_v1.network.model_4_legal_scorer import SplendorNetwork
    torch.set_num_threads(1)
    checkpoint_path = Path(__file__).parents[1] / "training_v6/data/model4_v6_inference_latest.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = SplendorNetwork()
    model.load_state_dict(checkpoint["model_state_dict"])
    before = model.action_embedding.weight.detach().clone()
    settings = dict(NUM_ITERATIONS=2, SELF_PLAY_GAMES_PER_ITERATION=2, NUM_SELF_PLAY_WORKERS=2,
        GPU_MAX_BATCH_SIZE=2, SIMULATIONS=24, BATCH_SIZE=16, TRAINING_RATIO=0.01,
        MIN_VALIDATION_GAMES=1, MIN_VALIDATION_POSITIONS=1, VALIDATION_STEPS=1,
        BASE_SEED=10000, CHECKPOINT_EVERY_GAMES=2, device=torch.device("cpu"),
        INFERENCE_SNAPSHOT_PATH=str(tmp_path / "inference.pt"),
        OUTPUT_CHECKPOINT_DIR=str(tmp_path / "checkpoints"),
        OUTPUT_REPLAY_PATH=str(tmp_path / "replay.pkl"),
        MultiprocessSelfPlayCoordinator=NativeSelfPlayCoordinator)
    for name, value in settings.items(): monkeypatch.setattr(v6, name, value)
    generations = []
    load_evaluator = NativeSelfPlayCoordinator._load_evaluator
    def record_loaded_model(owner):
        load_evaluator(owner)
        generations.append(owner.evaluator.model.action_embedding.weight.detach().cpu().clone())
    monkeypatch.setattr(NativeSelfPlayCoordinator, "_load_evaluator", record_loaded_model)
    original_factory = v6.make_job_factory
    def split_factory(**kwargs):
        factory = original_factory(**kwargs)
        return lambda offset, gid, seed: replace(factory(offset, gid, seed),
            split="train" if offset == 0 else "val")
    monkeypatch.setattr(v6, "make_job_factory", split_factory)
    optimizer = v6.create_optimizer(model)
    scheduler = v6.create_scheduler(optimizer)
    buffer = ReplayBuffer(5000)
    history = v6.run_training_v6(model=model, optimizer=optimizer, scheduler=scheduler,
        replay_buffer=buffer, active_checkpoint_path=str(checkpoint_path),
        starting_games_played=0, starting_games_attempted=0)
    assert len(history) == 2 and all(np.isfinite(h["average_total_loss"]) for h in history)
    assert not torch.equal(before, model.action_embedding.weight)
    assert len(generations) == 2 and torch.equal(generations[0], before)
    assert not torch.equal(generations[0], generations[1])
    assert scheduler.last_epoch == 2
    assert len(buffer.games) == 4
    assert {g["split"] for g in buffer.games.values()} == {"train", "val"}
    assert all(g["self_play_execution"] == EXECUTION for g in buffer.games.values())
    final = torch.load(tmp_path / "checkpoints/model_4_games_last.pt", map_location="cpu", weights_only=True)
    assert final["games_played"] == 4
    assert final["optimizer_state_dict"]["state"]
    assert final["scheduler_state_dict"]["last_epoch"] == 2
    inference = torch.load(tmp_path / "inference.pt", map_location="cpu", weights_only=True)
    assert inference["games_played"] == 4
    restored = v6.load_rich_replay_buffer(str(tmp_path / "replay.pkl"))
    assert len(restored) == len(buffer)
    assert v6.infer_games_attempted_from_replay(restored, 0) == 4
