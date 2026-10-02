"""Batch protocol, exact search/replay parity, failures, and real V6 training."""
from dataclasses import replace
from pathlib import Path
import json
import numpy as np
import pytest
import torch

from splendor_rust_v2 import RustArena
from splendor_v1.env.env import SplendorEnv
from splendor_v1.mcts_batched.multiprocess_self_play_worker import (
    SelfPlayGameJob, make_multiprocess_mcts, make_model_replay_generator,
)
from splendor_v1.rust_engine import from_python as old_from_python
from splendor_v1.rust_engine.mcts import RustMCTS
from splendor_v1.rust_engine.test_self_play import worker_config, compare_samples
from splendor_v1.rust_engine.test_search_parity import DeterministicEvaluator
from splendor_v1.rust_engine_v2 import from_python, reset
from splendor_v1.rust_engine_v2.arena import ArenaSearch, decode_batch, response_bytes
from splendor_v1.rust_engine_v2.coordinator import (
    NativeSelfPlayCoordinator, NativeSelfPlayCoordinatorConfig, NativeSelfPlayError,
)
from splendor_v1.rust_engine_v2.self_play_settings import EXECUTION, native_search_settings
from splendor_v1.training_v2.replay_buffer import ReplayBuffer


class PackedArraysEvaluator:
    def __init__(self):
        self.evaluator = DeterministicEvaluator()
        self.batch_sizes = []

    def evaluate_packed(self, observations, actions, mask):
        self.batch_sizes.append(len(observations))
        output = np.zeros((len(observations), actions.shape[1] + 1), dtype=np.float64)
        for row in range(len(observations)):
            priors, value = self.evaluator.evaluate_arrays(observations[row], actions[row, mask[row]])
            output[row, :len(priors)] = priors
            output[row, -1] = value
        return output


def owner(buffer, evaluator, workers=3, batch_size=None, config=None, **kwargs):
    config = NativeSelfPlayCoordinatorConfig(num_workers=workers, checkpoint_path="unused",
        worker_config=config or worker_config(), device="cpu", max_batch_size=batch_size or workers)
    return NativeSelfPlayCoordinator(replay_buffer=buffer, config=config, evaluator=evaluator,
                                     verbose=False, **kwargs)


@pytest.mark.parametrize("cap", [1, 2, 3])
def test_bulk_requests_and_reused_trees_match_v1(cap):
    config = worker_config()
    arena = RustArena(3)
    references, sessions = [], []
    for seed in range(3):
        env = SplendorEnv(); env.reset(seed=seed)
        settings = native_search_settings(config.search)
        reference = RustMCTS(old_from_python(env.state), **settings)
        a, b = np.random.SeedSequence(seed).spawn(2)
        reference.rng, reference.action_rng = np.random.default_rng(a), np.random.default_rng(b)
        references.append(reference)
        sessions.append(ArenaSearch(arena, seed, from_python(env.state), config.search, seed))
    evaluator = PackedArraysEvaluator()
    for _ in range(10):
        for reference, session in zip(references, sessions):
            reference.begin(add_root_noise=True); session.begin(add_root_noise=True)
        ready_set = set()
        while len(ready_set) < 3:
            token, slots, ready, obs, ids, mask = decode_batch(arena.gather(cap))
            for slot in ready:
                assert references[slot].next_request() is None
                assert json.loads(arena.tree_json(slot)) == json.loads(references[slot].native.tree_json())
                ready_set.add(slot)
            if not slots: continue
            for row, slot in enumerate(slots):
                expected_obs, expected_ids = references[slot].next_request()
                np.testing.assert_array_equal(expected_obs, obs[row])
                np.testing.assert_array_equal(expected_ids, ids[row, mask[row]])
            output = evaluator.evaluate_packed(obs, ids, mask)
            for row, slot in enumerate(slots):
                references[slot].respond(output[row, :mask[row].sum()], output[row, -1])
            payload, double = response_bytes(output, len(slots), ids.shape[1])
            arena.respond(token, payload, double)
        for reference, session in zip(references, sessions):
            action = session.select_action(1.0)
            assert action == reference.select_action(1.0)
            assert session.advance(action) == reference.advance(action)
            assert session.root_state().snapshot_json() == reference.root_state().snapshot_json()
    assert max(evaluator.batch_sizes) <= cap


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_complete_replay_exactly_matches_python(seed):
    reference = ReplayBuffer(5000)
    config = worker_config()
    mcts = make_multiprocess_mcts(evaluator=DeterministicEvaluator(), search_config=config.search)
    generator = make_model_replay_generator(env=SplendorEnv(), mcts=mcts, replay_buffer=reference,
                                           max_game_steps=config.search.max_game_steps)
    expected = generator.generate_game(seed=seed, split="train", model_generation=4,
                                       model_checkpoint=config.model_checkpoint_label)
    buffer = ReplayBuffer(5000)
    summary = owner(buffer, PackedArraysEvaluator(), workers=1).run_block(num_games=1, seed_start=seed)
    assert summary["committed_games"] == 1 and len(buffer) == len(reference)
    for a, b in zip(reference.buffer, buffer.buffer): compare_samples(a, b)
    metadata = next(iter(buffer.games.values()))
    for key in ("winner_ids", "final_scores", "final_turn_number", "seed", "split", "search"):
        assert metadata[key] == expected["game_metadata"][key]


def test_replenishes_slots_caps_batches_and_preserves_splits():
    buffer, evaluator = ReplayBuffer(5000), PackedArraysEvaluator()
    progress, checkpoints = [], []
    factory = lambda offset, gid, seed: SelfPlayGameJob(gid, seed,
        split="train" if offset % 2 == 0 else "val", extra_game_metadata={"game_index": gid})
    summary = owner(buffer, evaluator, workers=3, batch_size=2).run_block(
        num_games=4, seed_start=0, game_id_start=100, job_factory=factory,
        progress_callback=progress.append, checkpoint_every_games=2, checkpoint_callback=checkpoints.append)
    assert summary["committed_games"] == summary["submitted_games"] == 4
    assert summary["split_games"] == {"train": 2, "val": 2}
    assert summary["pipeline_profile"]["completed_decisions"] == len(buffer)
    assert summary["gpu_server"]["total_inference_positions"] == sum(evaluator.batch_sizes)
    assert max(evaluator.batch_sizes) == 2 and len(progress) == 4 and len(checkpoints) == 2
    assert sorted(g["game_index"] for g in buffer.games.values()) == [100, 101, 102, 103]
    assert all(g["self_play_execution"] == EXECUTION for g in buffer.games.values())
    assert all(g["inference"]["precision"] == "fp32" for g in buffer.games.values())


@pytest.mark.parametrize("kind", ["cap", "dead_end", "inference"])
def test_failed_games_never_commit_partial_replay(kind, tmp_path):
    buffer = ReplayBuffer(5000)
    evaluator = PackedArraysEvaluator()
    config = worker_config(max_steps=1) if kind == "cap" else worker_config()
    if kind == "inference":
        def fail(*args): raise RuntimeError("test inference failure")
        evaluator.evaluate_packed = fail
    coordinator = owner(buffer, evaluator, workers=1, config=config, failure_dir=tmp_path)
    with pytest.raises((NativeSelfPlayError, RuntimeError)):
        coordinator.run_block(num_games=1, seed_start=42)
    assert not buffer.buffer and not buffer.games and coordinator.arena is None
    with pytest.raises(NativeSelfPlayError): coordinator.run_block(num_games=1, seed_start=0)


def test_protocol_rejects_stale_and_malformed_responses_atomically():
    arena = RustArena(2)
    for slot in range(2): arena.add(slot, reset(slot)); arena.begin(slot)
    token, slots, ready, obs, actions, mask = decode_batch(arena.gather(2))
    before = [arena.tree_json(slot) for slot in slots]
    output = PackedArraysEvaluator().evaluate_packed(obs, actions, mask)
    payload, double = response_bytes(output, len(slots), actions.shape[1])
    with pytest.raises(ValueError): arena.respond(token + 1, payload, double)
    with pytest.raises(ValueError): arena.respond(token, payload[:-1], double)
    with pytest.raises(ValueError): arena.remove(slots[0])
    with pytest.raises(ValueError): arena.advance(slots[0], int(actions[0, 0]))
    invalid = output.copy(); invalid[-1, -1] = np.nan
    with pytest.raises(ValueError): arena.respond(token, invalid.tobytes(), True)
    assert [arena.tree_json(slot) for slot in slots] == before
    arena.respond(token, payload, double)
    with pytest.raises(ValueError): arena.respond(token, payload, double)


def test_128_games_gather_in_one_native_call():
    arena = RustArena(128)
    state = reset(0)
    config = json.dumps({"simulations": 1, "adaptive_simulations": False})
    for slot in range(128):
        arena.add(slot, state, config); arena.begin(slot)
    token, slots, ready, obs, actions, mask = decode_batch(arena.gather(128))
    assert slots == list(range(128)) and not ready and obs.shape == (128, 258)
    output = PackedArraysEvaluator().evaluate_packed(obs, actions, mask)
    payload, double = response_bytes(output, len(slots), actions.shape[1])
    arena.respond(token, payload, double)
    assert decode_batch(arena.gather(128))[2] == list(range(128))


def test_training_launcher_overrides_only_runtime_coordinator(monkeypatch):
    from splendor_v1.training_v6 import run_training_v6 as v6
    from splendor_v1.rust_engine_v2 import run_training
    original = v6.MultiprocessSelfPlayCoordinator
    before = (v6.NUM_SELF_PLAY_WORKERS, v6.GPU_MAX_BATCH_SIZE, v6.RESUME_CHECKPOINT_PATH)
    calls = []
    def main():
        config = NativeSelfPlayCoordinatorConfig(num_workers=16, checkpoint_path="unused", device="cpu")
        coordinator = v6.MultiprocessSelfPlayCoordinator(replay_buffer=ReplayBuffer(10), config=config)
        assert coordinator.config.num_workers == 128 and coordinator.config.max_batch_size == 64
        assert config.num_workers == 16 and coordinator.options.precision == "bf16"
        calls.append(True)
    monkeypatch.setattr(v6, "main", main)
    run_training.main(["--games-in-flight", "128", "--batch-size", "64", "--precision", "bf16"])
    assert calls == [True] and v6.MultiprocessSelfPlayCoordinator is original
    assert before == (v6.NUM_SELF_PLAY_WORKERS, v6.GPU_MAX_BATCH_SIZE, v6.RESUME_CHECKPOINT_PATH)


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
