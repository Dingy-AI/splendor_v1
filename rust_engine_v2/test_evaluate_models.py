"""Differential paired matches, model ownership, failure accounting and CLI."""
from collections import Counter
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from splendor_v1.env.env import SplendorEnv
from splendor_v1.evaluation.run_parallel_evaluation_v6 import (
    EvaluationSearchConfig, make_evaluation_mcts, find_selected_child, greedy_action_from_root,
)
from splendor_v1.mcts_batched.multiprocess_self_play_worker import SelfPlaySearchConfig
from splendor_v1.rust_engine.test_search_parity import DeterministicEvaluator
from splendor_v1.rust_engine_v2.evaluate_models import (
    MatchConfig, MatchGame, MatchRunner, MatchError, build_summary, main,
)


class ModelEvaluator(DeterministicEvaluator):
    def __init__(self, label):
        super().__init__()
        self.label = label
        self.leaf_players = []
        self.calls = []

    def evaluate_arrays(self, observation, ids):
        priors, value = super().evaluate_arrays(observation, ids)
        if self.label == "B":
            weights = ((np.asarray(ids) * 11 + 7) % 19 + 1).astype(np.float32)
            priors = weights / weights.sum()
            value *= -0.7
        return priors, value

    def evaluate(self, env, state, legal_actions):
        self.leaf_players.append(state.current_player)
        return super().evaluate(env, state, legal_actions)

    def evaluate_packed(self, observations, actions, mask):
        output = np.zeros((len(observations), actions.shape[1] + 1), dtype=np.float64)
        for row, observation in enumerate(observations):
            ids = actions[row, mask[row]]
            priors, value = self.evaluate_arrays(observation, ids)
            self.calls.append((observation.copy(), ids.copy()))
            output[row, :len(ids)] = priors
            output[row, -1] = value
        return output


def config(**kwargs):
    return MatchConfig(games=4, base_seed=0, search=SelfPlaySearchConfig(simulations=40,
        min_simulations=12, check_interval=5, target_visits_per_action=0.5),
        include_moves=True, **kwargs)


def python_game(seed, a_player, cfg, evaluators):
    env = SplendorEnv(); env.reset(seed=seed)
    fields = {name: getattr(cfg.search, name) for name in EvaluationSearchConfig.__dataclass_fields__}
    searches = {label: make_evaluation_mcts(evaluator=evaluator, search=EvaluationSearchConfig(**fields))
                for label, evaluator in evaluators.items()}
    root = None
    actions, counts, sims = [], Counter(), Counter()
    reasons = {label: Counter() for label in ("A", "B")}
    while not env.state.game_over:
        if len(actions) >= cfg.search.max_game_steps:
            status = "step_cap"; break
        if not env._legal_actions(env.state):
            status = "dead_end"; break
        player = env.state.current_player
        label = "A" if player == a_player else "B"
        mcts = searches[label]
        fallback, root = mcts.search(env, env.state, root=root, return_root=True,
                                     teacher_mode=False, add_root_noise=False)
        action = greedy_action_from_root(root, fallback)
        counts[label] += 1
        sims[label] += mcts.last_search_metadata["actual_simulations"]
        reasons[label][mcts.last_search_metadata["stop_reason"]] += 1
        child = find_selected_child(root, action)
        actions.append(env.action_to_id(action))
        _, _, terminated, _, _ = env.step(action)
        if terminated:
            status = "completed"; break
        root = child if env.state.current_player == player else None
        if root is not None: root.parent = None
    winners = env.state.winners if status == "completed" else None
    outcome = None if winners is None else "draw" if len(winners) == 2 else "A" if a_player in winners else "B"
    return dict(status=status, outcome=outcome, winner_ids=winners,
        final_scores=[p.points for p in env.state.players], final_turn_number=env.state.turn_number,
        positions=len(actions), action_ids=actions, searches=dict(counts), actual_simulations=dict(sims),
        stop_reasons={label: dict(values) for label, values in reasons.items()})


@pytest.mark.parametrize("workers,batch", [(1, 1), (3, 1), (3, 3)])
def test_complete_matches_match_python_v6(workers, batch, tmp_path):
    cfg = config(workers=workers, batch_size=batch, keep_going=True, failure_dir=str(tmp_path))
    evaluators = {label: ModelEvaluator(label) for label in ("A", "B")}
    runner = MatchRunner(evaluators, cfg, verbose=False)
    report = runner.run()
    reference_requests = {label: [] for label in ("A", "B")}
    for game in report["game_results"]:
        references = {label: ModelEvaluator(label) for label in ("A", "B")}
        expected = python_game(game["seed"], game["model_a_player"], cfg, references)
        for label, evaluator in references.items():
            reference_requests[label].extend((obs, np.asarray(ids, dtype=np.int64))
                for obs, ids, _, _ in evaluator.requests)
        for key, value in expected.items(): assert game[key] == value, (key, game["game_id"])
        if workers == 1:
            assert set(references["A"].leaf_players) == {0, 1}
            assert set(references["B"].leaf_players) == {0, 1}
    assert [r["seed"] for r in report["game_results"]] == [0, 0, 1, 1]
    assert [r["model_a_player"] for r in report["game_results"]] == [0, 1, 0, 1]
    assert report["resolved_games"] == report["requested_games"] == 4
    assert report["tree_reuse"] == "same_real_player_only"
    assert report["temperature"] == 0 and report["root_noise"] is False
    assert runner.arena is None
    for label in ("A", "B"):
        signature = lambda requests: Counter((obs.tobytes(), ids.tobytes()) for obs, ids in requests)
        assert signature(evaluators[label].calls) == signature(reference_requests[label])


def test_identical_models_are_balanced_by_paired_seats(tmp_path):
    cfg = config(workers=3, batch_size=2, failure_dir=str(tmp_path))
    report = MatchRunner({label: ModelEvaluator("A") for label in ("A", "B")}, cfg, verbose=False).run()
    assert report["completed_games"] == 4 and report["completed_pairs"] == 2
    assert report["model_a_wins"] == report["model_b_wins"]
    assert report["complete_pair_results"]["model_a_match_score"] == 0.5


def test_routes_by_search_owner_and_restores_mixed_row_order():
    seen = {"A": [], "B": []}
    class Sentinel:
        def __init__(self, label): self.label = label
        def evaluate_packed(self, obs, ids, mask):
            seen[self.label].extend(obs[:, 0].tolist())
            output = np.zeros((len(obs), ids.shape[1] + 1), dtype=np.float32)
            output[:, :-1] = mask / mask.sum(axis=1, keepdims=True)
            output[:, -1] = 0.25 if self.label == "A" else -0.5
            return output
    runner = MatchRunner({label: Sentinel(label) for label in seen}, MatchConfig(games=2), verbose=False)
    slots = [5, 2, 9, 0]
    active = {s: SimpleNamespace(owner=l) for s, l in zip(slots, ["B", "A", "A", "B"])}
    obs = np.zeros((4, 258), dtype=np.float32); obs[:, 0] = [0, 1, 2, 3]
    ids = np.tile(np.arange(3), (4, 1))
    mask = np.array([[True, False, False], [True, True, True], [True, False, False], [True, True, False]])
    output = runner.evaluate_batch(slots, active, obs, ids, mask)
    assert seen == {"A": [1, 2], "B": [0, 3]}
    np.testing.assert_array_equal(output[:, -1], [-0.5, 0.25, 0.25, -0.5])
    np.testing.assert_allclose(output[:, :-1].sum(axis=1), 1)
    assert not output[:, :-1][~mask].any()


@pytest.mark.parametrize("keep_going", [False, True])
def test_step_caps_are_saved_and_never_count_as_draws(tmp_path, keep_going):
    cfg = config(workers=1, batch_size=1, keep_going=keep_going, failure_dir=str(tmp_path))
    cfg.games = 2; cfg.search.max_game_steps = 1
    runner = MatchRunner({label: ModelEvaluator(label) for label in ("A", "B")}, cfg, verbose=False)
    if keep_going:
        report = runner.run()
        assert report["status"] == "completed_with_failures" and report["failed_games"] == 2
    else:
        with pytest.raises(MatchError, match="max_game_steps"): runner.run()
        report = runner.report()
        assert report["status"] == "failed" and report["failed_games"] == 1
    assert report["completed_games"] == report["draws"] == report["completed_pairs"] == 0
    assert report["model_a_match_score"] is None
    assert len(list(tmp_path.glob("*.json"))) == report["failed_games"]
    assert runner.arena is None


@pytest.mark.parametrize("failure", ["nan", "shape", "exception", "negative_prior"])
def test_inference_failures_stop_even_with_keep_going(failure):
    class Bad(ModelEvaluator):
        def evaluate_packed(self, obs, ids, mask):
            if failure == "exception": raise RuntimeError("test inference error")
            output = super().evaluate_packed(obs, ids, mask)
            if failure == "nan": output[0, -1] = np.nan
            if failure == "shape": output = output[:, :-1]
            if failure == "negative_prior": output[0, 0] = -1
            return output
    runner = MatchRunner({"A": Bad("A"), "B": ModelEvaluator("B")},
                         config(workers=2, keep_going=True), verbose=False)
    with pytest.raises((MatchError, RuntimeError, ValueError)): runner.run()
    assert runner.status == "failed" and runner.arena is None
    assert runner.report()["completed_games"] == 0


def test_only_complete_seat_pairs_enter_paired_score():
    def row(gid, status, outcome):
        return dict(game_id=gid, pair_id=gid // 2, model_a_player=gid % 2, status=status,
                    outcome=outcome, positions=1, searches={}, actual_simulations={}, stop_reasons={"A": {}, "B": {}})
    summary = build_summary([row(0, "completed", "A"), row(1, "step_cap", None),
                             row(2, "completed", "B"), row(3, "completed", "draw")])
    assert summary["completed_games"] == 3 and summary["failed_games"] == 1
    assert summary["completed_pairs"] == 1
    assert summary["model_a_match_score"] == 0.5
    assert summary["complete_pair_results"]["model_a_match_score"] == 0.25


def test_cli_two_real_checkpoints_readonly(tmp_path):
    source = Path(__file__).parents[1] / "training_v6/data/model4_v6_inference_latest.pt"
    a = tmp_path / "a.pt"; b = tmp_path / "b.pt"
    a.write_bytes(source.read_bytes())
    checkpoint = torch.load(a, map_location="cpu", weights_only=True)
    checkpoint["model_state_dict"]["wdl_head.bias"] = checkpoint["model_state_dict"]["wdl_head.bias"].clone()
    checkpoint["model_state_dict"]["wdl_head.bias"][0] += 0.01
    torch.save(checkpoint, b)
    hashes = [hashlib.sha256(path.read_bytes()).hexdigest() for path in (a, b)]
    report_path = tmp_path / "match.json"
    report = main(["--model-a", str(a), "--model-b", str(b), "--games", "2",
        "--workers", "2", "--batch-size", "2", "--device", "cpu", "--base-seed", "10000",
        "--simulations", "24", "--min-simulations", "12", "--check-interval", "5",
        "--target-visits-per-action", "0.5", "--report", str(report_path)])
    assert report["completed_games"] == 2 and report["completed_pairs"] == 1
    assert report["failed_games"] == 0 and report["inference_options"]["precision"] == "fp32"
    assert [report["models"][label]["checkpoint_sha256"] for label in ("A", "B")] == hashes
    assert [hashlib.sha256(path.read_bytes()).hexdigest() for path in (a, b)] == hashes
    assert json.loads(report_path.read_text()) == report


@pytest.mark.parametrize("games", [0, 1, 3])
def test_requires_even_game_count(games):
    with pytest.raises(ValueError, match="even"): MatchConfig(games=games).validate()


def test_cli_rejects_checkpoint_as_report(tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"unchanged")
    with pytest.raises(SystemExit):
        main(["--model-a", str(checkpoint), "--model-b", str(checkpoint), "--report", str(checkpoint)])
    assert checkpoint.read_bytes() == b"unchanged"


def test_fixed_simulations_apply_to_both_models(tmp_path):
    cfg = config(workers=2, batch_size=2, keep_going=True, failure_dir=str(tmp_path))
    cfg.games = 2; cfg.search.max_game_steps = 2
    cfg.search.simulations = 4; cfg.adaptive_simulations = False
    report = MatchRunner({label: ModelEvaluator(label) for label in ("A", "B")}, cfg, verbose=False).run()
    assert not report["adaptive_simulations"]
    for game in report["game_results"]:
        assert game["status"] == "step_cap"
        assert set(game["searches"]) == {"A", "B"}
        for label in ("A", "B"):
            assert game["actual_simulations"][label] == 4 * game["searches"][label]


@pytest.mark.parametrize("keep_going", [False, True])
def test_shared_dead_end_preserves_original_paired_jobs(tmp_path, monkeypatch, keep_going):
    from splendor_v1.mcts_batched.multiprocess_self_play_worker import SelfPlayGameJob
    from splendor_v1.rust_engine.failures import EmptyActionError
    from splendor_v1.rust_engine.self_play import run_native_game
    from splendor_v1.rust_engine.test_self_play import ArraysEvaluator, worker_config
    # Obtain a real, reproducible rules dead-end. Force its legal action path
    # below to exercise evaluation's boundary, diagnosis and paired accounting.
    with pytest.raises(EmptyActionError) as captured:
        run_native_game(worker_id=0, evaluator=ArraysEvaluator(),
                        job=SelfPlayGameJob(0, 42), config=worker_config())
    actions = captured.value.diagnostic["action_ids"]
    original = MatchGame.finish_decision
    def play_known_path(game):
        game.search.select_action = lambda **kwargs: actions[len(game.actions)]
        return original(game)
    monkeypatch.setattr(MatchGame, "finish_decision", play_known_path)
    cfg = config(workers=2, batch_size=2, keep_going=keep_going, failure_dir=str(tmp_path))
    cfg.games = 2; cfg.base_seed = 42; cfg.search.simulations = 1
    runner = MatchRunner({label: ModelEvaluator(label) for label in ("A", "B")}, cfg, verbose=False)
    if keep_going:
        report = runner.run()
        assert report["failed_games"] == 2
        assert [row["seed"] for row in report["game_results"]] == [42, 42]
        assert [row["model_a_player"] for row in report["game_results"]] == [0, 1]
        assert report["status"] == "completed_with_failures"
    else:
        with pytest.raises(MatchError, match="No legal actions"): runner.run()
        report = runner.report()
        assert report["failed_games"] == 1 and report["status"] == "failed"
    assert report["submitted_games"] == 2  # No replacement seeds/jobs.
    assert report["draws"] == report["completed_games"] == report["completed_pairs"] == 0
    for row in report["game_results"]:
        assert row["status"] == "dead_end" and row["action_ids"] == actions
        diagnostic = json.loads(Path(row["diagnostic_path"]).read_text())
        assert diagnostic["trajectory_verification"]["shared_nonterminal_dead_end"]
        assert diagnostic["models"] == {label: {"name": label} for label in ("A", "B")}


def test_rules_mismatch_stops_even_with_keep_going(tmp_path, monkeypatch):
    from splendor_v1.rust_engine.failures import empty_action_error
    from splendor_v1.rust_engine_v2 import to_python
    def mismatch(game):
        env = SplendorEnv(); env.state = to_python(game.search.root_state())
        error = empty_action_error(env, game.diagnostic()["state"], len(game.actions))
        error.diagnostic.update(game.diagnostic())
        raise error
    monkeypatch.setattr(MatchGame, "finish_decision", mismatch)
    runner = MatchRunner({label: ModelEvaluator(label) for label in ("A", "B")},
        config(workers=2, keep_going=True, failure_dir=str(tmp_path)), verbose=False)
    with pytest.raises(MatchError): runner.run()
    report = runner.report()
    assert report["status"] == "failed"
    assert report["failure_reasons"] == {"rules_mismatch": 1}
    diagnostic = json.loads(Path(report["game_results"][0]["diagnostic_path"]).read_text())
    assert diagnostic["python_legal_action_ids"] and not diagnostic["shared_dead_end"]


def test_cli_saves_partial_report_on_failure(tmp_path):
    checkpoint = Path(__file__).parents[1] / "training_v6/data/model4_v6_inference_latest.pt"
    report_path = tmp_path / "partial.json"
    with pytest.raises(MatchError, match="max_game_steps"):
        main(["--model-a", str(checkpoint), "--model-b", str(checkpoint),
            "--games", "2", "--workers", "1", "--batch-size", "1", "--device", "cpu",
            "--simulations", "1", "--fixed-simulations", "--max-game-steps", "1",
            "--failure-dir", str(tmp_path / "failures"), "--report", str(report_path)])
    report = json.loads(report_path.read_text())
    assert report["status"] == "failed" and report["failed_games"] == 1
    assert report["resolved_games"] == 1 and report["unresolved_games"] == 1
    assert report["completed_games"] == report["draws"] == 0
    assert report["error"] and report["models"]["A"]["checkpoint_sha256"]
    assert not list(tmp_path.glob("*.tmp"))
