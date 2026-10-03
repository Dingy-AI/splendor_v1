"""Direct-opponent match parity, RNG isolation, accounting and CLI integrity."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import pytest

from splendor_v1.env.env import SplendorEnv
from splendor_v1.evaluation.run_parallel_evaluation_v6 import (
    EvaluationSearchConfig, make_evaluation_mcts, find_selected_child, greedy_action_from_root,
)
from splendor_v1.mcts_batched.multiprocess_self_play_worker import SelfPlaySearchConfig
from splendor_v1.rust_engine_v2 import reset
from splendor_v1.rust_engine_v2.evaluate_models import MatchError
from splendor_v1.rust_engine_v2.evaluate_heuristic import (
    HeuristicMatchConfig, HeuristicMatchGame, HeuristicMatchRunner, OpponentPolicy, main,
)
from splendor_v1.rust_engine_v2.test_evaluate_models import ModelEvaluator


def config(tmp_path, opponent, **kwargs):
    return HeuristicMatchConfig(games=2, workers=2, batch_size=2, base_seed=0,
        opponent=opponent, heuristic_rollouts=1, heuristic_max_rollout_steps=12,
        search=SelfPlaySearchConfig(simulations=16, min_simulations=4, check_interval=4,
                                   target_visits_per_action=0.2), keep_going=True,
        include_moves=True, failure_dir=str(tmp_path), **kwargs)


def python_game(seed, a_player, cfg, evaluator):
    env = SplendorEnv(); env.reset(seed=seed)
    fields = {name: getattr(cfg.search, name) for name in EvaluationSearchConfig.__dataclass_fields__}
    mcts = make_evaluation_mcts(evaluator=evaluator, search=EvaluationSearchConfig(**fields))
    rng = random.Random(seed)
    if cfg.opponent == "greedy":
        from splendor_v1.agents.greedy_agent import GreedyAgent
        opponent = GreedyAgent()
    elif cfg.opponent == "h3":
        from splendor_v1.agents.heuristic_agent_3 import HeuristicAgent3
        opponent = HeuristicAgent3()
    elif cfg.opponent == "h12":
        from splendor_v1.agents.heuristic_agent_12 import HeuristicAgent12
        opponent = HeuristicAgent12(num_rollouts=cfg.heuristic_rollouts,
            max_rollout_steps=cfg.heuristic_max_rollout_steps, random_seed=seed)
    root = None
    actions, searches, sims = [], Counter(), Counter()
    reasons = {"A": Counter(), "B": Counter()}
    heuristic_decisions = 0
    while not env.state.game_over:
        if len(actions) >= cfg.search.max_game_steps:
            status = "step_cap"; break
        legal = env._legal_actions(env.state)
        if not legal:
            status = "dead_end"; break
        player = env.state.current_player
        if player == a_player:
            fallback, root = mcts.search(env, env.state, root=root, return_root=True,
                                        teacher_mode=False, add_root_noise=False)
            action = greedy_action_from_root(root, fallback)
            searches["A"] += 1
            sims["A"] += mcts.last_search_metadata["actual_simulations"]
            reasons["A"][mcts.last_search_metadata["stop_reason"]] += 1
            child = find_selected_child(root, action)
        else:
            action = rng.choice(legal) if cfg.opponent == "random" else opponent.select_action(env, env.state)
            heuristic_decisions += 1
            child = None
        actions.append(env.action_to_id(action))
        _, _, terminated, _, _ = env.step(action)
        root = child if player == a_player and env.state.current_player == player else None
        if root is not None: root.parent = None
        if terminated:
            status = "completed"; break
    winners = env.state.winners if status == "completed" else None
    outcome = None if winners is None else "draw" if len(winners) == 2 else "A" if a_player in winners else "B"
    return dict(status=status, outcome=outcome, winner_ids=winners,
        final_scores=[p.points for p in env.state.players], final_turn_number=env.state.turn_number,
        action_ids=actions, positions=len(actions), searches=dict(searches), actual_simulations=dict(sims),
        stop_reasons={label: dict(counts) for label, counts in reasons.items()},
        heuristic_decisions=heuristic_decisions)


@pytest.mark.parametrize("opponent", ["random", "greedy", "h3", "h12"])
@pytest.mark.parametrize("workers,batch", [(1, 1), (3, 2)])
def test_complete_games_match_python(tmp_path, opponent, workers, batch):
    cfg = config(tmp_path, opponent)
    cfg.workers, cfg.batch_size = workers, batch
    evaluator = ModelEvaluator("A")
    report = HeuristicMatchRunner(evaluator, cfg, verbose=False).run()
    references = []
    for game in report["game_results"]:
        reference = ModelEvaluator("A")
        expected = python_game(game["seed"], game["model_a_player"], cfg, reference)
        for key, value in expected.items(): assert game[key] == value, (key, opponent, game["game_id"])
        references.extend((obs, np.asarray(ids, dtype=np.int64)) for obs, ids, _, _ in reference.requests)
        assert not game["searches"].get("B", 0) and not game["actual_simulations"].get("B", 0)
        assert game["searches"]["A"] + game["heuristic_decisions"] == game["positions"]
        assert set(reference.leaf_players) == {0, 1}
    signature = lambda requests: Counter((obs.tobytes(), ids.tobytes()) for obs, ids in requests)
    assert signature(evaluator.calls) == signature(references)
    assert report["submitted_games"] == report["resolved_games"] == 2
    assert report["completed_games"] == 2 and report["completed_pairs"] == 1
    assert report["match_type"] == "model_vs_heuristic"
    assert set(report["gpu_servers"]) == {"A"}
    assert report["model_wins"] == report["model_a_wins"]
    assert report["heuristic_wins"] == report["model_b_wins"]
    assert [g["seed"] for g in report["game_results"]] == [0, 0]
    assert [g["model_a_player"] for g in report["game_results"]] == [0, 1]
    assert report["model_match_score"] == report["complete_pair_results"]["model_a_match_score"]


@pytest.mark.parametrize("opponent", ["random", "h12"])
def test_agents_use_local_rng(tmp_path, opponent):
    cfg = config(tmp_path, opponent)
    first, second = OpponentPolicy(cfg, 42), OpponentPolicy(cfg, 42)
    global_state = random.getstate()
    native = reset(42)
    for _ in range(3):
        assert first.select_action(native) == second.select_action(native)
    assert random.getstate() == global_state


@pytest.mark.parametrize("failure", ["illegal", "exception"])
def test_opponent_failures_stop_even_with_keep_going(tmp_path, monkeypatch, failure):
    def broken(policy, native):
        if failure == "exception": raise RuntimeError("test opponent error")
        return 1138  # Illegal discard at the initial main decision.
    monkeypatch.setattr(OpponentPolicy, "select_action", broken)
    cfg = config(tmp_path, "random"); cfg.workers = 1
    runner = HeuristicMatchRunner(ModelEvaluator("A"), cfg, verbose=False)
    with pytest.raises(MatchError): runner.run()
    report = runner.report()
    assert report["status"] == "failed" and report["failure_reasons"] == {"error": 1}
    assert report["completed_games"] == report["draws"] == 0


def test_python_native_legal_mismatch_is_rejected(tmp_path, monkeypatch):
    policy = OpponentPolicy(config(tmp_path, "h3"), 0)
    monkeypatch.setattr(policy.env, "_legal_actions", lambda state: [])
    with pytest.raises(MatchError, match="legal actions disagree"): policy.select_action(reset(0))


@pytest.mark.parametrize("workers", [1, 2])
def test_direct_opponent_step_caps_are_not_draws(tmp_path, workers):
    cfg = config(tmp_path, "random"); cfg.search.max_game_steps = 1
    cfg.workers = workers
    runner = HeuristicMatchRunner(ModelEvaluator("A"), cfg, verbose=False)
    report = runner.run()
    assert report["status"] == "completed_with_failures"
    assert report["failure_reasons"] == {"step_cap": 2}
    assert report["draws"] == report["completed_games"] == report["completed_pairs"] == 0
    assert report["model_match_score"] is None
    for row in report["game_results"]:
        diagnostic = json.loads(Path(row["diagnostic_path"]).read_text())
        assert diagnostic["match_type"] == "model_vs_heuristic"
        assert diagnostic["opponent"]["mcts"] is False
        assert diagnostic["step_index"] == len(diagnostic["action_ids"]) == 1


def test_shared_opponent_dead_end_is_verified_without_replacements(tmp_path, monkeypatch):
    from splendor_v1.mcts_batched.multiprocess_self_play_worker import SelfPlayGameJob
    from splendor_v1.rust_engine.failures import EmptyActionError
    from splendor_v1.rust_engine.self_play import run_native_game
    from splendor_v1.rust_engine.test_self_play import ArraysEvaluator, worker_config
    with pytest.raises(EmptyActionError) as captured:
        run_native_game(worker_id=0, evaluator=ArraysEvaluator(),
                        job=SelfPlayGameJob(0, 42), config=worker_config())
    actions = captured.value.diagnostic["action_ids"]
    start, finish = HeuristicMatchGame.start_decision, HeuristicMatchGame.finish_decision
    def scripted_start(game):
        game.opponent.select_action = lambda native: actions[len(game.actions)]
        return start(game)
    def scripted_finish(game):
        game.search.select_action = lambda **kwargs: actions[len(game.actions)]
        return finish(game)
    monkeypatch.setattr(HeuristicMatchGame, "start_decision", scripted_start)
    monkeypatch.setattr(HeuristicMatchGame, "finish_decision", scripted_finish)
    cfg = config(tmp_path, "random"); cfg.base_seed = 42; cfg.search.simulations = 1
    report = HeuristicMatchRunner(ModelEvaluator("A"), cfg, verbose=False).run()
    assert report["status"] == "completed_with_failures"
    assert report["submitted_games"] == 2 and report["failure_reasons"] == {"dead_end": 2}
    assert report["draws"] == report["completed_pairs"] == 0
    for row in report["game_results"]:
        assert row["seed"] == 42 and row["action_ids"] == actions
        diagnostic = json.loads(Path(row["diagnostic_path"]).read_text())
        assert diagnostic["trajectory_verification"]["shared_nonterminal_dead_end"]


def test_cli_real_checkpoint_readonly(tmp_path):
    checkpoint = Path(__file__).parents[1] / "training_v6/data/model4_v6_inference_latest.pt"
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    report_path = tmp_path / "match.json"
    report = main(["--model", str(checkpoint), "--opponent", "random", "--games", "2",
        "--workers", "2", "--batch-size", "2", "--device", "cpu", "--base-seed", "10000",
        "--simulations", "16", "--min-simulations", "4", "--check-interval", "4",
        "--target-visits-per-action", "0.2", "--report", str(report_path),
        "--failure-dir", str(tmp_path / "failures")])
    assert report["completed_games"] == 2 and report["completed_pairs"] == 1
    assert report["failed_games"] == 0
    assert report["models"]["A"]["checkpoint_sha256"] == digest
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == digest
    assert set(report["gpu_servers"]) == {"A"}
    assert json.loads(report_path.read_text()) == report


def test_cli_partial_report(tmp_path):
    checkpoint = Path(__file__).parents[1] / "training_v6/data/model4_v6_inference_latest.pt"
    report_path = tmp_path / "partial.json"
    with pytest.raises(MatchError, match="max_game_steps"):
        main(["--model", str(checkpoint), "--opponent", "h3", "--games", "2", "--workers", "1",
            "--device", "cpu", "--simulations", "1", "--fixed-simulations", "--max-game-steps", "1",
            "--failure-dir", str(tmp_path / "failures"), "--report", str(report_path)])
    report = json.loads(report_path.read_text())
    assert report["status"] == "failed" and report["failed_games"] == 1
    assert report["model_wins"] == report["heuristic_wins"] == report["draws"] == 0


def test_cli_rejects_checkpoint_as_report(tmp_path):
    checkpoint = tmp_path / "model.pt"; checkpoint.write_bytes(b"unchanged")
    with pytest.raises(SystemExit):
        main(["--model", str(checkpoint), "--report", str(checkpoint)])
    assert checkpoint.read_bytes() == b"unchanged"
