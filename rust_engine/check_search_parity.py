"""Extended development check: compare complete Python/Rust search trees across games.

Uses the deterministic evaluator and comparison helpers from test_search_parity;
requires the same pytest/PyTorch dependencies as that test suite. Not a speed test.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from splendor_v1.rust_engine.test_search_parity import (
    canonical, compare_search, compare_tree, make_env, make_searches,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=12)
    parser.add_argument("--simulations", type=int, default=200)
    parser.add_argument("--max-decisions", type=int, default=300)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if min(args.games, args.simulations, args.max_decisions) <= 0 or args.start_seed < 0:
        parser.error("Counts must be positive and start-seed nonnegative")
    decisions = simulations = evaluations = finished = capped = dead_ends = flips = retained_visits = 0
    stops, node_types = Counter(), Counter()
    for seed in range(args.start_seed, args.start_seed + args.games):
        env = make_env(seed)
        reference, native, evaluator = make_searches(env, seed=seed + 1000,
            simulations=args.simulations, adaptive_simulations=(seed % 2 == 0))
        root = None
        for decision in range(args.max_decisions):
            if env.state.game_over:
                finished += 1
                break
            if not env._legal_actions(env.state):
                dead_ends += 1
                break
            node_types[env.state.node_type.name] += 1
            action, root = compare_search(env, reference, native, evaluator, root,
                                          noise=(env.state.turn_number < 40))
            metadata = native.last_search_metadata
            simulations += metadata["actual_simulations"]
            evaluations += len(evaluator.requests)
            retained_visits += metadata["initial_root_visits"]
            stops[metadata["stop_reason"]] += 1
            # Vary chosen moves to cover reuse of lightly visited/lazy branches as well.
            if decision % 7 == 0:
                action = root.children[-1].action
            child = next(c for c in root.children if c.action == action)
            previous_player = env.state.current_player
            env.step(action)
            native.advance(env.action_to_id(action))
            if child.state is None:
                child.state = env.state.clone()
            if env.state.current_player != previous_player:
                reference.flip_tree_values(child)
                flips += 1
            child.parent = None
            root = child
            compare_tree(env, root, native)
            assert canonical(env.state) == json.loads(native.root_state().snapshot_json())
            decisions += 1
        else:
            if env.state.game_over: finished += 1
            else: capped += 1
        print(f"Checked seed {seed}: {decisions} decisions, {simulations} simulations total", flush=True)
    report = dict(games_checked=args.games, start_seed=args.start_seed,
        simulations_per_search=args.simulations, finished_games=finished,
        decision_capped_games=capped, no_legal_action_games=dead_ends,
        decisions_checked=decisions, simulations_checked=simulations,
        neural_requests_checked=evaluations, player_perspective_flips=flips,
        inherited_visits_across_searches=retained_visits,
        stop_reasons=dict(stops), node_types=dict(node_types),
        neural_requests_match_exactly=True, complete_trees_match=True,
        search_metadata_matches=True, chosen_actions_match=True,
        tree_reuse_and_compaction_match=True)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
