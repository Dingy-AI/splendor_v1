"""Seeded differential games; this measures correctness, not self-play throughput."""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np

from splendor_v1.env.env import SplendorEnv
from splendor_v1.rust_engine import from_python, observation
from splendor_v1.training_v2.state_serializer import serialize_state


def check_state(env, native):
    data = json.loads(native.snapshot_json())
    for field in ("visible_card_ids", "deck_card_ids"):
        data[field] = {int(t): row for t, row in data[field].items()}
    assert data == serialize_state(env.state), "Full state mismatch"
    actions = env._legal_actions(env.state)
    ids = [env.action_to_id(a) for a in actions]
    assert native.legal_action_ids() == ids, "Ordered legal-action IDs mismatch"
    np.testing.assert_array_equal(observation(native), env.observation_encoder.encoder(env.state))
    return actions, ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=10000)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--max-decisions", type=int, default=2000)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.games <= 0 or args.max_decisions <= 0 or args.start_seed < 0:
        parser.error("games and max-decisions must be positive; start-seed must be nonnegative")
    kinds = Counter()
    nodes = Counter()
    decisions = branches = finished = no_actions = capped = max_actions = 0
    for game in range(args.games):
        seed = args.start_seed + game
        env = SplendorEnv()
        env.reset(seed=seed)
        native = from_python(env.state)
        rng = random.Random(seed)
        for step in range(args.max_decisions):
            try:
                actions, ids = check_state(env, native)
                nodes[env.state.node_type.name] += 1
                max_actions = max(max_actions, len(actions))
                if env.state.game_over:
                    finished += 1
                    break
                if not actions:
                    no_actions += 1
                    break
                if step % 64 == 0:
                    # Branches cover alternate successors and ensure cloning is independent.
                    before = native.snapshot_json()
                    for index in rng.sample(range(len(actions)), min(3, len(actions))):
                        branch = native.clone()
                        child = env.state.clone()
                        _, reward, terminated, _, _ = env.step(actions[index], state=child)
                        assert branch.step(ids[index]) == (reward, terminated)
                        reference_state = env.state
                        env.state = child
                        try:
                            check_state(env, branch)
                        finally:
                            env.state = reference_state
                        branches += 1
                    assert native.snapshot_json() == before, "Clone modified its parent"
                index = rng.randrange(len(actions))
                kinds[actions[index].action_type.name] += 1
                _, reward, terminated, truncated, _ = env.step(actions[index])
                assert not truncated
                assert native.step(ids[index]) == (reward, terminated), "Reward/termination mismatch"
                decisions += 1
            except Exception as exc:
                raise RuntimeError(f"Parity failed at game seed={seed}, decision={step}") from exc
        else:
            capped += 1
            check_state(env, native)
        if (game + 1) % 500 == 0:
            print(f"Checked {game + 1}/{args.games} games, {decisions} decisions", flush=True)
    report = {
        "games_checked": args.games, "start_seed": args.start_seed,
        "finished_games": finished, "decision_capped_games": capped,
        "no_legal_action_games": no_actions, "decisions_checked": decisions,
        "branch_successors_checked": branches, "max_legal_actions": max_actions,
        "action_types": dict(sorted(kinds.items())), "node_types": dict(sorted(nodes.items())),
        "full_state_match": True, "ordered_action_ids_match": True,
        "float32_observations_match_exactly": True, "reward_and_termination_match": True,
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
