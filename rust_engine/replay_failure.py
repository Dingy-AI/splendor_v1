"""Replay a saved failure's actual moves in Python and Rust, without a model/GPU."""
import argparse
import json
from pathlib import Path

from splendor_v1.env.env import SplendorEnv
from splendor_v1.rust_engine import from_python
from splendor_v1.training_v2.state_serializer import serialize_state


def replay_failure(data):
    env = SplendorEnv(num_players=2)
    env.reset(seed=int(data["seed"]))
    if data["backend"] == "rust_v2":
        from splendor_v1.rust_engine_v2 import from_python as convert
    else:
        convert = from_python
    native = convert(env.state)
    def check(step):
        if json.loads(native.snapshot_json()) != json.loads(json.dumps(serialize_state(env.state))):
            raise RuntimeError(f"Python/Rust state mismatch at decision {step}")
        actions = {env.action_to_id(a): a for a in env._legal_actions(env.state)}
        if sorted(actions) != sorted(native.legal_action_ids()):
            raise RuntimeError(f"Python/Rust legal-action mismatch at decision {step}")
        return actions
    for step, action_id in enumerate(data["action_ids"]):
        actions = check(step)
        if action_id not in actions:
            raise RuntimeError(f"Recorded action {action_id} is illegal at decision {step}")
        _, reward, terminated, _, _ = env.step(actions[action_id])
        if native.step(action_id) != (reward, terminated):
            raise RuntimeError(f"Python/Rust transition mismatch at decision {step}")
        if terminated:
            raise RuntimeError(f"Recorded failure trajectory terminated at decision {step}")
    actions = check(len(data["action_ids"]))
    if json.loads(native.snapshot_json()) != data["state"]:
        # Diagnostics produced in memory may have integer tier keys.
        if json.loads(native.snapshot_json()) != json.loads(json.dumps(data["state"])):
            raise RuntimeError("Replayed final state differs from the saved failure")
    if actions or env.state.game_over:
        raise RuntimeError("Replayed position is not a shared nonterminal empty-action state")
    return dict(verified_decisions=len(data["action_ids"]), node_type=env.state.node_type.name,
                shared_nonterminal_dead_end=True, seed=data["seed"], job_id=data["job_id"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("diagnostic", type=Path)
    args = parser.parse_args()
    print(json.dumps(replay_failure(json.loads(args.diagnostic.read_text(encoding="utf-8"))), indent=2))


if __name__ == "__main__":
    main()
