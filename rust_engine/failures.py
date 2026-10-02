"""Diagnose empty-action positions; never turn unfinished games into training data."""
import copy
import json
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4


class EmptyActionError(RuntimeError):
    def __init__(self, diagnostic):
        super().__init__("No legal actions available for a non-terminal replay position.")
        self.diagnostic = diagnostic

    def add_context(self, job, config, trajectory, backend):
        self.diagnostic.update(job_id=int(job.game_id), seed=int(job.seed),
            backend=backend, search=asdict(config.search),
            model_checkpoint=config.model_checkpoint_label,
            model_generation=config.model_generation,
            extra_game_metadata=copy.deepcopy(job.extra_game_metadata),
            action_ids=[int(sample["chosen_action_id"]) for sample in trajectory])


def empty_action_error(env, snapshot, step_index):
    """Only a shared MAIN-node dead-end is eligible for bounded rejection."""
    diagnostic = dict(schema_version=1, step_index=int(step_index), state=snapshot,
        native_legal_action_ids=[], shared_dead_end=False)
    try:
        reference = copy.deepcopy(env.state)
        python_ids = sorted(env.action_to_id(a) for a in env._legal_actions(reference))
        terminated = bool(env._check_terminated(reference))
        diagnostic.update(python_legal_action_ids=python_ids, python_terminated=terminated)
        diagnostic["shared_dead_end"] = (not python_ids and not terminated
            and not snapshot["game_over"] and snapshot["node_type"] == "MAIN_DECISION")
    except Exception as exc:
        diagnostic["python_check_error"] = f"{type(exc).__name__}: {exc}"
    return EmptyActionError(diagnostic)


class FailurePolicy:
    def __init__(self, max_rejected_games=0, failure_dir="splendor_v1/training_v6/data/native_failures"):
        if not isinstance(max_rejected_games, int) or max_rejected_games < 0:
            raise ValueError("max_rejected_games must be a nonnegative integer")
        self.limit = max_rejected_games
        self.directory = Path(failure_dir)
        self.records = []

    def handle(self, exc):
        """Persist evidence before allowing recovery. Other errors are never eligible."""
        if not isinstance(exc, EmptyActionError):
            return False, ""
        self.directory.mkdir(parents=True, exist_ok=True)
        data = exc.diagnostic
        if data["shared_dead_end"]:
            # Check the whole recorded path, not only a state reconstructed from Rust.
            # This keeps a native transition bug from masquerading as a shared dead-end.
            from splendor_v1.rust_engine.replay_failure import replay_failure
            try:
                data["trajectory_verification"] = replay_failure(data)
            except Exception as mismatch:
                data["shared_dead_end"] = False
                data["trajectory_verification_error"] = f"{type(mismatch).__name__}: {mismatch}"
        path = self.directory / (f"job_{data['job_id']}_seed_{data['seed']}_{uuid4().hex}.json")
        with path.open("x", encoding="utf-8") as output:
            json.dump(data, output, indent=2, allow_nan=False)
        record = dict(job_id=data["job_id"], seed=data["seed"], diagnostic_path=str(path),
                      reason="shared_dead_end" if data["shared_dead_end"] else "empty_action_mismatch")
        self.records.append(record)
        allowed = data["shared_dead_end"] and len(self.records) <= self.limit
        message = f"Diagnostic: {path}."
        if allowed:
            print(f"Rejected unfinished native game: job={data['job_id']}, seed={data['seed']}. "
                  f"Starting a fresh job. {message}", flush=True)
        else:
            message += " Recovery disabled, limit exceeded, or Python/Rust mismatch."
        return allowed, message


def add_failure_arguments(parser):
    parser.add_argument("--max-rejected-games", type=int, default=10,
        help="Maximum shared rules dead-ends replaced per training block (0: stop immediately)")
    parser.add_argument("--failure-dir", default="splendor_v1/training_v6/data/native_failures",
        help="Directory for reproducible empty-action diagnostics")
