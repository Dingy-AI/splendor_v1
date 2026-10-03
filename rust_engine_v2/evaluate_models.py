"""Paired Model 4 checkpoint matches using Rust searches and shared GPU batching.

One agent's network evaluates every leaf of its search. Trees are reused only
through the same player's forced sub-decisions, matching the Python V6 evaluator.
No training or replay writes occur. Checkpoints are loaded once and hashed.
"""
import argparse
from collections import Counter
from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
import time
from uuid import uuid4

import numpy as np

from splendor_v1.mcts_batched.multiprocess_self_play_worker import SelfPlaySearchConfig
from splendor_v1.rust_engine.failures import EmptyActionError, FailurePolicy, empty_action_error
from splendor_v1.rust_engine_v2 import reset, to_python
from splendor_v1.rust_engine_v2.arena import ArenaSearch, decode_batch, response_bytes
from splendor_v1.rust_engine_v2.cli import add_inference_arguments, inference_options
from splendor_v1.rust_engine_v2.inference import PackedModel4Evaluator, load_model


@dataclass
class MatchConfig:
    games: int = 1000  # Total games, two games per seed.
    workers: int = 256  # Concurrent game slots, not CPU threads.
    batch_size: int = 256  # Combined gathered rows; each model gets its own subset.
    base_seed: int = 500_000
    search: SelfPlaySearchConfig = field(default_factory=SelfPlaySearchConfig)
    adaptive_simulations: bool = True
    keep_going: bool = False
    include_moves: bool = False
    heartbeat_seconds: float = 10.0
    iteration_timeout_seconds: float = 1800.0
    failure_dir: str = "splendor_v1/rust_engine_v2/evaluation_failures"

    def validate(self):
        if self.games < 2 or self.games % 2:
            raise ValueError("games must be a positive even number (two games per paired seed)")
        if min(self.workers, self.batch_size) < 1 or self.base_seed < 0:
            raise ValueError("workers/batch-size must be positive and base-seed nonnegative")
        for value in (self.heartbeat_seconds, self.iteration_timeout_seconds):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Heartbeat and iteration timeout must be finite and positive")
        self.search.validate()
        for value in (self.search.c_puct, self.search.target_visits_per_action):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Search coefficients must be finite and positive")


class MatchError(RuntimeError):
    pass


class StepCapError(RuntimeError):
    def __init__(self, diagnostic):
        super().__init__("Evaluation exceeded max_game_steps")
        self.diagnostic = diagnostic


class MatchGame:
    def __init__(self, arena, slot, game_id, config, models, *, autostart=True):
        self.arena, self.slot, self.game_id = arena, slot, game_id
        self.config, self.models = config, models
        self.seed = config.base_seed + game_id // 2
        self.a_player = game_id % 2
        self.started = time.perf_counter()
        self.actions = []
        self.searches = Counter()
        self.simulations = Counter()
        self.stop_reasons = {"A": Counter(), "B": Counter()}
        state = reset(self.seed)
        self.owner = self.model_for_player(state.current_player)
        self.search = self.make_search(state)
        if autostart:
            self.start_decision()

    def model_for_player(self, player):
        return "A" if player == self.a_player else "B"

    def make_search(self, state):
        return ArenaSearch(self.arena, self.slot, state, self.config.search, self.seed,
                           self.config.adaptive_simulations)

    def diagnostic(self):
        return dict(schema_version=1, backend="rust_v2", job_id=self.game_id,
            seed=self.seed, step_index=len(self.actions), action_ids=list(self.actions),
            state=json.loads(self.search.root_state().snapshot_json()),
            model_a_player=self.a_player, search=asdict(self.config.search),
            adaptive_simulations=self.config.adaptive_simulations, models=self.models,
            temperature=0.0, root_noise=False)

    def check_decision(self):
        if len(self.actions) >= self.config.search.max_game_steps:
            raise StepCapError(self.diagnostic())
        if not self.search.root_action_ids():
            from splendor_v1.env.env import SplendorEnv
            env = SplendorEnv(num_players=2)
            env.state = to_python(self.search.root_state())
            error = empty_action_error(env, self.diagnostic()["state"], len(self.actions))
            error.diagnostic.update(self.diagnostic())
            raise error

    def start_decision(self):
        self.check_decision()
        self.search.begin(add_root_noise=False)
        if not self.search.active:
            raise MatchError("Native search did not start on a nonterminal legal position")

    def finish_decision(self):
        state = self.search.root_state()
        previous_player = state.current_player
        if self.owner != self.model_for_player(previous_player):
            raise MatchError("Search was routed to the wrong model")
        metadata = self.search.summary["metadata"]
        self.searches[self.owner] += 1
        self.simulations[self.owner] += int(metadata["actual_simulations"])
        self.stop_reasons[self.owner][metadata["stop_reason"]] += 1
        action = self.search.select_action(temperature=0.0)
        if action is None or action not in self.search.root_action_ids():
            raise MatchError("Native search selected an illegal action")
        _, terminated = self.search.advance(action)
        self.actions.append(int(action))
        state = self.search.root_state()
        if terminated:
            winners = list(state.winners)
            if not state.game_over or not winners or any(w not in (0, 1) for w in winners):
                raise MatchError("Terminal game has invalid winner information")
            outcome = "draw" if len(winners) == 2 else self.model_for_player(winners[0])
            return self.record(status="completed", outcome=outcome, winner_ids=winners)
        if state.current_player != previous_player:
            # Discard this agent's predictions before its opponent takes control.
            self.arena.remove(self.slot)
            self.owner = self.model_for_player(state.current_player)
            self.search = self.make_search(state)
        self.start_decision()
        return None

    def record(self, *, status, outcome=None, winner_ids=None, error=None, diagnostic_path=None):
        state = json.loads(self.search.root_state().snapshot_json())
        scores = [p["points"] for p in state["players"]]
        row = dict(game_id=self.game_id, pair_id=self.game_id // 2, seed=self.seed,
            model_a_player=self.a_player, status=status, outcome=outcome,
            winner_ids=winner_ids, final_scores=scores, model_a_score=scores[self.a_player],
            model_b_score=scores[1 - self.a_player], positions=len(self.actions),
            final_turn_number=state["turn_number"], seconds=time.perf_counter() - self.started,
            searches=dict(self.searches), actual_simulations=dict(self.simulations),
            stop_reasons={label: dict(reasons) for label, reasons in self.stop_reasons.items()})
        if error is not None: row["error"] = str(error)
        if diagnostic_path is not None: row["diagnostic_path"] = diagnostic_path
        if self.config.include_moves: row["action_ids"] = list(self.actions)
        return row


def outcome_counts(rows):
    wins = sum(r["outcome"] == "A" for r in rows)
    losses = sum(r["outcome"] == "B" for r in rows)
    draws = sum(r["outcome"] == "draw" for r in rows)
    n = len(rows)
    return dict(games=n, model_a_wins=wins, model_b_wins=losses, draws=draws,
        model_a_win_rate=wins / n if n else None,
        model_b_win_rate=losses / n if n else None,
        model_a_match_score=(wins + 0.5 * draws) / n if n else None)


def build_summary(records):
    successful = [r for r in records if r["status"] == "completed"]
    pairs = {}
    for row in successful: pairs.setdefault(row["pair_id"], []).append(row)
    complete_pairs = [rows for rows in pairs.values()
                      if len(rows) == 2 and {r["model_a_player"] for r in rows} == {0, 1}]
    paired = [row for pair in complete_pairs for row in pair]
    summary = outcome_counts(successful)
    summary.update(completed_games=len(successful), failed_games=len(records) - len(successful),
        failure_reasons=dict(Counter(r["status"] for r in records if r["status"] != "completed")),
        completed_pairs=len(complete_pairs), complete_pair_results=outcome_counts(paired),
        model_a_as_player0=outcome_counts([r for r in successful if r["model_a_player"] == 0]),
        model_a_as_player1=outcome_counts([r for r in successful if r["model_a_player"] == 1]),
        average_positions=sum(r["positions"] for r in successful) / len(successful) if successful else None)
    for label in ("A", "B"):
        searches = sum(r["searches"].get(label, 0) for r in successful)
        sims = sum(r["actual_simulations"].get(label, 0) for r in successful)
        reasons = Counter()
        for r in successful: reasons.update(r["stop_reasons"][label])
        summary[f"model_{label.lower()}_searches"] = searches
        summary[f"model_{label.lower()}_average_simulations_per_search"] = sims / searches if searches else None
        summary[f"model_{label.lower()}_stop_reasons"] = dict(reasons)
    return summary


class MatchRunner:
    evaluator_labels = {"A", "B"}

    def __init__(self, evaluators, config=None, models=None, verbose=True, game_factory=MatchGame):
        self.config = config or MatchConfig()
        self.config.validate()
        if set(evaluators) != self.evaluator_labels:
            raise ValueError(f"Required evaluators: {sorted(self.evaluator_labels)}")
        self.evaluators = evaluators
        self.game_factory = game_factory
        self.models = models or {label: {"name": label} for label in ("A", "B")}
        self.verbose = verbose
        self.records = []
        self.submitted = self.batches = self.neural_positions = 0
        self.scheduler_seconds = self.boundary_seconds = self.evaluator_seconds = 0.0
        self.checkpoint_load_seconds = 0.0
        self.started = None
        self.arena = None
        self.status = "not_started"
        self.error = None

    def log(self, message):
        if self.verbose: print(message, flush=True)

    def handle_game_failure(self, session, error):
        if isinstance(error, EmptyActionError):
            policy = FailurePolicy(0, self.config.failure_dir)
            _, message = policy.handle(error)  # Save and verify; never replace evaluation seeds.
            path = policy.records[-1]["diagnostic_path"]
            eligible = error.diagnostic["shared_dead_end"]
            reason = "dead_end" if eligible else "rules_mismatch"
        elif isinstance(error, StepCapError):
            directory = Path(self.config.failure_dir)
            directory.mkdir(parents=True, exist_ok=True)
            path = str(directory / f"job_{session.game_id}_seed_{session.seed}_{uuid4().hex}.json")
            Path(path).write_text(json.dumps(error.diagnostic, indent=2, allow_nan=False) + "\n",
                                  encoding="utf-8")
            message, eligible, reason = f"Diagnostic: {path}", True, "step_cap"
        else:
            path, message, eligible, reason = None, "", False, "error"
        self.records.append(session.record(status=reason, error=error, diagnostic_path=path))
        self.log(f"Evaluation failed: game={session.game_id}, seed={session.seed}, "
                 f"A=P{session.a_player}: {error}. {message}")
        if not self.config.keep_going or not eligible:
            raise MatchError(f"Game {session.game_id}, seed {session.seed}: {error}. {message}") from error

    def evaluate_batch(self, slots, active, observations, actions, mask):
        # Route by the ROOT search owner, never by the simulated leaf player.
        if any(active[slot].owner not in self.evaluators for slot in slots):
            raise MatchError("A pending search has no matching neural evaluator")
        chunks = []
        for label in self.evaluators:
            indices = np.asarray([i for i, slot in enumerate(slots) if active[slot].owner == label], dtype=np.int64)
            if not len(indices): continue
            width = int(mask[indices].sum(axis=1).max())
            output = np.asarray(self.evaluators[label].evaluate_packed(
                observations[indices], actions[indices, :width], mask[indices, :width]))
            if output.shape != (len(indices), width + 1) or not np.isfinite(output).all():
                raise MatchError(f"Model {label} returned malformed or nonfinite inference results")
            chunks.append((indices, width, output))
        dtype = np.float64 if all(chunk[2].dtype == np.float64 for chunk in chunks) else np.float32
        merged = np.zeros((len(slots), actions.shape[1] + 1), dtype=dtype)
        for indices, width, output in chunks:
            merged[indices, :width] = output[:, :width]
            merged[indices, -1] = output[:, -1]
        return merged

    def run(self, *, wall_started=None, progress_callback=None):
        from splendor_rust_v2 import RustArena
        if self.started is not None: raise MatchError("A match runner can only run once")
        self.started = wall_started if wall_started is not None else time.perf_counter()
        self.status = "running"
        for evaluator in self.evaluators.values():
            if hasattr(evaluator, "reset_profile"): evaluator.reset_profile()
        self.arena = RustArena(min(self.config.workers, self.config.games))
        active = {}
        last_heartbeat = time.perf_counter()
        def submit(slot):
            while self.submitted < self.config.games:
                game_id = self.submitted
                self.submitted += 1
                session = self.game_factory(self.arena, slot, game_id, self.config, self.models,
                                            autostart=False)
                active[slot] = session
                try:
                    result = session.start_decision()
                except Exception as error:
                    self.handle_game_failure(session, error)
                    result = self.records[-1]
                else:
                    if result is not None: self.records.append(result)
                if result is None:
                    return
                self.arena.remove(slot)
                del active[slot]
                if progress_callback is not None: progress_callback(result)
        try:
            self.log(f"Rust evaluation: {self.config.games} games / {self.config.games // 2} paired seeds, "
                     f"{min(self.config.workers, self.config.games)} slots; no noise, temperature=0")
            started = time.perf_counter()
            for slot in range(min(self.config.workers, self.config.games)): submit(slot)
            self.boundary_seconds += time.perf_counter() - started
            while active:
                iteration_started = time.perf_counter()
                started = time.perf_counter()
                token, slots, ready, obs, actions, mask = decode_batch(self.arena.gather(self.config.batch_size))
                self.scheduler_seconds += time.perf_counter() - started
                started = time.perf_counter()
                for slot in ready:
                    session = active[slot]
                    try:
                        result = session.finish_decision()
                    except Exception as error:
                        self.handle_game_failure(session, error)
                        result = self.records[-1]  # Already recorded, without a winner/draw.
                    else:
                        if result is not None: self.records.append(result)
                    if result is None: continue
                    self.arena.remove(slot)
                    del active[slot]
                    if self.submitted < self.config.games: submit(slot)
                    if progress_callback is not None: progress_callback(result)
                self.boundary_seconds += time.perf_counter() - started
                if slots:
                    started = time.perf_counter()
                    output = self.evaluate_batch(slots, active, obs, actions, mask)
                    self.evaluator_seconds += time.perf_counter() - started
                    payload, double = response_bytes(output, len(slots), actions.shape[1])
                    started = time.perf_counter()
                    self.arena.respond(token, payload, double)
                    self.scheduler_seconds += time.perf_counter() - started
                    self.batches += 1; self.neural_positions += len(slots)
                now = time.perf_counter()
                if active and now - iteration_started > self.config.iteration_timeout_seconds:
                    raise MatchError("An evaluation iteration exceeded the progress timeout")
                if active and not slots and not ready:
                    raise MatchError("Rust arena made no progress with active games")
                if now - last_heartbeat >= self.config.heartbeat_seconds:
                    counts = build_summary(self.records)
                    self.log(f"Evaluation: {len(self.records)}/{self.config.games} resolved, "
                        f"A wins={counts['model_a_wins']}, B wins={counts['model_b_wins']}, "
                        f"draws={counts['draws']}, failures={counts['failed_games']}; "
                        f"{counts['completed_games'] / (now - self.started) * 3600:.0f} games/hour")
                    last_heartbeat = now
            self.status = "completed_with_failures" if any(r["status"] != "completed" for r in self.records) else "completed"
        except (Exception, KeyboardInterrupt) as error:
            self.status = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
            self.error = f"{type(error).__name__}: {error}"
            raise
        finally:
            self.arena = None  # Also releases unfinished trees on failure/interruption.
            self.finished = time.perf_counter()
        return self.report()

    def report(self):
        wall = getattr(self, "finished", time.perf_counter()) - self.started if self.started is not None else 0.0
        summary = build_summary(self.records)
        summary.update(status=self.status, error=self.error, requested_games=self.config.games,
            submitted_games=self.submitted, resolved_games=len(self.records),
            unresolved_games=self.config.games - len(self.records), wall_seconds=wall,
            games_per_hour=summary["completed_games"] / wall * 3600 if wall else None,
            execution_backend="rust_v2", native_game_slots=min(self.config.workers, self.config.games),
            evaluation_settings=asdict(self.config),
            native_worker_threads=0, paired_seats=True, temperature=0.0, root_noise=False,
            tree_reuse="same_real_player_only", search_settings=asdict(self.config.search),
            adaptive_simulations=self.config.adaptive_simulations, models=self.models,
            gpu_servers={label: evaluator.profile() if hasattr(evaluator, "profile") else {}
                         for label, evaluator in self.evaluators.items()},
            total_neural_requests=self.neural_positions, total_gathered_batches=self.batches,
            pipeline_profile=dict(checkpoint_load_seconds=self.checkpoint_load_seconds,
                native_scheduler_seconds=self.scheduler_seconds, game_boundary_seconds=self.boundary_seconds,
                evaluator_seconds=self.evaluator_seconds), game_results=sorted(self.records, key=lambda r: r["game_id"]))
        return summary


def write_report(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def add_match_arguments(parser, *, report="splendor_v1/rust_engine_v2/evaluation_report.json",
                        failure_dir=MatchConfig().failure_dir):
    """Search, scheduling and reporting options shared by both match CLIs."""
    parser.add_argument("--games", type=int, default=1000, help="Even total game count, two seat-swapped games per seed")
    parser.add_argument("--workers", type=int, default=256, help="Concurrent native game slots")
    parser.add_argument("--batch-size", type=int, default=256, help="Maximum gathered neural batch size")
    parser.add_argument("--base-seed", type=int, default=500_000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cpu-threads", type=int, default=1)
    defaults = SelfPlaySearchConfig()
    for flag, attribute, kind in (("simulations", "simulations", int),
            ("min-simulations", "min_simulations", int), ("check-interval", "check_interval", int),
            ("target-visits-per-action", "target_visits_per_action", float),
            ("single-action-simulations", "single_action_simulations", int),
            ("stability-checks", "stability_checks", int), ("c-puct", "c_puct", float),
            ("max-game-steps", "max_game_steps", int)):
        parser.add_argument("--" + flag, type=kind, default=getattr(defaults, attribute))
    parser.add_argument("--fixed-simulations", action="store_true", help="Disable adaptive early stopping for neural searches")
    parser.add_argument("--keep-going", action="store_true", help="Record shared dead-ends/step caps and continue original jobs; mismatches/inference errors still stop")
    parser.add_argument("--include-moves", action="store_true", help="Include played action IDs in each game result")
    parser.add_argument("--failure-dir", default=failure_dir)
    parser.add_argument("--iteration-timeout-seconds", type=float, default=1800.0)
    parser.add_argument("--report", type=Path, default=Path(report))
    add_inference_arguments(parser)


def match_config_from_args(args, config_class=MatchConfig, **extra):
    return config_class(games=args.games, workers=args.workers, batch_size=args.batch_size,
        base_seed=args.base_seed, search=SelfPlaySearchConfig(**{name: getattr(args, name)
            for name in ("simulations", "min_simulations", "check_interval", "target_visits_per_action",
                         "single_action_simulations", "stability_checks", "c_puct", "max_game_steps")}),
        adaptive_simulations=not args.fixed_simulations, keep_going=args.keep_going,
        include_moves=args.include_moves, heartbeat_seconds=args.heartbeat_seconds, failure_dir=args.failure_dir,
        iteration_timeout_seconds=args.iteration_timeout_seconds, **extra)


def main(argv=None):
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-a", type=Path, required=True)
    parser.add_argument("--model-b", type=Path, required=True)
    parser.add_argument("--name-a", default="A")
    parser.add_argument("--name-b", default="B")
    add_match_arguments(parser)
    args = parser.parse_args(argv)
    config = match_config_from_args(args)
    try:
        config.validate()
        if args.cpu_threads < 1: raise ValueError("cpu-threads must be positive")
        if args.report.resolve() in (args.model_a.resolve(), args.model_b.resolve()):
            raise ValueError("The report path must differ from both checkpoint paths")
    except ValueError as error:
        parser.error(str(error))
    torch.set_num_threads(args.cpu_threads)
    options = inference_options(args)
    wall_started = time.perf_counter()
    evaluators, models = {}, {}
    for label, path, name in (("A", args.model_a, args.name_a), ("B", args.model_b, args.name_b)):
        print(f"Loading model {label}: {path} on {args.device} ({options.precision})", flush=True)
        model = load_model(path, args.device)
        models[label] = dict(name=name, checkpoint=str(path), checkpoint_sha256=model.checkpoint_sha256)
        evaluators[label] = PackedModel4Evaluator(model, options)
    runner = MatchRunner(evaluators, config, models)
    runner.checkpoint_load_seconds = time.perf_counter() - wall_started
    try:
        report = runner.run(wall_started=wall_started)
    except (Exception, KeyboardInterrupt):
        report = runner.report()
        report.update(device=args.device, inference_options=asdict(options))
        write_report(args.report, report)
        print(f"Partial evaluation report saved: {args.report}", flush=True)
        raise
    report.update(device=args.device, inference_options=asdict(options))
    write_report(args.report, report)
    print(f"A wins: {report['model_a_wins']} | B wins: {report['model_b_wins']} | "
          f"draws: {report['draws']} | failed: {report['failed_games']}", flush=True)
    paired = report["complete_pair_results"]["model_a_match_score"]
    print(f"Complete pairs: {report['completed_pairs']}; A paired match score: "
          f"{paired:.2%}" if paired is not None else "No complete pairs to score", flush=True)
    print(f"Throughput: {report['games_per_hour']:.1f} games/hour; report: {args.report}", flush=True)
    return report


if __name__ == "__main__":
    main()
