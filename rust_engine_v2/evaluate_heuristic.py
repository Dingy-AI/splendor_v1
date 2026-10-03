"""Paired Rust Model 4 matches against random, greedy, H3 or H12 agents.

Only the model runs MCTS and neural inference. The opponent chooses a legal
move directly. Real transitions and model searches execute in Rust; existing
heuristics retain their Python scoring/rollouts. No training/replay writes.
"""
import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
import random
import time

from splendor_v1.env.env import SplendorEnv
from splendor_v1.rust_engine_v2 import to_python
from splendor_v1.rust_engine_v2.cli import inference_options
from splendor_v1.rust_engine_v2.evaluate_models import (
    MatchConfig, MatchError, MatchGame, MatchRunner, add_match_arguments,
    match_config_from_args, write_report,
)
from splendor_v1.rust_engine_v2.inference import PackedModel4Evaluator, load_model


OPPONENTS = ("random", "greedy", "h3", "h12")


@dataclass
class HeuristicMatchConfig(MatchConfig):
    opponent: str = "random"
    heuristic_rollouts: int = 8
    heuristic_max_rollout_steps: int = 200
    failure_dir: str = "splendor_v1/rust_engine_v2/heuristic_evaluation_failures"

    def validate(self):
        super().validate()
        if self.opponent not in OPPONENTS:
            raise ValueError(f"opponent must be one of {OPPONENTS}")
        if min(self.heuristic_rollouts, self.heuristic_max_rollout_steps) < 1:
            raise ValueError("Heuristic rollout count and step limit must be positive")

    def opponent_metadata(self):
        metadata = dict(name=self.opponent, kind="direct_agent", mcts=False,
                        rng="per_game_local", rng_seed="paired_board_seed")
        if self.opponent == "h12":
            metadata.update(rollouts=self.heuristic_rollouts,
                            max_rollout_steps=self.heuristic_max_rollout_steps)
        return metadata


class OpponentPolicy:
    """One independently seeded policy per game, with no global RNG changes."""
    def __init__(self, config, seed):
        self.kind = config.opponent
        self.rng = random.Random(int(seed))
        self.env = None
        self.agent = None
        if self.kind == "random":
            return
        self.env = SplendorEnv(num_players=2)
        if self.kind == "greedy":
            from splendor_v1.agents.greedy_agent import GreedyAgent
            self.agent = GreedyAgent()
        elif self.kind == "h3":
            from splendor_v1.agents.heuristic_agent_3 import HeuristicAgent3
            self.agent = HeuristicAgent3()
        elif self.kind == "h12":
            from splendor_v1.agents.heuristic_agent_12 import HeuristicAgent12
            self.agent = HeuristicAgent12(num_rollouts=config.heuristic_rollouts,
                max_rollout_steps=config.heuristic_max_rollout_steps, random_seed=int(seed))
        else:
            raise ValueError(f"Unknown opponent: {self.kind}")

    def select_action(self, native):
        ids = native.legal_action_ids()
        if not ids:
            raise MatchError("Opponent was asked to move with no legal actions")
        if self.kind == "random":
            return self.rng.choice(ids)
        self.env.state = state = to_python(native)
        python_ids = [self.env.action_to_id(action) for action in self.env._legal_actions(state)]
        if sorted(python_ids) != sorted(ids):
            raise MatchError("Python heuristic and Rust legal actions disagree")
        action = self.agent.select_action(self.env, state)
        if action is None:
            raise MatchError(f"{self.kind} returned no action for a legal position")
        action_id = self.env.action_to_id(action)
        if action_id not in ids:
            raise MatchError(f"{self.kind} selected an illegal action")
        return action_id


class HeuristicMatchGame(MatchGame):
    def __init__(self, arena, slot, game_id, config, models, *, autostart=True):
        self.opponent = OpponentPolicy(config, config.base_seed + game_id // 2)
        self.heuristic_decisions = 0
        self.heuristic_action_seconds = 0.0
        self.completed_result = None
        super().__init__(arena, slot, game_id, config, models, autostart=autostart)

    def start_decision(self):
        # Consume direct opponent decisions at the real-move boundary. No B
        # search is begun, so every gathered neural request belongs to A.
        while self.owner == "B":
            self.check_decision()
            started = time.perf_counter()
            action = self.opponent.select_action(self.search.root_state())
            self.heuristic_action_seconds += time.perf_counter() - started
            if action not in self.search.root_action_ids():
                raise MatchError("Opponent selected an illegal native action")
            self.heuristic_decisions += 1
            _, terminated = self.search.advance(action)
            self.actions.append(int(action))
            state = self.search.root_state()
            if terminated:
                winners = list(state.winners)
                if not state.game_over or not winners or any(w not in (0, 1) for w in winners):
                    raise MatchError("Terminal game has invalid winner information")
                outcome = "draw" if len(winners) == 2 else self.model_for_player(winners[0])
                self.completed_result = self.record(status="completed", outcome=outcome, winner_ids=winners)
                return self.completed_result
            if self.model_for_player(state.current_player) == "A":
                self.arena.remove(self.slot)
                self.owner = "A"
                self.search = self.make_search(state)
        super().start_decision()

    def finish_decision(self):
        result = super().finish_decision()
        return result if result is not None else self.completed_result

    def diagnostic(self):
        data = super().diagnostic()
        data.update(match_type="model_vs_heuristic", opponent=self.config.opponent_metadata(),
                    heuristic_seed=self.seed)
        return data

    def record(self, **kwargs):
        data = super().record(**kwargs)
        data.update(heuristic_decisions=self.heuristic_decisions,
                    heuristic_action_seconds=self.heuristic_action_seconds, heuristic_seed=self.seed)
        return data


class HeuristicMatchRunner(MatchRunner):
    evaluator_labels = {"A"}

    def __init__(self, evaluator, config=None, model=None, verbose=True):
        config = config or HeuristicMatchConfig()
        models = {"A": model or dict(name="model", kind="neural_mcts"),
                  "B": config.opponent_metadata()}
        super().__init__({"A": evaluator}, config, models, verbose,
                         game_factory=HeuristicMatchGame)

    def report(self):
        data = super().report()
        data.update(match_type="model_vs_heuristic", model_wins=data["model_a_wins"],
            heuristic_wins=data["model_b_wins"], opponent=self.config.opponent_metadata(),
            model_match_score=data["complete_pair_results"]["model_a_match_score"],
            heuristic_decisions=sum(row["heuristic_decisions"] for row in self.records))
        # This time is already included in game_boundary_seconds, not additive.
        data["pipeline_profile"]["heuristic_action_seconds"] = sum(
            row["heuristic_action_seconds"] for row in self.records)
        return data


def main(argv=None):
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--name", default="model")
    parser.add_argument("--opponent", choices=OPPONENTS, default="random")
    parser.add_argument("--heuristic-rollouts", type=int, default=8, help="H12 only: sampled worlds per candidate")
    parser.add_argument("--heuristic-max-rollout-steps", type=int, default=200, help="H12 only: rollout decision cap")
    add_match_arguments(parser, report="splendor_v1/rust_engine_v2/heuristic_match_report.json",
                        failure_dir=HeuristicMatchConfig().failure_dir)
    args = parser.parse_args(argv)
    config = match_config_from_args(args, HeuristicMatchConfig, opponent=args.opponent,
        heuristic_rollouts=args.heuristic_rollouts,
        heuristic_max_rollout_steps=args.heuristic_max_rollout_steps)
    try:
        config.validate()
        if args.cpu_threads < 1: raise ValueError("cpu-threads must be positive")
        if args.report.resolve() == args.model.resolve():
            raise ValueError("The report path must differ from the checkpoint path")
    except ValueError as error:
        parser.error(str(error))
    torch.set_num_threads(args.cpu_threads)
    options = inference_options(args)
    wall_started = time.perf_counter()
    print(f"Loading model: {args.model} on {args.device} ({options.precision})", flush=True)
    model = load_model(args.model, args.device)
    metadata = dict(name=args.name, kind="neural_mcts", checkpoint=str(args.model),
                    checkpoint_sha256=model.checkpoint_sha256)
    print(f"Opponent: {args.opponent} (direct moves, no MCTS)", flush=True)
    runner = HeuristicMatchRunner(PackedModel4Evaluator(model, options), config, metadata)
    runner.checkpoint_load_seconds = time.perf_counter() - wall_started
    try:
        report = runner.run(wall_started=wall_started)
    except (Exception, KeyboardInterrupt):
        report = runner.report()
        report.update(device=args.device, inference_options=asdict(options))
        write_report(args.report, report)
        print(f"Partial heuristic evaluation report saved: {args.report}", flush=True)
        raise
    report.update(device=args.device, inference_options=asdict(options))
    write_report(args.report, report)
    print(f"Model wins: {report['model_wins']} | {args.opponent} wins: {report['heuristic_wins']} | "
          f"draws: {report['draws']} | failed: {report['failed_games']}", flush=True)
    score = report["model_match_score"]
    if score is not None:
        print(f"Complete pairs: {report['completed_pairs']}; model paired match score: {score:.2%}", flush=True)
    else:
        print("No complete pairs to score", flush=True)
    print(f"Throughput: {report['games_per_hour']:.1f} games/hour; report: {args.report}", flush=True)
    return report


if __name__ == "__main__":
    main()
