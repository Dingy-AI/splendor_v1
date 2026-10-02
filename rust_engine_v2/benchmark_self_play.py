"""Completed-game benchmark into a fresh replay, using V6 search defaults."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import torch

from splendor_v1.rust_engine_v2.cli import add_inference_arguments, inference_options
from splendor_v1.rust_engine_v2.coordinator import NativeSelfPlayCoordinator, NativeSelfPlayCoordinatorConfig
from splendor_v1.training_v2.replay_buffer import ReplayBuffer


def main(argv=None):
    from splendor_v1.training_v6 import run_training_v6 as v6
    parser = argparse.ArgumentParser(description=__doc__)
    add_inference_arguments(parser)
    parser.add_argument("--checkpoint", type=Path, default=Path(v6.INFERENCE_SNAPSHOT_PATH))
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--workers", type=int, default=v6.NUM_SELF_PLAY_WORKERS,
                        help="Concurrent native game slots, not Python threads")
    parser.add_argument("--batch-size", type=int, default=v6.GPU_MAX_BATCH_SIZE)
    parser.add_argument("--simulations", type=int, default=v6.SIMULATIONS)
    parser.add_argument("--seed-start", type=int, default=v6.BASE_SEED)
    parser.add_argument("--device", default=str(v6.device))
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--replay-out", type=Path)
    parser.add_argument("--train-step", action="store_true")
    args = parser.parse_args(argv)
    if min(args.games, args.workers, args.batch_size, args.simulations, args.cpu_threads) <= 0 or args.seed_start < 0:
        parser.error("Counts must be positive and seed-start nonnegative")
    torch.set_num_threads(args.cpu_threads)
    print(f"V2 benchmark: {args.games} games, {args.workers} native slots, "
          f"batch cap={args.batch_size}, simulations={args.simulations}, device={args.device}", flush=True)
    worker = v6.make_worker_config(generating_model_id=str(args.checkpoint))
    worker.search.simulations = args.simulations
    worker.split = "train"
    config = NativeSelfPlayCoordinatorConfig(num_workers=args.workers,
        checkpoint_path=str(args.checkpoint), worker_config=worker, device=args.device,
        max_batch_size=args.batch_size, game_result_timeout_s=v6.SELF_PLAY_GAME_RESULT_TIMEOUT_S)
    replay = ReplayBuffer(capacity=max(1000, args.games * worker.search.max_game_steps),
                          metadata={"native_self_play_v2_benchmark": True})
    owner = NativeSelfPlayCoordinator(replay_buffer=replay, config=config,
        options=inference_options(args), heartbeat_s=args.heartbeat_seconds)
    def progress(info):
        print(f"Completed {info['committed_games']}/{info['requested_games']} games | "
              f"{info['games_per_hour']:.2f} games/hour | replay {len(replay)} positions", flush=True)
    summary = owner.run_block(num_games=args.games, seed_start=args.seed_start, progress_callback=progress)
    summary.update(device=args.device, checkpoint=str(args.checkpoint),
        simulations=args.simulations, search_settings=asdict(worker.search), adaptive_simulations=True,
        seed_start=args.seed_start, established_baseline_games_per_hour=200.0,
        measured_v1_32_worker_games_per_hour=381.9802252667691,
        comparison_note="Compare using the same hardware, checkpoint, seeds, and search workload")
    if args.train_step:
        from splendor_v1.network.model_4_legal_scorer import SplendorNetwork
        from splendor_v1.training_v5.train_v5 import train_network
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        model = SplendorNetwork()
        model.load_state_dict(checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint)))
        model.to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=v6.LEARNING_RATE)
        summary["training_step"] = train_network(model=model, replay_buffer=replay,
            optimizer=optimizer, batch_size=min(v6.BATCH_SIZE, len(replay)),
            training_steps=1, split="train", grad_clip=v6.GRAD_CLIP)
    if args.replay_out: replay.save(str(args.replay_out))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__": main()
