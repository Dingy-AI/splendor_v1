"""Run native self-play into a fresh replay buffer and report completed games/hour.

Defaults for search, seeds, and batching come from the existing V6 configuration.
This never loads or modifies the user's training replay or training checkpoint.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from splendor_v1.rust_engine.coordinator import NativeSelfPlayCoordinator, NativeSelfPlayCoordinatorConfig
from splendor_v1.training_v2.replay_buffer import ReplayBuffer


def main():
    from splendor_v1.training_v6 import run_training_v6 as v6
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path(v6.INFERENCE_SNAPSHOT_PATH))
    parser.add_argument("--games", type=int, default=4)
    parser.add_argument("--workers", type=int, default=v6.NUM_SELF_PLAY_WORKERS)
    parser.add_argument("--simulations", type=int, default=v6.SIMULATIONS)
    parser.add_argument("--seed-start", type=int, default=v6.BASE_SEED)
    parser.add_argument("--device", default=str(v6.device))
    parser.add_argument("--batch-size", type=int, default=v6.GPU_MAX_BATCH_SIZE)
    parser.add_argument("--batch-wait-ms", type=float, default=v6.GPU_BATCH_WAIT_MS)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--replay-out", type=Path)
    parser.add_argument("--train-step", action="store_true",
                        help="Verify one optimizer step on a separate loaded model; no model is saved")
    args = parser.parse_args()
    if min(args.games, args.workers, args.simulations, args.cpu_threads) <= 0 or args.seed_start < 0:
        parser.error("Counts must be positive and seed-start nonnegative")
    torch.set_num_threads(args.cpu_threads)
    config = v6.make_worker_config(generating_model_id=str(args.checkpoint))
    config.search.simulations = args.simulations
    config.split = "train"
    owner_config = NativeSelfPlayCoordinatorConfig(num_workers=args.workers,
        checkpoint_path=str(args.checkpoint), worker_config=config, device=args.device,
        max_batch_size=args.batch_size, batch_wait_ms=args.batch_wait_ms,
        game_result_timeout_s=v6.SELF_PLAY_GAME_RESULT_TIMEOUT_S)
    replay = ReplayBuffer(capacity=max(1000, args.games * config.search.max_game_steps),
                          metadata={"native_self_play_benchmark": True})
    owner = NativeSelfPlayCoordinator(replay_buffer=replay, config=owner_config)
    def progress(info):
        print(f"Completed {info['committed_games']}/{info['requested_games']} games | "
              f"{info['games_per_hour']:.2f} games/hour | replay {len(replay)} positions", flush=True)
    summary = owner.run_block(num_games=args.games, seed_start=args.seed_start,
                             progress_callback=progress)
    summary.update(device=args.device, checkpoint=str(args.checkpoint),
        checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        simulations=args.simulations, seed_start=args.seed_start,
        established_baseline_games_per_hour=200.0,
        comparison_note="Compare against 200 only on the baseline hardware/checkpoint/search workload")
    if args.train_step:
        from splendor_v1.network.model_4_legal_scorer import SplendorNetwork
        from splendor_v1.training_v5.train_v5 import train_network
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        model = SplendorNetwork()
        model.load_state_dict(checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint)))
        model.to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=v6.LEARNING_RATE)
        summary["training_step"] = train_network(model=model, replay_buffer=replay, optimizer=optimizer,
            batch_size=min(v6.BATCH_SIZE, len(replay)), training_steps=1, split="train", grad_clip=v6.GRAD_CLIP)
    if args.replay_out:
        replay.save(str(args.replay_out))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
