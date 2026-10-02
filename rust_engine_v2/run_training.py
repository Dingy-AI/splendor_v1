"""Existing V6 training configuration with V2 arena self-play, opt-in inference."""
import argparse
from dataclasses import replace

from splendor_v1.rust_engine_v2.cli import add_inference_arguments, inference_options
from splendor_v1.rust_engine_v2.coordinator import NativeSelfPlayCoordinator
from splendor_v1.rust_engine.failures import add_failure_arguments


def main(argv=None):
    from splendor_v1.training_v6 import run_training_v6 as training
    parser = argparse.ArgumentParser(description=__doc__)
    add_inference_arguments(parser)
    add_failure_arguments(parser)
    parser.add_argument("--games-in-flight", type=int,
                        help="Override V6 worker count with this many native game slots")
    parser.add_argument("--batch-size", type=int, help="Override V6 inference batch cap")
    args = parser.parse_args(argv)
    if args.max_rejected_games < 0: parser.error("max-rejected-games must be nonnegative")
    if ((args.games_in_flight is not None and args.games_in_flight < 1)
            or (args.batch_size is not None and args.batch_size < 1)):
        parser.error("Game slots and batch-size must be positive")
    options = inference_options(args)
    def factory(**kwargs):
        config = kwargs.pop("config")
        config = replace(config,
            num_workers=args.games_in_flight if args.games_in_flight is not None else config.num_workers,
            max_batch_size=args.batch_size if args.batch_size is not None else config.max_batch_size)
        return NativeSelfPlayCoordinator(config=config, options=options,
            max_rejected_games=args.max_rejected_games, failure_dir=args.failure_dir,
            heartbeat_s=args.heartbeat_seconds, **kwargs)
    original = training.MultiprocessSelfPlayCoordinator
    print(f"Self-play backend: rust_engine_v2 native arena; precision={args.precision}; "
          f"compile={args.compile_mode}", flush=True)
    print(f"Shared dead-end rejection limit: {args.max_rejected_games} per block; "
          f"diagnostics: {args.failure_dir}", flush=True)
    training.MultiprocessSelfPlayCoordinator = factory
    try:
        training.main()
    finally:
        training.MultiprocessSelfPlayCoordinator = original


if __name__ == "__main__": main()
