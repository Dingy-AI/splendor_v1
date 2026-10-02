"""Run the user's existing V6 training configuration with native self-play.

Run from the directory containing splendor_v1:
    python -m splendor_v1.rust_engine.run_training

All paths, resume choices, iteration sizes, search settings, optimizer/replay
settings, validation, and checkpoint handling come from run_training_v6.py.
"""
import argparse
from functools import partial
from splendor_v1.rust_engine.coordinator import NativeSelfPlayCoordinator
from splendor_v1.rust_engine.failures import add_failure_arguments


def main(argv=None):
    from splendor_v1.training_v6 import run_training_v6 as training
    parser = argparse.ArgumentParser(description=__doc__)
    add_failure_arguments(parser)
    args = parser.parse_args(argv)
    if args.max_rejected_games < 0:
        parser.error("max-rejected-games must be nonnegative")
    original = training.MultiprocessSelfPlayCoordinator
    print("Self-play backend: Rust game engine/MCTS, native worker threads, shared PyTorch batching")
    print(f"Shared dead-end rejection limit: {args.max_rejected_games} per block; "
          f"diagnostics: {args.failure_dir}", flush=True)
    # V6 resolves this dependency when it constructs each iteration's coordinator.
    training.MultiprocessSelfPlayCoordinator = partial(NativeSelfPlayCoordinator,
        max_rejected_games=args.max_rejected_games, failure_dir=args.failure_dir)
    try:
        training.main()
    finally:
        training.MultiprocessSelfPlayCoordinator = original


if __name__ == "__main__":
    main()
