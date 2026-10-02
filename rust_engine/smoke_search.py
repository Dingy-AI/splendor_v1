"""Run one native Model 4 search with the repository's inference checkpoint."""
import argparse
import json
from pathlib import Path

from splendor_v1.rust_engine import reset
from splendor_v1.rust_engine.mcts import RustMCTS


def main():
    import torch
    from splendor_v1.network.model_4_legal_scorer import SplendorNetwork
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path,
        default=Path(__file__).parents[1] / "training_v6/data/model4_v6_inference_latest.pt")
    parser.add_argument("--simulations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.simulations <= 0 or args.seed < 0:
        parser.error("simulations must be positive and seed nonnegative")
    if args.device == "cpu":
        torch.set_num_threads(1)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    weights = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    model = SplendorNetwork()
    model.load_state_dict(weights)
    model.to(args.device).eval()
    search = RustMCTS(reset(args.seed), model=model, seed=args.seed,
                      simulations=args.simulations)
    chosen = search.search()
    summary = search.summary
    print(json.dumps({"checkpoint": str(args.checkpoint), "device": args.device,
        "chosen_action_id": chosen, "root_visits": summary["visits"],
        "node_count": summary["node_count"], "materialized_states": summary["materialized_states"],
        "search_metadata": summary["metadata"]}, indent=2))
    if chosen is not None:
        search.advance(chosen)
        print(f"Move applied natively; next player: {search.root_state().current_player}")


if __name__ == "__main__":
    main()
