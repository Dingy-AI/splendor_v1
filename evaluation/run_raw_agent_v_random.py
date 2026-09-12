import torch

from splendor_v1.agents.random_agent import RandomAgent
from splendor_v1.agents.raw_network_agent import RawNetworkAgent
from splendor_v1.env.core.constants import OBSERVATION_SIZE
from splendor_v1.env.core.action_constants import ACTION_SPACE_SIZE
from splendor_v1.network.model import SplendorNetwork
from splendor_v1.evaluation.evaluate_agents import evaluate_agents


def main_random_vs_puct():
    # -------------------------
    # Create evaluation agents
    # -------------------------

    trained_agent = RawNetworkAgent(
        "checkpoints/heuristic_pretrain/heuristic_pretrain_best.pt",
    )

    random_agent = RandomAgent()

    # -------------------------
    # Evaluate
    # -------------------------

    print("\nEvaluating Raw Agent vs random...")

    results = evaluate_agents(
        agent_a=trained_agent,
        agent_b=random_agent,
        num_games=50,
        max_steps=300,
        debug_mode=True,
        seed=500000,
        is_evaluation_dynamic=True

    )

    # -------------------------
    # Print results
    # -------------------------

    print("\nEvaluation complete.")

    print(
        f"Raw Agent wins: "
        f"{results['agent_a_wins']}"
    )

    print(
        f"Random wins: "
        f"{results['agent_b_wins']}"
    )

    print(
        f"Ties: "
        f"{results['ties']}"
    )

    print(
        f"Deadlocks: "
        f"{results['deadlocks']}"
    )

    print(
        f"Aborted: "
        f"{results['aborted']}"
    )

    print(
        f"Raw Agent win rate: "
        f"{results['agent_a_win_rate']:.2%}"
    )

    print(
        f"Average steps: "
        f"{results['average_steps']:.1f}"
    )


if __name__ == "__main__":
    main_random_vs_puct()

