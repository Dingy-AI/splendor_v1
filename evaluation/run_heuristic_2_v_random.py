import torch

from splendor_v1.agents.random_agent import RandomAgent
from splendor_v1.agents.heuristic_agent_2 import HeuristicAgent2
from splendor_v1.env.core.constants import OBSERVATION_SIZE
from splendor_v1.env.core.action_constants import ACTION_SPACE_SIZE
from splendor_v1.network.model import SplendorNetwork
from splendor_v1.evaluation.evaluate_agents import evaluate_agents


def main_greedy_vs_random():

    # -------------------------
    # Create evaluation agents
    # -------------------------

    trained_agent = HeuristicAgent2(
    )

    random_agent = RandomAgent()

    # -------------------------
    # Evaluate
    # -------------------------

    print("\nEvaluating trained HeuristicAgent2 vs random...")

    results = evaluate_agents(
        agent_a=trained_agent,
        agent_b=random_agent,
        num_games=100,
        max_steps=300,
        debug_mode=True,
        seed=420
    )

    # Evaluation complete.
    # HeuristicAgent2 wins: 98
    # Random wins: 2
    # Ties: 0
    # Deadlocks: 0
    # Aborted: 0
    # HeuristicAgent2 win rate: 98.00%
    # Average steps: 75.1

    # -------------------------
    # Print results
    # -------------------------

    print("\nEvaluation complete.")

    print(
        f"HeuristicAgent2 wins: "
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
        f"HeuristicAgent2 win rate: "
        f"{results['agent_a_win_rate']:.2%}"
    )

    print(
        f"Average steps: "
        f"{results['average_steps']:.1f}"
    )


if __name__ == "__main__":
    main_greedy_vs_random()

