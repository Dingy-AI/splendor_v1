import torch

from splendor_v1.agents.random_agent import RandomAgent
from splendor_v1.agents.greedy_agent import GreedyAgent
from splendor_v1.env.core.constants import OBSERVATION_SIZE
from splendor_v1.env.core.action_constants import ACTION_SPACE_SIZE
from splendor_v1.network.model import SplendorNetwork
from splendor_v1.evaluation.evaluate_agents import evaluate_agents


def main_greedy_vs_random():

    # -------------------------
    # Create evaluation agents
    # -------------------------

    trained_agent = GreedyAgent(
    )

    random_agent = RandomAgent()

    # -------------------------
    # Evaluate
    # -------------------------

    print("\nEvaluating trained GreedyAgent vs random...")

    results = evaluate_agents(
        agent_a=trained_agent,
        agent_b=random_agent,
        num_games=100,
        max_steps=300,
        debug_mode=True,
        seed=420
    )

    #     Evaluation complete.
    # Greedy wins: 99
    # Random wins: 1
    # Ties: 0
    # Deadlocks: 0
    # Aborted: 0
    # Greedy win rate: 99.00%
    # Average steps: 77.0

    # -------------------------
    # Print results
    # -------------------------

    print("\nEvaluation complete.")

    print(
        f"Greedy wins: "
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
        f"Greedy win rate: "
        f"{results['agent_a_win_rate']:.2%}"
    )

    print(
        f"Average steps: "
        f"{results['average_steps']:.1f}"
    )


if __name__ == "__main__":
    main_greedy_vs_random()

