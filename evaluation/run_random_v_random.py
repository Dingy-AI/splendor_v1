import torch

from splendor_v1.agents.random_agent import RandomAgent
from splendor_v1.agents.greedy_agent import GreedyAgent
from splendor_v1.env.core.constants import OBSERVATION_SIZE
from splendor_v1.env.core.action_constants import ACTION_SPACE_SIZE
from splendor_v1.network.model import SplendorNetwork
from splendor_v1.evaluation.evaluate_agents import evaluate_agents


def main_random_vs_random():

    # -------------------------
    # Create evaluation agents
    # -------------------------

    random_agent_1 = RandomAgent(
    )

    random_agent_2 = RandomAgent()

    # -------------------------
    # Evaluate
    # -------------------------

    print("\nEvaluating trained PUCT vs random...")

    results = evaluate_agents(
        agent_a=random_agent_1,
        agent_b=random_agent_2,
        num_games=100,
        max_steps=300,
        debug_mode=True,
        seed=420
    )

    # -------------------------
    # Print results
    # -------------------------

    print("\nEvaluation complete.")

    print(
        f"Random_1 wins: "
        f"{results['agent_a_wins']}"
    )

    print(
        f"Random_2 wins: "
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
        f"Random_1 win rate: "
        f"{results['agent_a_win_rate']:.2%}"
    )

    print(
        f"Average steps: "
        f"{results['average_steps']:.1f}"
    )


if __name__ == "__main__":
    main_random_vs_random()

