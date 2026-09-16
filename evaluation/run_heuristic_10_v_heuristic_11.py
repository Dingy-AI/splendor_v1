import torch

from splendor_v1.agents.heuristic_agent_11 import HeuristicAgent11
from splendor_v1.agents.heuristic_agent_10 import HeuristicAgent10
from splendor_v1.env.core.constants import OBSERVATION_SIZE
from splendor_v1.env.core.action_constants import ACTION_SPACE_SIZE
from splendor_v1.network.model import SplendorNetwork
from splendor_v1.evaluation.evaluate_agents import evaluate_agents


def main_heuristic_vs_greedy():

    # -------------------------
    # Create evaluation agents
    # -------------------------

    trained_agent = HeuristicAgent10(
    )

    greedy_agent = HeuristicAgent11()

    # -------------------------
    # Evaluate
    # -------------------------

    print("\nEvaluating trained HeuristicAgent10 vs HeuristicAgent11...")


    results = evaluate_agents(
        agent_a=trained_agent,
        agent_b=greedy_agent,
        num_games=50,
        max_steps=300,
        debug_mode=True,
        seed=500000,
        is_evaluation_dynamic = True

    )
    # ONLY SLIGHTLY BETTER :)
    # Evaluation complete.
    # HeuristicAgent2 wins: 54
    # HeuristicAgent wins: 46
    # Ties: 0
    # Deadlocks: 0
    # Aborted: 0
    # HeuristicAgent2 win rate: 54.00%
    # Average steps: 61.8

    # -------------------------
    # Print results
    # -------------------------

    print("\nEvaluation complete.")

    print(
        f"HeuristicAgent10 wins: "
        f"{results['agent_a_wins']}"
    )

    print(
        f"HeuristicAgent11 wins: "
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
        f"HeuristicAgent10 win rate: "
        f"{results['agent_a_win_rate']:.2%}"
    )

    print(
        f"Average steps: "
        f"{results['average_steps']:.1f}"
    )


if __name__ == "__main__":
    main_heuristic_vs_greedy()

