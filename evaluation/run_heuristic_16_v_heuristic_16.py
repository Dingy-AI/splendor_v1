import torch

from splendor_v1.agents.random_agent import RandomAgent
from splendor_v1.agents.heuristic_agent_16.heuristic_agent_16 import HeuristicAgent16Diagnostics
from splendor_v1.env.core.constants import OBSERVATION_SIZE
from splendor_v1.env.core.action_constants import ACTION_SPACE_SIZE
from splendor_v1.network.model import SplendorNetwork
from splendor_v1.evaluation.evaluate_agents import evaluate_agents


def main_greedy_vs_random():

    # -------------------------
    # Create evaluation agents
    # -------------------------

    trained_agent = HeuristicAgent16Diagnostics(num_rollouts=8, name="default"
    )

    random_agent = HeuristicAgent16Diagnostics(num_rollouts=0, name="0 rollout")

    # -------------------------
    # Evaluate
    # -------------------------

    print("\nEvaluating trained HeuristicAgent16Diagnostics vs 0_rollout...")

    results = evaluate_agents(
        agent_a=trained_agent,
        agent_b=random_agent,
        num_games=10,
        max_steps=300,
        debug_mode=True,
        seed=200000,
        is_evaluation_dynamic = True

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
        f"HeuristicAgent16Diagnostics wins: "
        f"{results['agent_a_wins']}"
    )

    print(
        f"0_rollout wins: "
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
        f"HeuristicAgent16Diagnostics win rate: "
        f"{results['agent_a_win_rate']:.2%}"
    )

    print(
        f"Average steps: "
        f"{results['average_steps']:.1f}"
    )
    print(trained_agent.format_diagnostic_summary())
    print(random_agent.format_diagnostic_summary())


if __name__ == "__main__":
    main_greedy_vs_random()

