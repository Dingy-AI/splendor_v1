import torch

from splendor_v1.agents.heuristic_agent_2 import HeuristicAgent2
from splendor_v1.env.core.constants import OBSERVATION_SIZE
from splendor_v1.env.core.action_constants import ACTION_SPACE_SIZE
from splendor_v1.network.model import SplendorNetwork
from splendor_v1.evaluation.evaluate_agents import evaluate_agents
from splendor_v1.agents.neural_puct_agent import NeuralPUCTAgent


def main_puct_vs_heuristic_1():

    # -------------------------
    # Create evaluation agents
    # -------------------------
    model = SplendorNetwork(
        OBSERVATION_SIZE,
        ACTION_SPACE_SIZE,
    )

    checkpoint = torch.load(
        "checkpoints/model_2802_games.pt",
        map_location="cpu",
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )


    model.eval()

    trained_agent = NeuralPUCTAgent(
        model=model,
        simulations=200,
        debug_mode=False, 
        teacher_mode=False
    )

    random_agent = HeuristicAgent2()

    # -------------------------
    # Evaluate
    # -------------------------

    print("\nEvaluating trained PUCT vs HeuristicAgent2...")

    results = evaluate_agents(
        agent_a=trained_agent,
        agent_b=random_agent,
        num_games=100,
        max_steps=300,
        debug_mode=True,
        seed=None
    )

    # -------------------------
    # Print results
    # -------------------------

    print("\nEvaluation complete.")

    print(
        f"Puct wins: "
        f"{results['agent_a_wins']}"
    )

    print(
        f"HeuristicAgent2 wins: "
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
        f"Puct win rate: "
        f"{results['agent_a_win_rate']:.2%}"
    )

    print(
        f"Average steps: "
        f"{results['average_steps']:.1f}"
    )


if __name__ == "__main__":
    main_puct_vs_heuristic_1()

