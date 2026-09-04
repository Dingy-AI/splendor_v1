import torch

from splendor_v1.agents.random_agent import RandomAgent
from splendor_v1.agents.neural_puct_agent import NeuralPUCTAgent
from splendor_v1.env.core.constants import OBSERVATION_SIZE
from splendor_v1.env.core.action_constants import ACTION_SPACE_SIZE
from splendor_v1.network.model import SplendorNetwork
from splendor_v1.evaluation.evaluate_agents import evaluate_agents


def main_random_vs_puct():

    # -------------------------
    # Load trained model
    # -------------------------

    model = SplendorNetwork(
        OBSERVATION_SIZE,
        ACTION_SPACE_SIZE,
    )

    checkpoint = torch.load(
        "checkpoints/model_10_games_last.pt",
        map_location="cpu",
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )


    model.eval()

    # -------------------------
    # Create evaluation agents
    # -------------------------

    trained_agent = NeuralPUCTAgent(
        model=model,
        simulations=80,
    )

    random_agent = RandomAgent()

    # -------------------------
    # Evaluate
    # -------------------------

    print("\nEvaluating trained PUCT vs random...")

    results = evaluate_agents(
        agent_a=trained_agent,
        agent_b=random_agent,
        num_games=20,
        max_steps=300,
        debug_mode=False,
    )

    # -------------------------
    # Print results
    # -------------------------

    print("\nEvaluation complete.")

    print(
        f"PUCT wins: "
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
        f"PUCT win rate: "
        f"{results['agent_a_win_rate']:.2%}"
    )

    print(
        f"Average steps: "
        f"{results['average_steps']:.1f}"
    )


if __name__ == "__main__":
    main_random_vs_puct()

