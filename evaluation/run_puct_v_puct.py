import torch

from splendor_v1.agents.random_agent import RandomAgent
from splendor_v1.agents.neural_puct_agent import NeuralPUCTAgent
from splendor_v1.env.core.constants import OBSERVATION_SIZE
from splendor_v1.env.core.action_constants import ACTION_SPACE_SIZE
from splendor_v1.network.model import SplendorNetwork
from splendor_v1.evaluation.evaluate_agents import evaluate_agents


def main_puct_vs_puct():

    # -------------------------
    # Load trained model
    # -------------------------

    model_1 = SplendorNetwork(
        OBSERVATION_SIZE,
        ACTION_SPACE_SIZE,
    )

    checkpoint_1 = torch.load(
        "checkpoints_archived/model_1409_games.pt",
        map_location="cpu",
        weights_only=False

    )

    model_1.load_state_dict(
        checkpoint_1["model_state_dict"]
    )


    model_1.eval()

    model_2 = SplendorNetwork(
        OBSERVATION_SIZE,
        ACTION_SPACE_SIZE,
    )

    checkpoint_2 = torch.load(
        "checkpoints/model_906_games.pt",
        map_location="cpu",
        weights_only=False

    )

    model_2.load_state_dict(
        checkpoint_2["model_state_dict"]
    )


    model_2.eval()


    # -------------------------
    # Create evaluation agents
    # -------------------------

    puct_agent_1 = NeuralPUCTAgent(
        model=model_1,
        simulations=400,
        debug_mode=False, 
        teacher_mode=False,
        name="PUCT AGENT 1409 400sim"
    )

    puct_agent_2 = NeuralPUCTAgent(
        model=model_2,
        simulations=400,
        debug_mode=False, 
        teacher_mode=False,
        name="PUCT Agent 906 400sim"

    )
    # -------------------------
    # Evaluate
    # -------------------------

    print("\nEvaluating trained PUCT vs PUCT...")

    results = evaluate_agents(
        agent_a=puct_agent_1,
        agent_b=puct_agent_2,
        num_games=50,
        max_steps=300,
        debug_mode=True,
        # seed=500000,
        seed=500025,
        is_evaluation_dynamic = True
    )

    # -------------------------
    # Print results
    # -------------------------

    print("\nEvaluation complete.")

    print(
        f"PUCT_1 wins: "
        f"{results['agent_a_wins']}"
    )

    print(
        f"PUCT_2 wins: "
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
        f"PUCT_1 win rate: "
        f"{results['agent_a_win_rate']:.2%}"
    )

    print(
        f"Average steps: "
        f"{results['average_steps']:.1f}"
    )


if __name__ == "__main__":
    main_puct_vs_puct()

