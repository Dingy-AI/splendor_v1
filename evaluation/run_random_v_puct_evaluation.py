import torch

from splendor_v1.agents.random_agent import RandomAgent
from splendor_v1.agents.neural_puct_agent import NeuralPUCTAgent
from splendor_v1.env.env import SplendorEnv
from splendor_v1.env.core.constants import OBSERVATION_SIZE
from splendor_v1.env.core.action_constants import ACTION_SPACE_SIZE
from splendor_v1.network.model import SplendorNetwork
from splendor_v1.training.replay_buffer import ReplayBuffer
from splendor_v1.training.train import run_training
from splendor_v1.evaluation.evaluate_agents import evaluate_agents


def main_random_vs_puct():

    # -------------------------
    # Create training objects
    # -------------------------

    env = SplendorEnv()

    model = SplendorNetwork(
        OBSERVATION_SIZE,
        ACTION_SPACE_SIZE,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    replay_buffer = ReplayBuffer(
        max_size=100_000,
    )

    # -------------------------
    # Train model in-place
    # -------------------------

    print("Training model...")

    run_training(
        env=env,
        model=model,
        optimizer=optimizer,
        replay_buffer=replay_buffer,
        num_iterations=3,
        self_play_games_per_iteration=2,
        simulations=5,
        batch_size=32,
        training_steps=10,
    )

    # -------------------------
    # Create evaluation agents
    # -------------------------

    trained_agent = NeuralPUCTAgent(
        model=model,
        simulations=5,
    )

    random_agent = RandomAgent()

    # -------------------------
    # Evaluate
    # -------------------------

    print("\nEvaluating trained PUCT vs random...")

    results = evaluate_agents(
        agent_a=trained_agent,
        agent_b=random_agent,
        num_games=10,
        max_steps=300,
        debug_mode=True,
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