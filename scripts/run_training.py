import torch

from splendor_v1.env.env import SplendorEnv
from splendor_v1.env.core.constants import OBSERVATION_SIZE

from splendor_v1.env.core.action_constants import ACTION_SPACE_SIZE
from splendor_v1.network.model import SplendorNetwork
from splendor_v1.training.replay_buffer import ReplayBuffer
from splendor_v1.training.train import run_training


def main():

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

    history = run_training(
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

    print("\nTraining complete.")
    print(
        f"Replay buffer size: "
        f"{len(replay_buffer)}"
    )

    for iteration, stats in enumerate(
        history,
        start=1,
    ):
        print(
            f"\nIteration {iteration}"
        )
        print(
            f"Loss: "
            f"{stats['loss']:.4f}"
        )
        print(
            f"Policy loss: "
            f"{stats['policy_loss']:.4f}"
        )
        print(
            f"Value loss: "
            f"{stats['value_loss']:.4f}"
        )


if __name__ == "__main__":
    main()