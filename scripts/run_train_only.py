import numpy as np
import torch

from splendor_v1.env.core.constants import OBSERVATION_SIZE
from splendor_v1.env.core.action_constants import ACTION_SPACE_SIZE
from splendor_v1.network.model import SplendorNetwork
from splendor_v1.training.replay_buffer import ReplayBuffer
from splendor_v1.network.losses import policy_value_loss
from splendor_v1.training.checkpoint import (
    load_checkpoint,
    save_checkpoint,
)


CHECKPOINT_PATH = "checkpoints/keep_model_603_games_clone.pt"
REPLAY_BUFFER_PATH = "checkpoints/keep_replay_603_games.pkl"

BATCH_SIZE = 32

# How many optimizer steps between saved models.
STEPS_PER_ROUND = 50

# 20 rounds * 50 = 1000 additional optimizer steps.
NUM_ROUNDS = 20


def train_steps(
    model,
    optimizer,
    replay_buffer,
    batch_size,
    training_steps,
):
    model.train()

    total_losses = []
    policy_losses = []
    value_losses = []
    policy_kls = []

    for _ in range(training_steps):

        batch = replay_buffer.sample(
            batch_size
        )

        observations = torch.as_tensor(
            np.stack([
                sample[0]
                for sample in batch
            ]),
            dtype=torch.float32,
        )

        target_policies = torch.as_tensor(
            np.stack([
                sample[1]
                for sample in batch
            ]),
            dtype=torch.float32,
        )

        target_values = torch.as_tensor(
            np.array([
                sample[2]
                for sample in batch
            ]),
            dtype=torch.float32,
        )

        policy_logits, predicted_values = model(
            observations
        )

        (
            loss,
            policy_loss,
            value_loss,
            policy_kl,
        ) = policy_value_loss(
            policy_logits,
            predicted_values,
            target_policies,
            target_values,
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_losses.append(
            loss.item()
        )

        policy_losses.append(
            policy_loss.item()
        )

        value_losses.append(
            value_loss.item()
        )

        policy_kls.append(
            policy_kl.item()
        )

    return {
        "total_loss": np.mean(
            total_losses
        ),
        "policy_loss": np.mean(
            policy_losses
        ),
        "value_loss": np.mean(
            value_losses
        ),
        "policy_kl": np.mean(
            policy_kls
        ),
    }


def main():

    model = SplendorNetwork(
        OBSERVATION_SIZE,
        ACTION_SPACE_SIZE,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    # -----------------------------
    # Load 603-game checkpoint
    # -----------------------------

    checkpoint_info = load_checkpoint(
        path=CHECKPOINT_PATH,
        model=model,
        optimizer=optimizer,
    )

    print(
        f"Loaded model from "
        f"{CHECKPOINT_PATH}"
    )

    print(
        f"Original games played: "
        f"{checkpoint_info['games_played']}"
    )

    # -----------------------------
    # Load frozen replay buffer
    # -----------------------------

    replay_buffer = ReplayBuffer.load(
        REPLAY_BUFFER_PATH
    )

    print(
        f"Loaded replay buffer: "
        f"{len(replay_buffer)} positions"
    )

    # IMPORTANT:
    # replay_buffer is NEVER modified.
    # No self-play occurs in this script.

    cumulative_steps = 0

    # -----------------------------
    # Training-only experiment
    # -----------------------------

    for round_index in range(
        1,
        NUM_ROUNDS + 1,
    ):

        metrics = train_steps(
            model=model,
            optimizer=optimizer,
            replay_buffer=replay_buffer,
            batch_size=BATCH_SIZE,
            training_steps=STEPS_PER_ROUND,
        )

        cumulative_steps += STEPS_PER_ROUND

        print(
            f"\nRound {round_index}"
        )

        print(
            f"Additional training steps: "
            f"{cumulative_steps}"
        )

        print(
            f"Total loss: "
            f"{metrics['total_loss']:.4f}"
        )

        print(
            f"Policy loss: "
            f"{metrics['policy_loss']:.4f}"
        )

        print(
            f"Value loss: "
            f"{metrics['value_loss']:.4f}"
        )

        print(
            f"Policy KL: "
            f"{metrics['policy_kl']:.4f}"
        )

        # Save each stage so we can evaluate them
        # separately afterward.

        save_path = (
            "checkpoints/"
            f"model_603_fixed_replay_"
            f"plus_{cumulative_steps}_steps.pt"
        )

        save_checkpoint(
            path=save_path,
            model=model,
            optimizer=optimizer,
            games_played=checkpoint_info[
                "games_played"
            ],
            history=[],
        )

        print(
            f"Saved: {save_path}"
        )


if __name__ == "__main__":
    main()