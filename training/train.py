import torch

from splendor_v1.network.losses import policy_value_loss
import numpy as np
from splendor_v1.training.self_play import play_self_play_game
from splendor_v1.mcts.mcts import MCTS


def run_training(
    env,
    model,
    optimizer,
    replay_buffer,
    num_iterations,
    self_play_games_per_iteration,
    simulations,
    batch_size,
    training_steps,
):
    history = []

    for iteration in range(num_iterations):

        mcts = MCTS(
            simulations=simulations,
            rollout_type="neural",
            selection_type="puct",
            model=model,
        )

        for _ in range(
            self_play_games_per_iteration
        ):
            play_self_play_game(
                env,
                mcts,
                replay_buffer,
            )

        stats = train_network(
            model=model,
            replay_buffer=replay_buffer,
            optimizer=optimizer,
            batch_size=batch_size,
            training_steps=training_steps,
        )

        history.append(stats)

    return history



def train_network(
    model,
    replay_buffer,
    optimizer,
    batch_size,
    training_steps,
):
    """
    Train the network using samples from the replay buffer.

    Returns:
        dict containing average losses across all training steps.
    """

    if len(replay_buffer) < batch_size:
        raise ValueError(
            f"Not enough samples in replay buffer. "
            f"Need {batch_size}, have {len(replay_buffer)}."
        )

    model.train()

    total_loss_sum = 0.0
    policy_loss_sum = 0.0
    value_loss_sum = 0.0

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
        ) = policy_value_loss(
            policy_logits,
            predicted_values,
            target_policies,
            target_values,
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss_sum += loss.item()
        policy_loss_sum += policy_loss.item()
        value_loss_sum += value_loss.item()

    return {
        "loss": (
            total_loss_sum
            / training_steps
        ),
        "policy_loss": (
            policy_loss_sum
            / training_steps
        ),
        "value_loss": (
            value_loss_sum
            / training_steps
        ),
    }