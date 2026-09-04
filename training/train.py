import torch

from splendor_v1.network.losses import policy_value_loss
import numpy as np
from splendor_v1.training.checkpoint import save_model_if_needed, save_checkpoint
from splendor_v1.training.self_play import play_self_play_game
from splendor_v1.mcts.mcts import MCTS
import time

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
    checkpoint_every_games=None,
    checkpoint_dir="checkpoints",
    starting_games_played=0,
):
    history = []

    games_played = starting_games_played

    if checkpoint_every_games is not None:
        next_checkpoint = (
            (
                games_played
                // checkpoint_every_games
            )
            + 1
        ) * checkpoint_every_games
    else:
        next_checkpoint = None

    
    for iteration in range(num_iterations):
        start = time.perf_counter()

        total_self_play_time = 0.0
        total_mcts_time = 0.0

        mcts = MCTS(
            simulations=simulations,
            rollout_type="neural",
            selection_type="puct",
            model=model,
        )

        for _ in range(
            self_play_games_per_iteration
        ):
            game_stats = play_self_play_game(
                env,
                mcts,
                replay_buffer,
            )



            total_self_play_time += (
                time.perf_counter() - start
            )
            if  game_stats['completed']:
                total_mcts_time += (
                    game_stats["mcts_time"]
                )

                games_played += 1

            else:
                print("Game crashed and Terminated Early.")

        train_start = time.perf_counter()

        stats = train_network(
            model=model,
            replay_buffer=replay_buffer,
            optimizer=optimizer,
            batch_size=batch_size,
            training_steps=training_steps,
        )



        train_time = (
            time.perf_counter()
            - train_start
        )

        history.append(stats)


        if checkpoint_every_games is not None:
            next_checkpoint = save_model_if_needed(
                model=model,
                optimizer=optimizer,
                games_played=games_played,
                history=history,
                checkpoint_every_games=checkpoint_every_games,
                next_checkpoint=next_checkpoint,
                checkpoint_dir=checkpoint_dir,
            )


        print(
            f"\nIteration {iteration + 1}"
        )

        print(
            f"Self-play time: "
            f"{total_self_play_time:.2f}s"
        )

        print(
            f"MCTS search time: "
            f"{total_mcts_time:.2f}s"
        )

        print(
            f"Network training time: "
            f"{train_time:.2f}s"
        )

    save_checkpoint(
        path=(
            f"{checkpoint_dir}/"
            f"model_{games_played}_games_last.pt"
        ),
        model=model,
        optimizer=optimizer,
        games_played=games_played,
        history=history,
    )

    print(
        f"Final checkpoint saved: "
        f"{games_played} games"
    )




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

    for step in range(training_steps):

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

        # ---------------------------------
        # Debug first batch only
        # ---------------------------------


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

