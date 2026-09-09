import torch

from splendor_v1.network.losses import policy_value_loss
import numpy as np
from splendor_v1.training.checkpoint import save_model_if_needed, save_checkpoint
from splendor_v1.training.self_play import play_self_play_game
from splendor_v1.mcts.mcts import MCTS
import time
from splendor_v1.evaluation.evaluate_agents import evaluate_model_vs_greedy, evaluate_model_vs_random

def run_training(
    env,
    model,
    optimizer,
    replay_buffer,
    num_iterations,
    self_play_games_per_iteration,
    simulations,
    batch_size,
    training_ratio=1.5,
    checkpoint_every_games=None,
    checkpoint_dir="checkpoints",
    starting_games_played=0,
    policy_debug_samples=None,
    seed=None,
    dynamic_seeding=False,
    teacher_mode=False,
    writer=None,
    evaluation_seed=None,
    is_evaluation_dynamic=False,
    scheduler=None
):
    history = []

    games_attempted = starting_games_played
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
        iteration_start = time.perf_counter()
        positions_added = 0
        iteration_game_lengths = []


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
                policy_debug_samples,
                game_index= games_attempted,
                seed=seed,
                dynamic_seeding=dynamic_seeding,
                teacher_mode=teacher_mode
            )

            games_attempted += 1
            if  game_stats['completed']:
                total_mcts_time += (
                    game_stats["mcts_time"]
                )
                # print("Games Completed Successfully: ", games_played)
                
                iteration_game_lengths.append(
                    game_stats["positions_added"]
                )


                games_played += 1
                positions_added += game_stats['positions_added']
            else:
                print("Game crashed and Terminated Early.")


        training_steps = max(
            1,
            round(
                positions_added
                * training_ratio
                / batch_size
            )
        )
        print("Number of Training Steps: ", training_steps)
        training_results = train_network(
            model=model,
            replay_buffer=replay_buffer,
            optimizer=optimizer,
            batch_size=batch_size,
            training_steps=training_steps,
            scheduler=scheduler
        )

        iteration_time = (
            time.perf_counter()
            - iteration_start
        )

        average_game_length = np.mean(
            iteration_game_lengths
        )


        if writer is not None:
            writer.add_scalar(
                "SelfPlay/average_game_length",
                average_game_length,
                games_played,
            )

            writer.add_scalar(
                "Loss/total",
                training_results["average_total_loss"],
                games_played,
            )

            writer.add_scalar(
                "Loss/policy",
                training_results["average_policy_loss"],
                games_played,
            )

            writer.add_scalar(
                "Loss/value",
                training_results["average_value_loss"],
                games_played,
            )

            writer.add_scalar(
                "Loss/policy_kl",
                training_results["average_policy_kl"],
                games_played,
            )

            writer.add_scalar(
                "Training/gradient_norm",
                training_results["average_grad_norm"],
                games_played,
            )

            writer.add_scalar(
                "Training/predicted_value",
                training_results["average_predicted_value"],
                games_played,
            )

            writer.add_scalar(
                "Training/target_value",
                training_results["average_target_value"],
                games_played,
            )

            writer.add_scalar(
                "Training/target_policy_entropy",
                training_results[
                    "average_target_policy_entropy"
                ],
                games_played,
            )

            writer.add_scalar(
                "Training/replay_size",
                len(replay_buffer),
                games_played,
            )

            writer.add_scalar(
                "Training/positions_added",
                positions_added,
                games_played,
            )

            writer.add_scalar(
                "Training/training_steps",
                training_steps,
                games_played,
            )

            writer.add_scalar(
                "Training/learning_rate",
                optimizer.param_groups[0]["lr"],
                games_played,
            )
            writer.add_scalar(
                "Timing/mcts_seconds",
                total_mcts_time,
                games_played,
            )

            writer.add_scalar(
                "Timing/iteration_seconds",
                iteration_time,
                games_played,
            )

            if positions_added > 0:
                writer.add_scalar(
                    "Timing/mcts_seconds_per_position",
                    total_mcts_time / positions_added,
                    games_played,
                )

        history.append(training_results)


        if checkpoint_every_games is not None:
            next_checkpoint, checkpoint_saved = save_model_if_needed(
                model=model,
                optimizer=optimizer,
                games_played=games_played,
                history=history,
                checkpoint_every_games=checkpoint_every_games,
                next_checkpoint=next_checkpoint,
                checkpoint_dir=checkpoint_dir,
                replay_buffer=replay_buffer,
                scheduler=scheduler
            )

            if checkpoint_saved:
                results_random = evaluate_model_vs_random(
                    model=model,
                    num_games=20,
                    simulations=200,
                    seed=evaluation_seed,
                    is_evaluation_dynamic=is_evaluation_dynamic
                )
                print(results_random)

                if writer is not None:

                    writer.add_scalar(
                        "Evaluation/Random_win_rate",
                        results_random["agent_a_win_rate"],
                        games_played,
                    )

                    writer.add_scalar(
                        "Evaluation/Random_average_steps",
                        results_random["average_steps"],
                        games_played,
                    )

                    writer.flush()

                results_greedy = evaluate_model_vs_greedy(
                    model=model,
                    num_games=20,
                    simulations=200,
                    seed=evaluation_seed,
                    is_evaluation_dynamic=is_evaluation_dynamic
                )
                if writer is not None:

                    writer.add_scalar(
                        "Evaluation/Greedy_win_rate",
                        results_greedy["agent_a_win_rate"],
                        games_played,
                    )

                    writer.add_scalar(
                        "Evaluation/Greedy_average_steps",
                        results_greedy["average_steps"],
                        games_played,
                    )
                    writer.flush()

        print(
            f"\nIteration {iteration + 1}"
        )

        print(
            f"MCTS search time: "
            f"{total_mcts_time:.2f}s"
        )

        print(
            f"Iteration time: "
            f"{iteration_time:.2f}s"
        )

        print("Games played:", games_played)
        print("Replay size:", len(replay_buffer))
        print("Replay position:", replay_buffer.position)

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
    scheduler=None
):
    """
    Train the network using samples from the replay buffer.

    Returns:
        dict containing average training metrics
        across all optimizer steps.
    """

    if len(replay_buffer) < batch_size:
        raise ValueError(
            f"Not enough samples in replay buffer. "
            f"Need {batch_size}, have {len(replay_buffer)}."
        )

    model.train()
    iteration_total_losses = []
    iteration_policy_losses = []
    iteration_value_losses = []
    iteration_policy_kls = []

    iteration_grad_norms = []

    iteration_predicted_value_means = []
    iteration_target_value_means = []
    iteration_target_policy_entropies = []
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


        loss, policy_loss, value_loss, policy_kl = (
            policy_value_loss(
                policy_logits,
                predicted_values,
                target_policies,
                target_values,
            )
        )


             
        optimizer.zero_grad()

        loss.backward()


        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=float("inf"),
        )



        optimizer.step()


        if scheduler is not None:
            scheduler.step()


        with torch.no_grad():

            predicted_value_mean = (
                predicted_values.mean().item()
            )

            target_value_mean = (
                target_values.mean().item()
            )

            target_policy_entropy = -(
                target_policies
                * torch.log(
                    target_policies.clamp_min(1e-8)
                )
            ).sum(dim=-1).mean().item()

        iteration_predicted_value_means.append(
            predicted_value_mean
        )

        iteration_target_value_means.append(
            target_value_mean
        )

        iteration_target_policy_entropies.append(
            target_policy_entropy
        )





        iteration_grad_norms.append(grad_norm.item())

        iteration_total_losses.append(
                loss.item()
            )

        iteration_policy_losses.append(
                policy_loss.item()
            )

        iteration_value_losses.append(
                value_loss.item()
            )

        iteration_policy_kls.append(
                policy_kl.item()
            )


    return {
        "average_total_loss": np.mean(
            iteration_total_losses
        ),
        "average_policy_loss": np.mean(
            iteration_policy_losses
        ),
        "average_value_loss": np.mean(
            iteration_value_losses
        ),
        "average_policy_kl": np.mean(
            iteration_policy_kls
        ),
        "average_grad_norm": np.mean(
            iteration_grad_norms
        ),
        "average_predicted_value": np.mean(
            iteration_predicted_value_means
        ),
        "average_target_value": np.mean(
            iteration_target_value_means
        ),
        "average_target_policy_entropy": np.mean(
            iteration_target_policy_entropies
        ),
    }

