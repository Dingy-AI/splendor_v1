import os
import time
import traceback
import random
import numpy as np
import torch

from splendor_v1.env.core.action_constants import (
    ACTION_SPACE_SIZE,
)

from splendor_v1.mcts.mcts_v3 import MCTS

from splendor_v1.network.losses_3_wdl import (
    policy_wdl_loss,
)

from splendor_v1.training.checkpoint import (
    save_model_if_needed,
    save_checkpoint,
)

from splendor_v1.training_v2.model_replay_generator import (
    ModelReplayGenerator,
)

from splendor_v1.training_v2.state_serializer import (
    serialize_state,
)


# ============================================================
# MODEL 3 WDL LABELS
# ============================================================
#
# These class IDs must match model_3_attention_wdl.py and
# policy_wdl_loss.py.
#
#     0 = LOSS
#     1 = DRAW
#     2 = WIN
#
# All labels are from sample["current_player"]'s perspective.
# ============================================================

WDL_LOSS = 0
WDL_DRAW = 1
WDL_WIN = 2


# ============================================================
# SEED
# ============================================================

def resolve_game_seed(
    base_seed,
    game_index,
    dynamic_seeding,
):
    """
    Resolve the environment seed for one self-play game.

    dynamic_seeding=False:
        every game uses base_seed.

    dynamic_seeding=True:
        game N uses base_seed + N.

    base_seed=None:
        environment uses its normal random seeding.
    """

    if base_seed is None:
        return None

    if dynamic_seeding:
        return (
            int(base_seed)
            + int(game_index)
        )

    return int(
        base_seed
    )

# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

def choose_game_split(
    seed,
    game_index,
    validation_fraction=0.10,
    split_seed=20260917,
):
    """
    Deterministically assign an ENTIRE game
    to train or validation.

    Both game_index and seed contribute so each
    game receives an independent deterministic
    assignment even when environment seeds repeat.
    """

    if not 0.0 <= validation_fraction <= 1.0:

        raise ValueError(
            "validation_fraction must be "
            "between 0.0 and 1.0."
        )

    seed_component = (
        0
        if seed is None
        else int(seed)
    )

    split_value = (
        (seed_component << 32)
        ^ int(game_index)
        ^ int(split_seed)
    )

    rng = random.Random(
        split_value
    )

    if (
        rng.random()
        < validation_fraction
    ):
        return "val"

    return "train"

# ============================================================
# POLICY TARGET
# ============================================================

def build_policy_target(
    sample,
    action_space_size=ACTION_SPACE_SIZE,
):
    """
    Convert raw MCTS visit counts into the dense policy
    target expected by the existing policy/value network.

    Replay stores:

        policy_action_ids
        visit_counts

    rather than permanently storing a dense policy vector.

    This allows future training code to reinterpret raw
    search information differently if desired.
    """

    action_ids = np.asarray(
        sample["policy_action_ids"],
        dtype=np.int64,
    )

    visits = np.asarray(
        sample["visit_counts"],
        dtype=np.float64,
    )

    # --------------------------------------------------------
    # VALIDATE
    # --------------------------------------------------------

    if len(action_ids) != len(visits):

        raise RuntimeError(
            "policy_action_ids and visit_counts "
            "have different lengths."
        )

    if len(action_ids) == 0:

        raise RuntimeError(
            "Replay sample contains no MCTS "
            "policy actions."
        )

    if np.any(
        action_ids < 0
    ) or np.any(
        action_ids >= action_space_size
    ):

        raise RuntimeError(
            "Replay sample contains an action ID "
            "outside the current action space."
        )

    if np.any(
        visits < 0
    ):

        raise RuntimeError(
            "Replay sample contains negative "
            "MCTS visit counts."
        )

    # --------------------------------------------------------
    # DENSE TARGET
    # --------------------------------------------------------

    target = np.zeros(
        action_space_size,
        dtype=np.float32,
    )

    visit_sum = float(
        visits.sum()
    )

    # --------------------------------------------------------
    # NORMAL CASE
    # --------------------------------------------------------

    if visit_sum > 0:

        probabilities = (
            visits
            / visit_sum
        )

        target[
            action_ids
        ] = probabilities.astype(
            np.float32
        )

        return target

    # --------------------------------------------------------
    # FALLBACK
    #
    # This should be rare.
    #
    # If a forced/search position somehow produced no visits,
    # use the actual action taken as a one-hot policy instead
    # of creating an invalid all-zero target.
    # --------------------------------------------------------

    chosen_action_id = sample.get(
        "chosen_action_id"
    )

    if chosen_action_id is None:

        raise RuntimeError(
            "Replay sample has zero MCTS visits "
            "and no chosen_action_id fallback."
        )

    chosen_action_id = int(
        chosen_action_id
    )

    target[
        chosen_action_id
    ] = 1.0

    return target


# ============================================================
# WDL TARGET
# ============================================================

def build_wdl_target(
    sample,
    replay_buffer,
):
    """
    Build the Model 3 Win / Draw / Loss target from raw replay facts.

    Class order:

        0 = LOSS
        1 = DRAW
        2 = WIN

    Perspective:

        sample["current_player"]

    Replay stores the game's winner_ids separately from each position,
    so old Model 2 replay data can be reinterpreted as WDL without
    regenerating self-play games.

    Splendor can have multiple winners after the score/card-count
    tie-break. For a WDL head, a shared result is represented as DRAW.
    A winnerless result is also represented as DRAW.
    """

    game_id = sample.get(
        "game_id"
    )

    if game_id is None:

        raise RuntimeError(
            "Replay sample is missing game_id."
        )

    game = replay_buffer.games.get(
        game_id
    )

    if game is None:

        raise RuntimeError(
            f"Replay metadata missing for "
            f"game_id={game_id}."
        )

    winner_ids = list(
        game.get(
            "winner_ids",
            [],
        )
    )

    player = int(
        sample["current_player"]
    )

    # --------------------------------------------------------
    # DRAW
    # --------------------------------------------------------
    #
    # No winner:
    #     explicit / future winnerless draw.
    #
    # Multiple winners:
    #     shared Splendor result.
    # --------------------------------------------------------

    if len(winner_ids) != 1:
        return WDL_DRAW

    # --------------------------------------------------------
    # WIN / LOSS
    # --------------------------------------------------------

    if player == int(
        winner_ids[0]
    ):
        return WDL_WIN

    return WDL_LOSS


def wdl_targets_to_scalar(
    target_wdl,
):
    """
    Convert integer WDL classes into the familiar WDL target:

        LOSS -> -1
        DRAW ->  0
        WIN  -> +1

    This is for diagnostics only. Model 3 trains the WDL head with
    classification loss rather than MSE on this scalar.
    """

    target_values = torch.zeros_like(
        target_wdl,
        dtype=torch.float32,
    )

    target_values = torch.where(
        target_wdl == WDL_WIN,
        torch.ones_like(
            target_values
        ),
        target_values,
    )

    target_values = torch.where(
        target_wdl == WDL_LOSS,
        -torch.ones_like(
            target_values
        ),
        target_values,
    )

    return target_values


# ============================================================
# BUILD TRAINING BATCH
# ============================================================

def build_training_batch(
    batch,
    replay_buffer,
):
    """
    Convert rich replay records into Model 3 targets.

    Returns:

        observations
            float32 [batch, 258]

        target_policies
            float32 [batch, ACTION_SPACE_SIZE]

        target_wdl
            int64 [batch]

            0 = LOSS
            1 = DRAW
            2 = WIN
    """

    observations = np.stack(
        [
            sample["observation"]
            for sample
            in batch
        ]
    ).astype(
        np.float32,
        copy=False,
    )

    target_policies = np.stack(
        [
            build_policy_target(
                sample
            )
            for sample
            in batch
        ]
    )

    target_wdl = np.asarray(
        [
            build_wdl_target(
                sample,
                replay_buffer,
            )
            for sample
            in batch
        ],
        dtype=np.int64,
    )

    return (
        observations,
        target_policies,
        target_wdl,
    )


# ============================================================
# TRAIN NETWORK
# ============================================================

def train_network(
    model,
    replay_buffer,
    optimizer,
    batch_size,
    training_steps,
    scheduler=None,
    split="train",
):
    """
    Train Model 3 from the rich replay format.

    Policy:
        dense MCTS visit distribution -> cross entropy

    Value:
        WDL class from current-player perspective -> cross entropy

    Scalar values are derived only for diagnostics:

        V = P(WIN) - P(LOSS)
    """

    if len(replay_buffer) == 0:

        raise ValueError(
            "Replay buffer is empty."
        )

    model.train()

    device = next(
        model.parameters()
    ).device

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    iteration_total_losses = []
    iteration_policy_losses = []
    iteration_wdl_losses = []
    iteration_policy_kls = []

    iteration_grad_norms = []

    iteration_predicted_value_means = []
    iteration_target_value_means = []
    iteration_target_policy_entropies = []

    iteration_wdl_accuracies = []
    iteration_loss_probabilities = []
    iteration_draw_probabilities = []
    iteration_win_probabilities = []

    iteration_target_loss_fractions = []
    iteration_target_draw_fractions = []
    iteration_target_win_fractions = []

    # ========================================================
    # OPTIMIZER STEPS
    # ========================================================

    for _ in range(
        training_steps
    ):

        # ----------------------------------------------------
        # SAMPLE
        # ----------------------------------------------------

        if split is None:

            batch = replay_buffer.sample(
                batch_size
            )

        else:

            batch = replay_buffer.sample(
                batch_size,
                split=split,
            )

        if not batch:

            raise RuntimeError(
                "ReplayBuffer returned an empty "
                "training batch."
            )

        # ----------------------------------------------------
        # BUILD MODEL-3 TARGETS FROM RAW REPLAY
        # ----------------------------------------------------

        (
            observations_np,
            target_policies_np,
            target_wdl_np,
        ) = build_training_batch(
            batch=batch,
            replay_buffer=replay_buffer,
        )

        # ----------------------------------------------------
        # TORCH
        # ----------------------------------------------------

        observations = torch.as_tensor(
            observations_np,
            dtype=torch.float32,
            device=device,
        )

        target_policies = torch.as_tensor(
            target_policies_np,
            dtype=torch.float32,
            device=device,
        )

        # Cross entropy expects integer class IDs.
        target_wdl = torch.as_tensor(
            target_wdl_np,
            dtype=torch.long,
            device=device,
        )

        # ----------------------------------------------------
        # FORWARD
        # ----------------------------------------------------

        (
            policy_logits,
            wdl_logits,
        ) = model(
            observations
        )

        # ----------------------------------------------------
        # LOSS
        # ----------------------------------------------------

        (
            loss,
            policy_loss,
            wdl_loss,
            policy_kl,
        ) = policy_wdl_loss(
            policy_logits,
            wdl_logits,
            target_policies,
            target_wdl,
        )

        # ----------------------------------------------------
        # BACKWARD
        # ----------------------------------------------------

        optimizer.zero_grad()

        loss.backward()

        grad_norm = (
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=float(
                    "inf"
                ),
            )
        )

        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        # ----------------------------------------------------
        # DIAGNOSTICS
        # ----------------------------------------------------

        with torch.no_grad():

            wdl_probabilities = torch.softmax(
                wdl_logits,
                dim=-1,
            )

            predicted_values = (
                wdl_probabilities[
                    :,
                    WDL_WIN
                ]
                - wdl_probabilities[
                    :,
                    WDL_LOSS
                ]
            )

            target_values = (
                wdl_targets_to_scalar(
                    target_wdl
                )
            )

            predicted_value_mean = (
                predicted_values
                .mean()
                .item()
            )

            target_value_mean = (
                target_values
                .mean()
                .item()
            )

            wdl_accuracy = (
                wdl_logits.argmax(
                    dim=-1
                )
                == target_wdl
            ).float().mean().item()

            average_loss_probability = (
                wdl_probabilities[
                    :,
                    WDL_LOSS
                ]
                .mean()
                .item()
            )

            average_draw_probability = (
                wdl_probabilities[
                    :,
                    WDL_DRAW
                ]
                .mean()
                .item()
            )

            average_win_probability = (
                wdl_probabilities[
                    :,
                    WDL_WIN
                ]
                .mean()
                .item()
            )

            target_loss_fraction = (
                target_wdl
                .eq(
                    WDL_LOSS
                )
                .float()
                .mean()
                .item()
            )

            target_draw_fraction = (
                target_wdl
                .eq(
                    WDL_DRAW
                )
                .float()
                .mean()
                .item()
            )

            target_win_fraction = (
                target_wdl
                .eq(
                    WDL_WIN
                )
                .float()
                .mean()
                .item()
            )

            target_policy_entropy = -(
                target_policies
                * torch.log(
                    target_policies
                    .clamp_min(
                        1e-8
                    )
                )
            ).sum(
                dim=-1
            ).mean().item()

        iteration_predicted_value_means.append(
            predicted_value_mean
        )

        iteration_target_value_means.append(
            target_value_mean
        )

        iteration_target_policy_entropies.append(
            target_policy_entropy
        )

        iteration_wdl_accuracies.append(
            wdl_accuracy
        )

        iteration_loss_probabilities.append(
            average_loss_probability
        )

        iteration_draw_probabilities.append(
            average_draw_probability
        )

        iteration_win_probabilities.append(
            average_win_probability
        )

        iteration_target_loss_fractions.append(
            target_loss_fraction
        )

        iteration_target_draw_fractions.append(
            target_draw_fraction
        )

        iteration_target_win_fractions.append(
            target_win_fraction
        )

        iteration_grad_norms.append(
            grad_norm.item()
        )

        iteration_total_losses.append(
            loss.item()
        )

        iteration_policy_losses.append(
            policy_loss.item()
        )

        iteration_wdl_losses.append(
            wdl_loss.item()
        )

        iteration_policy_kls.append(
            policy_kl.item()
        )

    # ========================================================
    # RESULTS
    # ========================================================

    return {

        "average_total_loss":
            float(
                np.mean(
                    iteration_total_losses
                )
            ),

        "average_policy_loss":
            float(
                np.mean(
                    iteration_policy_losses
                )
            ),

        "average_wdl_loss":
            float(
                np.mean(
                    iteration_wdl_losses
                )
            ),

        "average_policy_kl":
            float(
                np.mean(
                    iteration_policy_kls
                )
            ),

        "average_grad_norm":
            float(
                np.mean(
                    iteration_grad_norms
                )
            ),

        "average_predicted_value":
            float(
                np.mean(
                    iteration_predicted_value_means
                )
            ),

        "average_target_value":
            float(
                np.mean(
                    iteration_target_value_means
                )
            ),

        "average_target_policy_entropy":
            float(
                np.mean(
                    iteration_target_policy_entropies
                )
            ),

        "average_wdl_accuracy":
            float(
                np.mean(
                    iteration_wdl_accuracies
                )
            ),

        "average_loss_probability":
            float(
                np.mean(
                    iteration_loss_probabilities
                )
            ),

        "average_draw_probability":
            float(
                np.mean(
                    iteration_draw_probabilities
                )
            ),

        "average_win_probability":
            float(
                np.mean(
                    iteration_win_probabilities
                )
            ),

        "average_target_loss_fraction":
            float(
                np.mean(
                    iteration_target_loss_fractions
                )
            ),

        "average_target_draw_fraction":
            float(
                np.mean(
                    iteration_target_draw_fractions
                )
            ),

        "average_target_win_fraction":
            float(
                np.mean(
                    iteration_target_win_fractions
                )
            ),
    }


# ============================================================
# REPLAY SPLIT STATS
# ============================================================

def get_split_stats(
    replay_buffer,
    split,
):
    """
    Count games and CURRENTLY SURVIVING positions
    in a replay split.

    Uses game_sample_counts so the numbers remain
    correct after the circular replay buffer begins
    overwriting old positions.
    """

    num_games = 0
    num_positions = 0

    for (
        game_id,
        game
    ) in replay_buffer.games.items():

        if (
            game.get("split")
            != split
        ):
            continue

        surviving_positions = int(
            replay_buffer
            .game_sample_counts
            .get(
                game_id,
                0,
            )
        )

        if surviving_positions <= 0:
            continue

        num_games += 1

        num_positions += (
            surviving_positions
        )

    return {
        "games":
            num_games,

        "positions":
            num_positions,
    }

# ============================================================
# VALIDATE NETWORK
# ============================================================

def validate_network(
    model,
    replay_buffer,
    batch_size,
    validation_steps=10,
    split="val",
):
    """
    Evaluate Model 3 on held-out replay positions.

    Uses the same target construction as training:

        raw MCTS visits -> policy target
        winner IDs      -> WDL target
    """

    if validation_steps <= 0:
        return None

    device = next(
        model.parameters()
    ).device

    was_training = (
        model.training
    )

    model.eval()

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    total_losses = []
    policy_losses = []
    wdl_losses = []
    policy_kls = []

    predicted_value_means = []
    target_value_means = []
    target_policy_entropies = []

    wdl_accuracies = []
    loss_probabilities = []
    draw_probabilities = []
    win_probabilities = []

    target_loss_fractions = []
    target_draw_fractions = []
    target_win_fractions = []

    batches_evaluated = 0

    # ========================================================
    # VALIDATION
    # ========================================================

    with torch.no_grad():

        for _ in range(
            validation_steps
        ):

            batch = replay_buffer.sample(
                batch_size,
                split=split,
            )

            if not batch:
                break

            (
                observations_np,
                target_policies_np,
                target_wdl_np,
            ) = build_training_batch(
                batch=batch,
                replay_buffer=replay_buffer,
            )

            observations = torch.as_tensor(
                observations_np,
                dtype=torch.float32,
                device=device,
            )

            target_policies = torch.as_tensor(
                target_policies_np,
                dtype=torch.float32,
                device=device,
            )

            target_wdl = torch.as_tensor(
                target_wdl_np,
                dtype=torch.long,
                device=device,
            )

            (
                policy_logits,
                wdl_logits,
            ) = model(
                observations
            )

            (
                loss,
                policy_loss,
                wdl_loss,
                policy_kl,
            ) = policy_wdl_loss(
                policy_logits,
                wdl_logits,
                target_policies,
                target_wdl,
            )

            # ------------------------------------------------
            # DIAGNOSTICS
            # ------------------------------------------------

            wdl_probabilities = torch.softmax(
                wdl_logits,
                dim=-1,
            )

            predicted_values = (
                wdl_probabilities[
                    :,
                    WDL_WIN
                ]
                - wdl_probabilities[
                    :,
                    WDL_LOSS
                ]
            )

            target_values = (
                wdl_targets_to_scalar(
                    target_wdl
                )
            )

            predicted_value_mean = (
                predicted_values
                .mean()
                .item()
            )

            target_value_mean = (
                target_values
                .mean()
                .item()
            )

            wdl_accuracy = (
                wdl_logits.argmax(
                    dim=-1
                )
                == target_wdl
            ).float().mean().item()

            average_loss_probability = (
                wdl_probabilities[
                    :,
                    WDL_LOSS
                ]
                .mean()
                .item()
            )

            average_draw_probability = (
                wdl_probabilities[
                    :,
                    WDL_DRAW
                ]
                .mean()
                .item()
            )

            average_win_probability = (
                wdl_probabilities[
                    :,
                    WDL_WIN
                ]
                .mean()
                .item()
            )

            target_loss_fraction = (
                target_wdl
                .eq(
                    WDL_LOSS
                )
                .float()
                .mean()
                .item()
            )

            target_draw_fraction = (
                target_wdl
                .eq(
                    WDL_DRAW
                )
                .float()
                .mean()
                .item()
            )

            target_win_fraction = (
                target_wdl
                .eq(
                    WDL_WIN
                )
                .float()
                .mean()
                .item()
            )

            target_policy_entropy = -(
                target_policies
                * torch.log(
                    target_policies
                    .clamp_min(
                        1e-8
                    )
                )
            ).sum(
                dim=-1
            ).mean().item()

            # ------------------------------------------------
            # RECORD
            # ------------------------------------------------

            total_losses.append(
                loss.item()
            )

            policy_losses.append(
                policy_loss.item()
            )

            wdl_losses.append(
                wdl_loss.item()
            )

            policy_kls.append(
                policy_kl.item()
            )

            predicted_value_means.append(
                predicted_value_mean
            )

            target_value_means.append(
                target_value_mean
            )

            target_policy_entropies.append(
                target_policy_entropy
            )

            wdl_accuracies.append(
                wdl_accuracy
            )

            loss_probabilities.append(
                average_loss_probability
            )

            draw_probabilities.append(
                average_draw_probability
            )

            win_probabilities.append(
                average_win_probability
            )

            target_loss_fractions.append(
                target_loss_fraction
            )

            target_draw_fractions.append(
                target_draw_fraction
            )

            target_win_fractions.append(
                target_win_fraction
            )

            batches_evaluated += 1

    if was_training:
        model.train()

    if not total_losses:
        return None

    return {

        "average_total_loss":
            float(
                np.mean(
                    total_losses
                )
            ),

        "average_policy_loss":
            float(
                np.mean(
                    policy_losses
                )
            ),

        "average_wdl_loss":
            float(
                np.mean(
                    wdl_losses
                )
            ),

        "average_policy_kl":
            float(
                np.mean(
                    policy_kls
                )
            ),

        "average_predicted_value":
            float(
                np.mean(
                    predicted_value_means
                )
            ),

        "average_target_value":
            float(
                np.mean(
                    target_value_means
                )
            ),

        "average_target_policy_entropy":
            float(
                np.mean(
                    target_policy_entropies
                )
            ),

        "average_wdl_accuracy":
            float(
                np.mean(
                    wdl_accuracies
                )
            ),

        "average_loss_probability":
            float(
                np.mean(
                    loss_probabilities
                )
            ),

        "average_draw_probability":
            float(
                np.mean(
                    draw_probabilities
                )
            ),

        "average_win_probability":
            float(
                np.mean(
                    win_probabilities
                )
            ),

        "average_target_loss_fraction":
            float(
                np.mean(
                    target_loss_fractions
                )
            ),

        "average_target_draw_fraction":
            float(
                np.mean(
                    target_draw_fractions
                )
            ),

        "average_target_win_fraction":
            float(
                np.mean(
                    target_win_fractions
                )
            ),

        "batches_evaluated":
            int(
                batches_evaluated
            ),
    }


# ============================================================
# RUN TRAINING
# ============================================================

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
    starting_games_attempted=None,

    seed=None,
    dynamic_seeding=False,

    teacher_mode=False,

    writer=None,
    scheduler=None,

    # --------------------------------------------------------
    # SELF-PLAY SETTINGS
    # --------------------------------------------------------

    temperature=1.0,
    temperature_fn=None,

    add_root_noise=True,
    root_noise_fn=None,

    model_generation=3,
    action_space_version=1,
    model_name="Model3",
    run_start_checkpoint=None,
    # Optional extra MCTS constructor arguments.
    mcts_kwargs=None,

    # --------------------------------------------------------
    # TRAIN / VALIDATION SPLIT
    # --------------------------------------------------------

    validation_fraction=0.10,
    split_seed=20260917,

    train_sample_split="train",
    val_sample_split="val",

    # Don't validate until enough held-out
    # data exists to be useful.
    min_validation_games=5,
    min_validation_positions=256,

    validation_steps=10,
    validation_batch_size=None,

    # --------------------------------------------------------
    # REPLAY SAVE
    # --------------------------------------------------------

    replay_buffer_path=None,
):

    history = []

    if validation_batch_size is None:

        validation_batch_size = (
            batch_size
        )

    if min_validation_games < 1:

        raise ValueError(
            "min_validation_games must be >= 1."
        )

    if min_validation_positions < 1:

        raise ValueError(
            "min_validation_positions must be >= 1."
        )

    if validation_steps < 1:

        raise ValueError(
            "validation_steps must be >= 1."
        )


    games_played = int(
        starting_games_played
    )

    if starting_games_attempted is None:

        games_attempted = (
            games_played
        )

    else:

        games_attempted = int(
            starting_games_attempted
        )



    # ========================================================
    # CHECKPOINT SCHEDULE
    # ========================================================

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

    # ========================================================
    # ITERATIONS
    # ========================================================

    for iteration in range(
        num_iterations
    ):

        iteration_start = (
            time.perf_counter()
        )

        positions_added = 0

        iteration_game_lengths = []

        total_self_play_time = 0.0

        successful_games_this_iteration = 0

        # ----------------------------------------------------
        # Split diagnostics for this iteration
        # ----------------------------------------------------

        train_games_this_iteration = 0
        val_games_this_iteration = 0

        train_positions_this_iteration = 0
        val_positions_this_iteration = 0

        # ----------------------------------------------------
        # This identifies the model that generated this batch.
        #
        # The model stays fixed during self-play for this
        # iteration. Training happens afterward.
        # ----------------------------------------------------

        generating_model_self_play_games = int(
            games_played
        )

        generating_model_id = (
            f"{model_name}_"
            f"{generating_model_self_play_games}_games"
        )

        # ====================================================
        # SELF-PLAY MODEL MODE
        # ====================================================

        # Self-play is inference only.
        #
        # neural_evaluate() already uses
        # torch.inference_mode(), while model.eval()
        # ensures layers such as dropout / batch norm
        # use evaluation behavior.
        model.eval()


        # ====================================================
        # MCTS
        # ====================================================

        current_mcts_kwargs = {

            "simulations":
                simulations,

            "rollout_type":
                "neural",

            "selection_type":
                "puct",

            "model":
                model,
        }


        if mcts_kwargs:

            current_mcts_kwargs.update(
                mcts_kwargs
            )

        mcts = MCTS(
            **current_mcts_kwargs
        )

        # ====================================================
        # MODEL REPLAY GENERATOR
        # ====================================================

        replay_generator = (
            ModelReplayGenerator(
                env=env,
                mcts=mcts,
                replay_buffer=replay_buffer,
                state_serializer=serialize_state,

                temperature=temperature,
                temperature_fn=temperature_fn,

                add_root_noise=add_root_noise,
                root_noise_fn=root_noise_fn,

                teacher_mode=teacher_mode,

                action_space_version=(
                    action_space_version
                ),
            )
        )

        # ====================================================
        # SELF PLAY
        # ====================================================

        successful_games_this_iteration = 0

        while (
            successful_games_this_iteration
            < self_play_games_per_iteration
        ):

            game_index = games_attempted

            game_seed = resolve_game_seed(
                base_seed=seed,
                game_index=game_index,
                dynamic_seeding=dynamic_seeding,
            )

            game_split = choose_game_split(
                seed=game_seed,
                game_index=game_index,
                validation_fraction=validation_fraction,
                split_seed=split_seed,
            )

            games_attempted += 1

            game_start = (
                time.perf_counter()
            )


            try:

                result = replay_generator.generate_game(
                    seed=game_seed,
                    split=game_split,
                    model_generation=model_generation,
                    model_checkpoint=None,

                    extra_game_metadata={
                        "game_index":
                            game_index,

                        "validation_fraction":
                            validation_fraction,

                        "split_seed":
                            split_seed,

                        "model_name":
                            model_name,

                        "model_id":
                            generating_model_id,

                        "self_play_games_at_generation":
                            generating_model_self_play_games,

                        "run_start_checkpoint":
                            run_start_checkpoint,

                        "self_play_games_at_generation":
                            generating_model_self_play_games,
                        
                    },
                )

            except Exception as error:

                print()
                print("=" * 70)
                print(
                    f"SELF-PLAY GAME FAILED "
                    f"(attempt {game_index})"
                )
                print(
                    f"{type(error).__name__}: {error}"
                )
                traceback.print_exc()
                print("=" * 70)
                print()

                continue


            # =================================================
            # SUCCESSFUL GAME
            # =================================================

            game_time = (
                time.perf_counter()
                - game_start
            )

            total_self_play_time += (
                game_time
            )

            game_positions = int(
                result[
                    "num_positions"
                ]
            )

            iteration_game_lengths.append(
                game_positions
            )

            positions_added += (
                game_positions
            )

            games_played += 1

            successful_games_this_iteration += 1

            # ------------------------------------------------
            # SPLIT COUNTS
            # ------------------------------------------------

            if game_split == "train":

                train_games_this_iteration += 1

                train_positions_this_iteration += (
                    game_positions
                )

            elif game_split == "val":

                val_games_this_iteration += 1

                val_positions_this_iteration += (
                    game_positions
                )

            else:

                raise RuntimeError(
                    "Unexpected replay split: "
                    f"{game_split}"
                )

            # ------------------------------------------------
            # GAME SUMMARY
            # ------------------------------------------------

            print(
                f"Game {games_played} complete "
                f"- split={game_split} "
                f"- {game_positions} positions "
                f"- replay size "
                f"{len(replay_buffer):,}"
            )

        # ====================================================
        # CURRENT REPLAY SPLIT STATS
        # ====================================================

        train_stats = get_split_stats(
            replay_buffer=replay_buffer,
            split=train_sample_split,
        )

        val_stats = get_split_stats(
            replay_buffer=replay_buffer,
            split=val_sample_split,
        )


        # ====================================================
        # TRAINING STEPS
        # ====================================================

        # Important:
        #
        # Training ratio should be based only on positions
        # actually assigned to the TRAIN split.
        #
        # Validation positions should not increase the
        # number of optimizer steps.
        # ====================================================

        if (
            train_positions_this_iteration
            > 0
        ):

            training_steps = max(
                1,
                round(
                    train_positions_this_iteration
                    * training_ratio
                    / batch_size
                ),
            )

        else:

            training_steps = 0

        print(
            "Number of Training Steps:",
            training_steps,
        )

        # ====================================================
        # TRAIN NETWORK
        # ====================================================

        training_results = None
        validation_results = None

        if training_steps > 0:

            training_results = (
                train_network(
                    model=model,

                    replay_buffer=(
                        replay_buffer
                    ),

                    optimizer=optimizer,

                    batch_size=batch_size,

                    training_steps=(
                        training_steps
                    ),

                    scheduler=scheduler,

                    split=(
                        train_sample_split
                    ),
                )
            )

            history.append(
                training_results
            )


        # ====================================================
        # VALIDATION
        #
        # Keep this OUTSIDE:
        #
        #     if training_steps > 0
        #
        # Validation should be safe regardless of whether
        # this particular iteration performed an update.
        # ====================================================

        enough_validation_data = (
            val_stats["games"]
            >= min_validation_games
            and
            val_stats["positions"]
            >= min_validation_positions
        )

        if enough_validation_data:

            validation_results = (
                validate_network(
                    model=model,

                    replay_buffer=replay_buffer,

                    batch_size=(
                        validation_batch_size
                    ),

                    validation_steps=(
                        validation_steps
                    ),

                    split=(
                        val_sample_split
                    ),
                )
            )

            if validation_results is not None:

                print(
                    "Validation loss:",
                    f"{validation_results['average_total_loss']:.4f}",
                )

                print(
                    "Validation policy loss:",
                    f"{validation_results['average_policy_loss']:.4f}",
                )

                print(
                    "Validation WDL loss:",
                    f"{validation_results['average_wdl_loss']:.4f}",
                )

                print(
                    "Validation WDL accuracy:",
                    f"{validation_results['average_wdl_accuracy']:.4f}",
                )

        else:

            print(
                "Validation skipped "
                f"- {val_stats['games']} / "
                f"{min_validation_games} games "
                f"- {val_stats['positions']} / "
                f"{min_validation_positions} positions"
            )

        # ====================================================
        # TIMING
        # ====================================================

        iteration_time = (
            time.perf_counter()
            - iteration_start
        )

        if iteration_game_lengths:

            average_game_length = float(
                np.mean(
                    iteration_game_lengths
                )
            )

        else:

            average_game_length = 0.0

        # ====================================================
        # TENSORBOARD
        # ====================================================

        if writer is not None:

            writer.add_scalar(
                "Replay/train_games",
                train_stats["games"],
                games_played,
            )

            writer.add_scalar(
                "Replay/train_positions",
                train_stats["positions"],
                games_played,
            )

            writer.add_scalar(
                "Replay/val_games",
                val_stats["games"],
                games_played,
            )

            writer.add_scalar(
                "Replay/val_positions",
                val_stats["positions"],
                games_played,
            )

            writer.add_scalar(
                "SelfPlay/average_game_length",
                average_game_length,
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
                "Training/train_positions_added",
                train_positions_this_iteration,
                games_played,
            )

            writer.add_scalar(
                "Validation/positions_added",
                val_positions_this_iteration,
                games_played,
            )

            writer.add_scalar(
                "Training/train_games_added",
                train_games_this_iteration,
                games_played,
            )

            writer.add_scalar(
                "Validation/games_added",
                val_games_this_iteration,
                games_played,
            )

            writer.add_scalar(
                "Training/training_steps",
                training_steps,
                games_played,
            )

            writer.add_scalar(
                "Training/learning_rate",
                optimizer.param_groups[
                    0
                ]["lr"],
                games_played,
            )

            writer.add_scalar(
                "Timing/self_play_seconds",
                total_self_play_time,
                games_played,
            )

            writer.add_scalar(
                "Timing/iteration_seconds",
                iteration_time,
                games_played,
            )

            if positions_added > 0:

                writer.add_scalar(
                    "Timing/self_play_seconds_per_position",
                    (
                        total_self_play_time
                        / positions_added
                    ),
                    games_played,
                )

            # ------------------------------------------------
            # TRAINING METRICS
            # ------------------------------------------------

            if training_results is not None:

                writer.add_scalar(
                    "Loss/total",
                    training_results[
                        "average_total_loss"
                    ],
                    games_played,
                )

                writer.add_scalar(
                    "Loss/policy",
                    training_results[
                        "average_policy_loss"
                    ],
                    games_played,
                )

                writer.add_scalar(
                    "Loss/wdl",
                    training_results[
                        "average_wdl_loss"
                    ],
                    games_played,
                )

                writer.add_scalar(
                    "Loss/policy_kl",
                    training_results[
                        "average_policy_kl"
                    ],
                    games_played,
                )

                writer.add_scalar(
                    "Training/gradient_norm",
                    training_results[
                        "average_grad_norm"
                    ],
                    games_played,
                )

                writer.add_scalar(
                    "Training/predicted_value",
                    training_results[
                        "average_predicted_value"
                    ],
                    games_played,
                )

                writer.add_scalar(
                    "Training/target_value",
                    training_results[
                        "average_target_value"
                    ],
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
                    "Training/wdl_accuracy",
                    training_results[
                        "average_wdl_accuracy"
                    ],
                    games_played,
                )

                writer.add_scalar(
                    "Training/wdl_loss_probability",
                    training_results[
                        "average_loss_probability"
                    ],
                    games_played,
                )

                writer.add_scalar(
                    "Training/wdl_draw_probability",
                    training_results[
                        "average_draw_probability"
                    ],
                    games_played,
                )

                writer.add_scalar(
                    "Training/wdl_win_probability",
                    training_results[
                        "average_win_probability"
                    ],
                    games_played,
                )

                writer.add_scalar(
                    "Training/target_loss_fraction",
                    training_results[
                        "average_target_loss_fraction"
                    ],
                    games_played,
                )

                writer.add_scalar(
                    "Training/target_draw_fraction",
                    training_results[
                        "average_target_draw_fraction"
                    ],
                    games_played,
                )

                writer.add_scalar(
                    "Training/target_win_fraction",
                    training_results[
                        "average_target_win_fraction"
                    ],
                    games_played,
                )

            # ------------------------------------------------
            # VALIDATION METRICS
            #
            # Keep this independent of training_results. Validation
            # may legitimately run on an iteration with zero updates.
            # ------------------------------------------------

            if validation_results is not None:

                writer.add_scalar(
                    "Validation/loss_total",
                    validation_results[
                        "average_total_loss"
                    ],
                    games_played,
                )

                writer.add_scalar(
                    "Validation/loss_policy",
                    validation_results[
                        "average_policy_loss"
                    ],
                    games_played,
                )

                writer.add_scalar(
                    "Validation/loss_wdl",
                    validation_results[
                        "average_wdl_loss"
                    ],
                    games_played,
                )

                writer.add_scalar(
                    "Validation/policy_kl",
                    validation_results[
                        "average_policy_kl"
                    ],
                    games_played,
                )

                writer.add_scalar(
                    "Validation/predicted_value",
                    validation_results[
                        "average_predicted_value"
                    ],
                    games_played,
                )

                writer.add_scalar(
                    "Validation/target_value",
                    validation_results[
                        "average_target_value"
                    ],
                    games_played,
                )

                writer.add_scalar(
                    "Validation/wdl_accuracy",
                    validation_results[
                        "average_wdl_accuracy"
                    ],
                    games_played,
                )

                writer.add_scalar(
                    "Validation/wdl_loss_probability",
                    validation_results[
                        "average_loss_probability"
                    ],
                    games_played,
                )

                writer.add_scalar(
                    "Validation/wdl_draw_probability",
                    validation_results[
                        "average_draw_probability"
                    ],
                    games_played,
                )

                writer.add_scalar(
                    "Validation/wdl_win_probability",
                    validation_results[
                        "average_win_probability"
                    ],
                    games_played,
                )

                writer.add_scalar(
                    "Validation/target_loss_fraction",
                    validation_results[
                        "average_target_loss_fraction"
                    ],
                    games_played,
                )

                writer.add_scalar(
                    "Validation/target_draw_fraction",
                    validation_results[
                        "average_target_draw_fraction"
                    ],
                    games_played,
                )

                writer.add_scalar(
                    "Validation/target_win_fraction",
                    validation_results[
                        "average_target_win_fraction"
                    ],
                    games_played,
                )

        # ====================================================
        # CHECKPOINT
        # ====================================================

        if (
            checkpoint_every_games
            is not None
        ):

            (
                next_checkpoint,
                checkpoint_saved,
            ) = save_model_if_needed(
                model=model,

                optimizer=optimizer,

                games_played=games_played,

                history=history,

                checkpoint_every_games=(
                    checkpoint_every_games
                ),

                next_checkpoint=(
                    next_checkpoint
                ),

                checkpoint_dir=(
                    checkpoint_dir
                ),

                replay_buffer=(
                    replay_buffer
                ),

                scheduler=scheduler,
            )

            if (
                checkpoint_saved
                and replay_buffer_path
                is not None
            ):

                replay_buffer.save(
                    replay_buffer_path
                )

        # ====================================================
        # ITERATION SUMMARY
        # ====================================================

        print()

        print(
            "=" * 70
        )

        print(
            f"Iteration {iteration + 1}"
        )

        print(
            f"Successful games this iteration: "
            f"{successful_games_this_iteration}"
        )

        print(
            f"Train games: "
            f"{train_games_this_iteration}"
        )

        print(
            f"Validation games: "
            f"{val_games_this_iteration}"
        )

        print(
            f"Train positions: "
            f"{train_positions_this_iteration}"
        )

        print(
            f"Validation positions: "
            f"{val_positions_this_iteration}"
        )

        print(
            f"Self-play time: "
            f"{total_self_play_time:.2f}s"
        )

        print(
            f"Iteration time: "
            f"{iteration_time:.2f}s"
        )

        print(
            "Games played:",
            games_played,
        )

        print(
            "Games attempted:",
            games_attempted,
        )

        print(
            "Replay size:",
            len(
                replay_buffer
            ),
        )

        print(
            "Replay position:",
            replay_buffer.position,
        )

        print(
            "Positions added:",
            positions_added,
        )

        print(
            "Average game length:",
            f"{average_game_length:.2f}",
        )

        print(
            "Learning rate:",
            optimizer.param_groups[
                0
            ]["lr"],
        )

        print(
            "=" * 70
        )

        print()

        print(
            "Replay train games:",
            train_stats["games"],
        )

        print(
            "Replay train positions:",
            train_stats["positions"],
        )

        print(
            "Replay validation games:",
            val_stats["games"],
        )

        print(
            "Replay validation positions:",
            val_stats["positions"],
        )

        if validation_results is not None:

            print(
                "Validation total loss:",
                f"{validation_results['average_total_loss']:.4f}",
            )

            print(
                "Validation policy loss:",
                f"{validation_results['average_policy_loss']:.4f}",
            )

            print(
                "Validation WDL loss:",
                f"{validation_results['average_wdl_loss']:.4f}",
            )

            print(
                "Validation WDL accuracy:",
                f"{validation_results['average_wdl_accuracy']:.4f}",
            )

    # ========================================================
    # FINAL MODEL CHECKPOINT
    # ========================================================

    os.makedirs(
        checkpoint_dir,
        exist_ok=True,
    )

    final_checkpoint_path = (
        f"{checkpoint_dir}/"
        f"model_{games_played}_games_last.pt"
    )

    save_checkpoint(
        path=final_checkpoint_path,

        model=model,

        optimizer=optimizer,

        games_played=games_played,

        history=history,
    )

    # ========================================================
    # FINAL REPLAY SAVE
    # ========================================================

    if replay_buffer_path is not None:

        replay_buffer.save(
            replay_buffer_path
        )

        print(
            "Final replay buffer saved:",
            replay_buffer_path,
        )

    print(
        f"Final checkpoint saved: "
        f"{games_played} games"
    )

    return history