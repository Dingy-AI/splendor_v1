import os

import time

import traceback

import random



import numpy as np

import torch



from splendor_v1.mcts.mcts_v5_pruning import (

    MCTS,

)



from splendor_v1.training.checkpoint import (

    save_model_if_needed,

    save_checkpoint,

)



from splendor_v1.training_v2.state_serializer import (

    serialize_state,

)



from splendor_v1.training_v4.batch_builder_v4 import (

    collate_model4_batch,

    wdl_target_from_winners,

)



from splendor_v1.network.losses_4_wdl import (

    policy_wdl_loss,

    WDL_LOSS,

    WDL_DRAW,

    WDL_WIN,

)



from splendor_v1.training_v5.model_replay_generator_v5_pruning import (

    ModelReplayGenerator,

)





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



    dynamic_seeding=True:

        game N uses base_seed + N.



    dynamic_seeding=False:

        every game uses base_seed.



    For Model 4, dynamic seeding is recommended and is the

    run_training default.

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

    Deterministically assign an ENTIRE game to train or validation.



    This avoids position-level leakage between train and validation.

    """



    if not (

        0.0

        <= validation_fraction

        <= 1.0

    ):



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

# RAW REPLAY -> MODEL 4 BATCH

# ============================================================





def build_training_batch(

    batch,

    replay_buffer,

):

    """

    Convert raw rich-replay samples returned by ReplayBuffer.sample()

    into the dynamically padded Model 4 batch format.



    Model 4 trains only over the legal candidate actions represented

    by each MCTS root:



        observations        [B, 258]

        legal_action_ids    [B, N]

        legal_action_mask   [B, N]

        target_policy       [B, N]

        target_wdl          [B]



    N is dynamic: the largest legal-action count in this batch.

    """



    if not batch:



        raise ValueError(

            "Cannot build a Model 4 batch from an empty sample list."

        )



    items = []



    for sample_index, sample in enumerate(

        batch

    ):



        if not isinstance(

            sample,

            dict,

        ):



            raise TypeError(

                "Model 4 rich replay samples must be dictionaries."

            )



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

                "Replay metadata missing for "

                f"game_id={game_id}."

            )



        if "winner_ids" not in game:



            raise RuntimeError(

                "Replay game metadata is missing winner_ids."

            )



        if "current_player" not in sample:



            raise RuntimeError(

                "Replay sample is missing current_player."

            )



        target_wdl = (

            wdl_target_from_winners(

                current_player=sample[

                    "current_player"

                ],

                winner_ids=game[

                    "winner_ids"

                ],

            )

        )



        items.append(

            {

                "sample":

                    sample,



                # ReplayBuffer.sample() does not expose the original

                # circular-buffer index. The collator retains this

                # only for diagnostics, so the local batch index is

                # sufficient here.

                "sample_index":

                    int(

                        sample_index

                    ),



                "game":

                    game,



                "target_wdl":

                    int(

                        target_wdl

                    ),

            }

        )



    return collate_model4_batch(

        items

    )





# ============================================================

# DEVICE

# ============================================================





def move_batch_to_device(

    batch,

    device,

):

    return {

        "observations":

            batch[

                "observations"

            ].to(

                device,

                non_blocking=True,

            ),



        "legal_action_ids":

            batch[

                "legal_action_ids"

            ].to(

                device,

                non_blocking=True,

            ),



        "legal_action_mask":

            batch[

                "legal_action_mask"

            ].to(

                device,

                non_blocking=True,

            ),



        "target_policy":

            batch[

                "target_policy"

            ].to(

                device,

                non_blocking=True,

            ),



        "target_wdl":

            batch[

                "target_wdl"

            ].to(

                device,

                non_blocking=True,

            ),



        "action_counts":

            batch[

                "action_counts"

            ].to(

                device,

                non_blocking=True,

            ),

    }





# ============================================================

# WDL DIAGNOSTICS

# ============================================================





def wdl_targets_to_scalar(

    target_wdl,

):

    """

    Diagnostics only:



        LOSS -> -1

        DRAW ->  0

        WIN  -> +1

    """



    values = torch.zeros_like(

        target_wdl,

        dtype=torch.float32,

    )



    values = torch.where(

        target_wdl == WDL_WIN,

        torch.ones_like(

            values

        ),

        values,

    )



    values = torch.where(

        target_wdl == WDL_LOSS,

        -torch.ones_like(

            values

        ),

        values,

    )



    return values





def collect_batch_diagnostics(

    policy_logits,

    wdl_logits,

    target_policy,

    target_wdl,

    action_counts,

):

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



        target_log_probs = torch.log(

            target_policy.clamp_min(

                1e-8

            )

        )



        target_policy_entropy = -(

            target_policy

            * target_log_probs

        ).sum(

            dim=-1

        ).mean().item()



        return {

            "predicted_value_mean":

                predicted_values.mean().item(),



            "target_value_mean":

                target_values.mean().item(),



            "target_policy_entropy":

                target_policy_entropy,



            "wdl_accuracy":

                (

                    wdl_logits.argmax(

                        dim=-1

                    )

                    == target_wdl

                ).float().mean().item(),



            "policy_top1":

                (

                    policy_logits.argmax(

                        dim=-1

                    )

                    == target_policy.argmax(

                        dim=-1

                    )

                ).float().mean().item(),



            "loss_probability":

                wdl_probabilities[

                    :,

                    WDL_LOSS

                ].mean().item(),



            "draw_probability":

                wdl_probabilities[

                    :,

                    WDL_DRAW

                ].mean().item(),



            "win_probability":

                wdl_probabilities[

                    :,

                    WDL_WIN

                ].mean().item(),



            "target_loss_fraction":

                target_wdl.eq(

                    WDL_LOSS

                ).float().mean().item(),



            "target_draw_fraction":

                target_wdl.eq(

                    WDL_DRAW

                ).float().mean().item(),



            "target_win_fraction":

                target_wdl.eq(

                    WDL_WIN

                ).float().mean().item(),



            "mean_legal_actions":

                action_counts.float().mean().item(),

        }





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

    grad_clip=1.0,

):

    """

    Train Model 4 from rich replay.



    Policy:

        MCTS visit distribution over actual legal candidates.



    Value:

        WDL class from the sample current-player perspective.

    """



    if len(

        replay_buffer

    ) == 0:



        raise ValueError(

            "Replay buffer is empty."

        )



    if training_steps <= 0:



        raise ValueError(

            "training_steps must be positive."

        )



    model.train()



    device = next(

        model.parameters()

    ).device



    metrics = {

        "total_loss": [],

        "policy_loss": [],

        "wdl_loss": [],

        "policy_kl": [],

        "grad_norm": [],

        "predicted_value": [],

        "target_value": [],

        "target_policy_entropy": [],

        "wdl_accuracy": [],

        "policy_top1": [],

        "loss_probability": [],

        "draw_probability": [],

        "win_probability": [],

        "target_loss_fraction": [],

        "target_draw_fraction": [],

        "target_win_fraction": [],

        "mean_legal_actions": [],

    }



    for _ in range(

        training_steps

    ):



        if split is None:



            raw_batch = replay_buffer.sample(

                batch_size

            )



        else:



            raw_batch = replay_buffer.sample(

                batch_size,

                split=split,

            )



        if not raw_batch:



            raise RuntimeError(

                "ReplayBuffer returned an empty "

                "training batch."

            )



        batch = build_training_batch(

            batch=raw_batch,

            replay_buffer=replay_buffer,

        )



        batch = move_batch_to_device(

            batch,

            device,

        )



        optimizer.zero_grad(

            set_to_none=True

        )



        (

            policy_logits,

            wdl_logits,

        ) = model(

            batch[

                "observations"

            ],

            batch[

                "legal_action_ids"

            ],

            batch[

                "legal_action_mask"

            ],

        )



        (

            loss,

            policy_loss,

            wdl_loss,

            policy_kl,

        ) = policy_wdl_loss(

            policy_logits=policy_logits,

            wdl_logits=wdl_logits,

            target_policy=batch[

                "target_policy"

            ],

            target_wdl=batch[

                "target_wdl"

            ],

            legal_action_mask=batch[

                "legal_action_mask"

            ],

        )



        loss.backward()



        if grad_clip is None:



            grad_norm = (

                torch.nn.utils.clip_grad_norm_(

                    model.parameters(),

                    max_norm=float(

                        "inf"

                    ),

                )

            )



        else:



            grad_norm = (

                torch.nn.utils.clip_grad_norm_(

                    model.parameters(),

                    max_norm=float(

                        grad_clip

                    ),

                )

            )



        optimizer.step()



        if scheduler is not None:

            scheduler.step()



        diagnostics = (

            collect_batch_diagnostics(

                policy_logits=policy_logits,

                wdl_logits=wdl_logits,

                target_policy=batch[

                    "target_policy"

                ],

                target_wdl=batch[

                    "target_wdl"

                ],

                action_counts=batch[

                    "action_counts"

                ],

            )

        )



        metrics[

            "total_loss"

        ].append(

            loss.item()

        )



        metrics[

            "policy_loss"

        ].append(

            policy_loss.item()

        )



        metrics[

            "wdl_loss"

        ].append(

            wdl_loss.item()

        )



        metrics[

            "policy_kl"

        ].append(

            policy_kl.item()

        )



        metrics[

            "grad_norm"

        ].append(

            grad_norm.item()

        )



        for key in (

            "predicted_value",

            "target_value",

            "target_policy_entropy",

            "wdl_accuracy",

            "policy_top1",

            "loss_probability",

            "draw_probability",

            "win_probability",

            "target_loss_fraction",

            "target_draw_fraction",

            "target_win_fraction",

            "mean_legal_actions",

        ):



            diagnostic_key = {

                "predicted_value":

                    "predicted_value_mean",



                "target_value":

                    "target_value_mean",

            }.get(

                key,

                key,

            )



            metrics[

                key

            ].append(

                diagnostics[

                    diagnostic_key

                ]

            )



    return {

        "average_total_loss":

            float(

                np.mean(

                    metrics[

                        "total_loss"

                    ]

                )

            ),



        "average_policy_loss":

            float(

                np.mean(

                    metrics[

                        "policy_loss"

                    ]

                )

            ),



        "average_wdl_loss":

            float(

                np.mean(

                    metrics[

                        "wdl_loss"

                    ]

                )

            ),



        "average_policy_kl":

            float(

                np.mean(

                    metrics[

                        "policy_kl"

                    ]

                )

            ),



        "average_grad_norm":

            float(

                np.mean(

                    metrics[

                        "grad_norm"

                    ]

                )

            ),



        "average_predicted_value":

            float(

                np.mean(

                    metrics[

                        "predicted_value"

                    ]

                )

            ),



        "average_target_value":

            float(

                np.mean(

                    metrics[

                        "target_value"

                    ]

                )

            ),



        "average_target_policy_entropy":

            float(

                np.mean(

                    metrics[

                        "target_policy_entropy"

                    ]

                )

            ),



        "average_wdl_accuracy":

            float(

                np.mean(

                    metrics[

                        "wdl_accuracy"

                    ]

                )

            ),



        "average_policy_top1":

            float(

                np.mean(

                    metrics[

                        "policy_top1"

                    ]

                )

            ),



        "average_loss_probability":

            float(

                np.mean(

                    metrics[

                        "loss_probability"

                    ]

                )

            ),



        "average_draw_probability":

            float(

                np.mean(

                    metrics[

                        "draw_probability"

                    ]

                )

            ),



        "average_win_probability":

            float(

                np.mean(

                    metrics[

                        "win_probability"

                    ]

                )

            ),



        "average_target_loss_fraction":

            float(

                np.mean(

                    metrics[

                        "target_loss_fraction"

                    ]

                )

            ),



        "average_target_draw_fraction":

            float(

                np.mean(

                    metrics[

                        "target_draw_fraction"

                    ]

                )

            ),



        "average_target_win_fraction":

            float(

                np.mean(

                    metrics[

                        "target_win_fraction"

                    ]

                )

            ),



        "average_legal_actions":

            float(

                np.mean(

                    metrics[

                        "mean_legal_actions"

                    ]

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

    Count games and CURRENTLY SURVIVING positions in a split.



    This remains correct after circular replay-buffer overwrites.

    """



    num_games = 0

    num_positions = 0



    for (

        game_id,

        game

    ) in replay_buffer.games.items():



        if (

            game.get(

                "split"

            )

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

    Evaluate Model 4 on held-out rich replay positions.

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



    metric_names = [

        "total_loss",

        "policy_loss",

        "wdl_loss",

        "policy_kl",

        "predicted_value",

        "target_value",

        "target_policy_entropy",

        "wdl_accuracy",

        "policy_top1",

        "loss_probability",

        "draw_probability",

        "win_probability",

        "target_loss_fraction",

        "target_draw_fraction",

        "target_win_fraction",

        "mean_legal_actions",

    ]



    metrics = {

        name: []

        for name

        in metric_names

    }



    batches_evaluated = 0



    with torch.inference_mode():



        for _ in range(

            validation_steps

        ):



            raw_batch = replay_buffer.sample(

                batch_size,

                split=split,

            )



            if not raw_batch:

                break



            batch = build_training_batch(

                batch=raw_batch,

                replay_buffer=replay_buffer,

            )



            batch = move_batch_to_device(

                batch,

                device,

            )



            (

                policy_logits,

                wdl_logits,

            ) = model(

                batch[

                    "observations"

                ],

                batch[

                    "legal_action_ids"

                ],

                batch[

                    "legal_action_mask"

                ],

            )



            (

                loss,

                policy_loss,

                wdl_loss,

                policy_kl,

            ) = policy_wdl_loss(

                policy_logits=policy_logits,

                wdl_logits=wdl_logits,

                target_policy=batch[

                    "target_policy"

                ],

                target_wdl=batch[

                    "target_wdl"

                ],

                legal_action_mask=batch[

                    "legal_action_mask"

                ],

            )



            diagnostics = (

                collect_batch_diagnostics(

                    policy_logits=policy_logits,

                    wdl_logits=wdl_logits,

                    target_policy=batch[

                        "target_policy"

                    ],

                    target_wdl=batch[

                        "target_wdl"

                    ],

                    action_counts=batch[

                        "action_counts"

                    ],

                )

            )



            metrics[

                "total_loss"

            ].append(

                loss.item()

            )



            metrics[

                "policy_loss"

            ].append(

                policy_loss.item()

            )



            metrics[

                "wdl_loss"

            ].append(

                wdl_loss.item()

            )



            metrics[

                "policy_kl"

            ].append(

                policy_kl.item()

            )



            metrics[

                "predicted_value"

            ].append(

                diagnostics[

                    "predicted_value_mean"

                ]

            )



            metrics[

                "target_value"

            ].append(

                diagnostics[

                    "target_value_mean"

                ]

            )



            for key in (

                "target_policy_entropy",

                "wdl_accuracy",

                "policy_top1",

                "loss_probability",

                "draw_probability",

                "win_probability",

                "target_loss_fraction",

                "target_draw_fraction",

                "target_win_fraction",

                "mean_legal_actions",

            ):



                metrics[

                    key

                ].append(

                    diagnostics[

                        key

                    ]

                )



            batches_evaluated += 1



    if was_training:

        model.train()



    if not metrics[

        "total_loss"

    ]:

        return None



    return {

        "average_total_loss":

            float(

                np.mean(

                    metrics[

                        "total_loss"

                    ]

                )

            ),



        "average_policy_loss":

            float(

                np.mean(

                    metrics[

                        "policy_loss"

                    ]

                )

            ),



        "average_wdl_loss":

            float(

                np.mean(

                    metrics[

                        "wdl_loss"

                    ]

                )

            ),



        "average_policy_kl":

            float(

                np.mean(

                    metrics[

                        "policy_kl"

                    ]

                )

            ),



        "average_predicted_value":

            float(

                np.mean(

                    metrics[

                        "predicted_value"

                    ]

                )

            ),



        "average_target_value":

            float(

                np.mean(

                    metrics[

                        "target_value"

                    ]

                )

            ),



        "average_target_policy_entropy":

            float(

                np.mean(

                    metrics[

                        "target_policy_entropy"

                    ]

                )

            ),



        "average_wdl_accuracy":

            float(

                np.mean(

                    metrics[

                        "wdl_accuracy"

                    ]

                )

            ),



        "average_policy_top1":

            float(

                np.mean(

                    metrics[

                        "policy_top1"

                    ]

                )

            ),



        "average_loss_probability":

            float(

                np.mean(

                    metrics[

                        "loss_probability"

                    ]

                )

            ),



        "average_draw_probability":

            float(

                np.mean(

                    metrics[

                        "draw_probability"

                    ]

                )

            ),



        "average_win_probability":

            float(

                np.mean(

                    metrics[

                        "win_probability"

                    ]

                )

            ),



        "average_target_loss_fraction":

            float(

                np.mean(

                    metrics[

                        "target_loss_fraction"

                    ]

                )

            ),



        "average_target_draw_fraction":

            float(

                np.mean(

                    metrics[

                        "target_draw_fraction"

                    ]

                )

            ),



        "average_target_win_fraction":

            float(

                np.mean(

                    metrics[

                        "target_win_fraction"

                    ]

                )

            ),



        "average_legal_actions":

            float(

                np.mean(

                    metrics[

                        "mean_legal_actions"

                    ]

                )

            ),



        "batches_evaluated":

            int(

                batches_evaluated

            ),

    }





# ============================================================

# V5 PRUNING SEARCH STATS

# ============================================================





def get_pruning_search_stats(

    replay_buffer,

    model_id,

):

    """

    Summarize V5 adaptive-search diagnostics for the games

    generated by one training iteration/model snapshot.

    """



    game_ids = {

        game_id

        for game_id, game

        in replay_buffer.games.items()

        if (

            game.get("model_id") == model_id

            and int(

                replay_buffer.game_sample_counts.get(

                    game_id,

                    0,

                )

            ) > 0

        )

    }



    if not game_ids:

        return None



    samples = [

        sample

        for sample in replay_buffer.buffer

        if (

            isinstance(sample, dict)

            and sample.get("game_id") in game_ids

            and "search_actual_simulations" in sample

        )

    ]



    if not samples:

        return None



    actual = np.asarray(

        [

            int(sample["search_actual_simulations"])

            for sample in samples

        ],

        dtype=np.float64,

    )



    maximum = np.asarray(

        [

            int(

                sample.get(

                    "search_max_simulations",

                    sample["search_actual_simulations"],

                )

            )

            for sample in samples

        ],

        dtype=np.float64,

    )



    initial_visits = np.asarray(

        [

            int(sample.get("search_initial_root_visits", 0))

            for sample in samples

        ],

        dtype=np.float64,

    )



    final_visits = np.asarray(

        [

            int(sample.get("search_final_root_visits", 0))

            for sample in samples

        ],

        dtype=np.float64,

    )



    legal_actions = np.asarray(

        [

            int(

                sample.get(

                    "search_num_legal_actions",

                    len(sample.get("legal_action_ids", [])),

                )

            )

            for sample in samples

        ],

        dtype=np.float64,

    )



    stop_reason_counts = {}



    for sample in samples:



        reason = str(

            sample.get(

                "search_stop_reason",

                "unknown",

            )

        )



        stop_reason_counts[reason] = (

            stop_reason_counts.get(reason, 0)

            + 1

        )



    total_maximum = float(maximum.sum())



    simulation_savings_fraction = (

        1.0

        - float(actual.sum()) / total_maximum

        if total_maximum > 0.0

        else 0.0

    )



    hit_maximum_fraction = float(

        np.mean(actual >= maximum)

    )



    return {

        "positions":

            int(len(samples)),



        "average_actual_simulations":

            float(actual.mean()),



        "median_actual_simulations":

            float(np.median(actual)),



        "min_actual_simulations":

            int(actual.min()),



        "max_actual_simulations":

            int(actual.max()),



        "simulation_savings_fraction":

            float(simulation_savings_fraction),



        "hit_maximum_fraction":

            hit_maximum_fraction,



        "average_initial_root_visits":

            float(initial_visits.mean()),



        "average_final_root_visits":

            float(final_visits.mean()),



        "average_legal_actions":

            float(legal_actions.mean()),



        "stop_reason_counts":

            stop_reason_counts,

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



    # Model 4 defaults to a fresh deterministic seed per game.

    dynamic_seeding=True,



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



    model_generation=4,

    action_space_version=1,

    model_name="Model4",

    run_start_checkpoint=None,



    # Optional extra MCTS constructor arguments.

    mcts_kwargs=None,



    # One pathological game must never stall the whole run.

    max_game_steps=300,



    # Prevent repeated failed seeds / environment bugs from creating

    # an infinite retry loop at the outer training level.

    max_consecutive_failures=10,



    # --------------------------------------------------------

    # OPTIMIZATION

    # --------------------------------------------------------



    grad_clip=1.0,



    # --------------------------------------------------------

    # TRAIN / VALIDATION SPLIT

    # --------------------------------------------------------



    validation_fraction=0.10,

    split_seed=20260917,



    train_sample_split="train",

    val_sample_split="val",



    min_validation_games=5,

    min_validation_positions=256,



    validation_steps=10,

    validation_batch_size=None,



    # --------------------------------------------------------

    # REPLAY SAVE

    # --------------------------------------------------------



    replay_buffer_path=None,

):

    """

    Model 4 self-play + training loop using MCTS V5 adaptive pruning.



    Each iteration:



        1. Freeze the current model for self-play.

        2. Generate self-play games with MCTS v5 adaptive pruning.

        3. Commit only fully completed rich replay games.

        4. Train on the train split using dynamic legal-action batches.

        5. Validate on held-out games.

        6. Save model/replay checkpoints on schedule.

    """



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



    if max_game_steps < 1:



        raise ValueError(

            "max_game_steps must be >= 1."

        )



    if max_consecutive_failures < 1:



        raise ValueError(

            "max_consecutive_failures must be >= 1."

        )



    if grad_clip is not None and grad_clip <= 0:



        raise ValueError(

            "grad_clip must be positive or None."

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



    consecutive_failures = 0

    failed_game_seeds = []



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

        failed_games_this_iteration = 0



        train_games_this_iteration = 0

        val_games_this_iteration = 0



        train_positions_this_iteration = 0

        val_positions_this_iteration = 0



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



        model.eval()



        # ====================================================

        # MCTS V5 PRUNING

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

        # V5 PRUNING REPLAY GENERATOR

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



                max_game_steps=(

                    max_game_steps

                ),

            )

        )



        # ====================================================

        # SELF PLAY

        # ====================================================



        while (

            successful_games_this_iteration

            < self_play_games_per_iteration

        ):



            game_index = (

                games_attempted

            )



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



                result = (

                    replay_generator.generate_game(

                        seed=game_seed,

                        split=game_split,

                        model_generation=(

                            model_generation

                        ),

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



                            "trainer_version":

                                5,

                        },

                    )

                )



            except Exception as error:



                failed_games_this_iteration += 1

                consecutive_failures += 1



                failed_game_seeds.append(

                    game_seed

                )



                print()

                print(

                    "=" * 70

                )



                print(

                    "SELF-PLAY GAME FAILED "

                    f"(attempt {game_index})"

                )



                print(

                    f"seed={game_seed}"

                )



                print(

                    f"{type(error).__name__}: "

                    f"{error}"

                )



                traceback.print_exc()



                print(

                    "Consecutive failures: "

                    f"{consecutive_failures}/"

                    f"{max_consecutive_failures}"

                )



                print(

                    "=" * 70

                )



                print()



                if (

                    consecutive_failures

                    >= max_consecutive_failures

                ):



                    raise RuntimeError(

                        "Model 4 / MCTS V5 self-play aborted after "

                        f"{consecutive_failures} consecutive "

                        "failed games. "

                        f"Recent failed seeds: "

                        f"{failed_game_seeds[-max_consecutive_failures:]}"

                    ) from error



                continue



            # ------------------------------------------------

            # SUCCESS

            # ------------------------------------------------



            consecutive_failures = 0



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



            print(

                f"Game {games_played} complete "

                f"- attempt={game_index} "

                f"- seed={game_seed} "

                f"- split={game_split} "

                f"- {game_positions} positions "

                f"- replay size "

                f"{len(replay_buffer):,}"

            )



        # ====================================================

        # V5 PRUNING SEARCH STATS

        # ====================================================



        pruning_stats = (

            get_pruning_search_stats(

                replay_buffer=replay_buffer,

                model_id=generating_model_id,

            )

        )



        if pruning_stats is not None:



            print(

                "MCTS V5 average simulations:",

                f"{pruning_stats['average_actual_simulations']:.2f}",

            )



            print(

                "MCTS V5 median simulations:",

                f"{pruning_stats['median_actual_simulations']:.2f}",

            )



            print(

                "MCTS V5 simulation savings:",

                f"{100.0 * pruning_stats['simulation_savings_fraction']:.2f}%",

            )



            print(

                "MCTS V5 hit-max fraction:",

                f"{100.0 * pruning_stats['hit_maximum_fraction']:.2f}%",

            )



            print(

                "MCTS V5 average inherited root visits:",

                f"{pruning_stats['average_initial_root_visits']:.2f}",

            )



            print(

                "MCTS V5 stop reasons:",

                pruning_stats["stop_reason_counts"],

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

                    grad_clip=grad_clip,

                )

            )



            history.append(

                training_results

            )



        # ====================================================

        # VALIDATION

        # ====================================================



        enough_validation_data = (

            val_stats[

                "games"

            ]

            >= min_validation_games

            and

            val_stats[

                "positions"

            ]

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



            if (

                validation_results

                is not None

            ):



                print(

                    "Validation loss:",

                    f"{validation_results['average_total_loss']:.4f}",

                )



                print(

                    "Validation policy loss:",

                    f"{validation_results['average_policy_loss']:.4f}",

                )



                print(

                    "Validation policy KL:",

                    f"{validation_results['average_policy_kl']:.4f}",

                )



                print(

                    "Validation policy top-1:",

                    f"{validation_results['average_policy_top1']:.4f}",

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

                "SelfPlay/failed_games_iteration",

                failed_games_this_iteration,

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

                "Training/training_steps",

                training_steps,

                games_played,

            )



            writer.add_scalar(

                "Training/learning_rate",

                optimizer.param_groups[

                    0

                ][

                    "lr"

                ],

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




            if pruning_stats is not None:



                writer.add_scalar(

                    "MCTS_V5/average_simulations",

                    pruning_stats["average_actual_simulations"],

                    games_played,

                )



                writer.add_scalar(

                    "MCTS_V5/median_simulations",

                    pruning_stats["median_actual_simulations"],

                    games_played,

                )



                writer.add_scalar(

                    "MCTS_V5/simulation_savings_fraction",

                    pruning_stats["simulation_savings_fraction"],

                    games_played,

                )



                writer.add_scalar(

                    "MCTS_V5/hit_maximum_fraction",

                    pruning_stats["hit_maximum_fraction"],

                    games_played,

                )



                writer.add_scalar(

                    "MCTS_V5/average_initial_root_visits",

                    pruning_stats["average_initial_root_visits"],

                    games_played,

                )



                writer.add_scalar(

                    "MCTS_V5/average_final_root_visits",

                    pruning_stats["average_final_root_visits"],

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



            if (

                training_results

                is not None

            ):



                training_tensorboard = {

                    "Loss/total":

                        "average_total_loss",



                    "Loss/policy":

                        "average_policy_loss",



                    "Loss/wdl":

                        "average_wdl_loss",



                    "Loss/policy_kl":

                        "average_policy_kl",



                    "Training/gradient_norm":

                        "average_grad_norm",



                    "Training/policy_top1":

                        "average_policy_top1",



                    "Training/wdl_accuracy":

                        "average_wdl_accuracy",



                    "Training/predicted_value":

                        "average_predicted_value",



                    "Training/target_value":

                        "average_target_value",



                    "Training/target_policy_entropy":

                        "average_target_policy_entropy",



                    "Training/mean_legal_actions":

                        "average_legal_actions",

                }



                for (

                    tag,

                    key

                ) in training_tensorboard.items():



                    writer.add_scalar(

                        tag,

                        training_results[

                            key

                        ],

                        games_played,

                    )



            if (

                validation_results

                is not None

            ):



                validation_tensorboard = {

                    "Validation/loss_total":

                        "average_total_loss",



                    "Validation/loss_policy":

                        "average_policy_loss",



                    "Validation/loss_wdl":

                        "average_wdl_loss",



                    "Validation/policy_kl":

                        "average_policy_kl",



                    "Validation/policy_top1":

                        "average_policy_top1",



                    "Validation/wdl_accuracy":

                        "average_wdl_accuracy",



                    "Validation/predicted_value":

                        "average_predicted_value",



                    "Validation/target_value":

                        "average_target_value",



                    "Validation/mean_legal_actions":

                        "average_legal_actions",

                }



                for (

                    tag,

                    key

                ) in validation_tensorboard.items():



                    writer.add_scalar(

                        tag,

                        validation_results[

                            key

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

            "Successful games this iteration:",

            successful_games_this_iteration,

        )



        print(

            "Failed games this iteration:",

            failed_games_this_iteration,

        )



        print(

            "Train games:",

            train_games_this_iteration,

        )



        print(

            "Validation games:",

            val_games_this_iteration,

        )



        print(

            "Train positions:",

            train_positions_this_iteration,

        )



        print(

            "Validation positions:",

            val_positions_this_iteration,

        )



        print(

            "Self-play time:",

            f"{total_self_play_time:.2f}s",

        )



        print(

            "Iteration time:",

            f"{iteration_time:.2f}s",

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




        if pruning_stats is not None:



            print(

                "Average MCTS V5 simulations:",

                f"{pruning_stats['average_actual_simulations']:.2f}",

            )



            print(

                "MCTS V5 simulation savings:",

                f"{100.0 * pruning_stats['simulation_savings_fraction']:.2f}%",

            )



        print(

            "Learning rate:",

            optimizer.param_groups[

                0

            ][

                "lr"

            ],

        )



        print(

            "=" * 70

        )



        if (

            validation_results

            is not None

        ):



            print(

                "Validation total loss:",

                f"{validation_results['average_total_loss']:.4f}",

            )



            print(

                "Validation policy loss:",

                f"{validation_results['average_policy_loss']:.4f}",

            )



            print(

                "Validation policy KL:",

                f"{validation_results['average_policy_kl']:.4f}",

            )



            print(

                "Validation WDL loss:",

                f"{validation_results['average_wdl_loss']:.4f}",

            )



            print(

                "Validation WDL accuracy:",

                f"{validation_results['average_wdl_accuracy']:.4f}",

            )



        print()



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

        "Final checkpoint saved:",

        final_checkpoint_path,

    )



    if failed_game_seeds:



        print(

            "Failed game seeds during run:",

            failed_game_seeds,

        )



    return history
