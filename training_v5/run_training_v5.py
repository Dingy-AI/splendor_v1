import os

import pickle



import torch



from splendor_v1.env.env import SplendorEnv



from splendor_v1.network.model_4_legal_scorer import (

    SplendorNetwork,

)



from splendor_v1.training_v2.replay_buffer import (

    ReplayBuffer,

)



from splendor_v1.training_v5.train_v5 import (

    run_training,

)





# ============================================================

# MODE

# ============================================================

#

# False:

#     Start a NEW Model 4 self-play run.

#

#     - Load START_CHECKPOINT_PATH as the initial network.

#     - Create a fresh empty V5 pruning rich replay buffer.

#     - Start self-play game numbering at 0.

#     - Do NOT restore optimizer state from supervised pretraining.

#

# True:

#     CONTINUE a previous Model 4 self-play run.

#

#     - Load RESUME_CHECKPOINT_PATH.

#     - Load RESUME_REPLAY_PATH.

#     - Restore optimizer state when available.

#     - Recover successful-game and attempted-game counters.

#     - Continue deterministic seed progression instead of starting over.

#

# ============================================================



RESUME_TRAINING = True





# ============================================================

# TEMPERATURE SCHEDULE

# ============================================================





def splendor_temperature(

    turn_number,

):

    """

    Self-play action-selection temperature.



    Early game:

        broad exploration.



    Mid game:

        bias toward higher-visit actions.



    Late game:

        deterministic highest-visit action.

    """



    if turn_number < 20:

        return 1.0



    if turn_number < 40:

        return 0.5



    return 0.0





def splendor_root_noise(

    turn_number,

):

    """

    Apply Dirichlet root noise only during the exploratory

    early/mid game.

    """



    return (

        turn_number < 40

    )





# ============================================================

# PATHS

# ============================================================



# ------------------------------------------------------------

# Fresh-start checkpoint

# ------------------------------------------------------------



START_CHECKPOINT_PATH = (

    "splendor_v1/training_v5/data/"

    "model_1900_games.pt"

)





# ------------------------------------------------------------

# Resume checkpoint

# ------------------------------------------------------------

#

# Set this to the exact V5 pruning self-play checkpoint you want to resume.

#

# Example:

#

#     checkpoints/gen_4/self_play_v5_pruning/model_1000_games.pt

#

# or:

#

#     checkpoints/gen_4/self_play_v5_pruning/model_1000_games_last.pt

#

# depending on your checkpoint naming.

# ------------------------------------------------------------



RESUME_CHECKPOINT_PATH = (

    "splendor_v1/training_v5/data/"

    "model_2000_games.pt"

)





# ------------------------------------------------------------

# Replay

# ------------------------------------------------------------



OUTPUT_REPLAY_PATH = (

    "splendor_v1/training_v5/data/"

    "replay_buffer_model4_mcts_v5_pruning.pkl"

)



# Normally this is the same file as OUTPUT_REPLAY_PATH.

#

# It is separate so you can resume from a copied/archive replay

# while saving new progress somewhere else if desired.



RESUME_REPLAY_PATH = (

    "splendor_v1/training_v5/data/"

    "replay_2000_games.pkl"

)





# ------------------------------------------------------------

# Checkpoints

# ------------------------------------------------------------



OUTPUT_CHECKPOINT_DIR = (

    "splendor_v1/training_v5/data/"

)





# ============================================================

# TRAINING CONFIG

# ============================================================



# Intended long-run shape:

#

# 2500 iterations * 10 games = 25,000 successful games.

#

# For the first smoke test, temporarily use:

#

#     NUM_ITERATIONS = 1

#     SELF_PLAY_GAMES_PER_ITERATION = 10

#     SIMULATIONS = 400



# ------------------------------------------------------------
# MCTS V5 adaptive pruning
# ------------------------------------------------------------
#
# SIMULATIONS remains the absolute hard ceiling.
#
# V5.0 does not remove legal actions. It only reduces how long
# a root is searched when enough evidence has accumulated.
# ------------------------------------------------------------

ADAPTIVE_SIMULATIONS = True

MIN_SIMULATIONS = 80

CHECK_INTERVAL = 20

TARGET_VISITS_PER_ACTION = 20.0

SINGLE_ACTION_SIMULATIONS = 4

STABILITY_CHECKS = 3
#

# V5 keeps the 400-simulation hard cap during the smoke test;
# adaptive pruning determines the actual simulations used.



NUM_ITERATIONS = 100



SELF_PLAY_GAMES_PER_ITERATION = 10



SIMULATIONS = 400



BATCH_SIZE = 256



TRAINING_RATIO = 1.5





# ------------------------------------------------------------

# Optimizer

# ------------------------------------------------------------



LEARNING_RATE = 1e-4



WEIGHT_DECAY = 0.0



GRAD_CLIP = 1.0





# ------------------------------------------------------------

# Replay

# ------------------------------------------------------------



REPLAY_CAPACITY = 500_000





# ------------------------------------------------------------

# Checkpoint cadence

# ------------------------------------------------------------



CHECKPOINT_EVERY_GAMES = 100





# ------------------------------------------------------------

# Self-play safety

# ------------------------------------------------------------



MAX_GAME_STEPS = 300



MAX_CONSECUTIVE_FAILURES = 10





# ------------------------------------------------------------

# Reproducible game seeds

# ------------------------------------------------------------

#

# Fresh run:

#

#     attempt 0 -> seed 10000

#     attempt 1 -> seed 10001

#     ...

#

# Resume:

#

# starting_games_attempted is restored, so seed numbering continues.

#

# ------------------------------------------------------------



BASE_SEED = 10000



DYNAMIC_SEEDING = True





# ------------------------------------------------------------

# Train / validation

# ------------------------------------------------------------



VALIDATION_FRACTION = 0.10



SPLIT_SEED = 20260917



MIN_VALIDATION_GAMES = 5



MIN_VALIDATION_POSITIONS = 256



VALIDATION_STEPS = 10





# ============================================================

# DEVICE

# ============================================================



device = torch.device(

    "cuda"

    if torch.cuda.is_available()

    else "cpu"

)



print(

    "Device:",

    device,

)





# ============================================================

# HELPERS: CHECKPOINT

# ============================================================





def extract_model_state_dict(

    checkpoint,

):

    """

    Support:



        1. normal checkpoint dict:

               {"model_state_dict": ...}



        2. raw state_dict

    """



    if (

        isinstance(

            checkpoint,

            dict,

        )

        and "model_state_dict"

        in checkpoint

    ):



        return checkpoint[

            "model_state_dict"

        ]



    if (

        isinstance(

            checkpoint,

            dict,

        )

        and checkpoint

        and all(

            torch.is_tensor(

                value

            )

            for value

            in checkpoint.values()

        )

    ):



        return checkpoint



    raise RuntimeError(

        "Checkpoint must be either a raw "

        "model state_dict or contain "

        "'model_state_dict'."

    )





def load_model_checkpoint(

    checkpoint_path,

):

    if not os.path.exists(

        checkpoint_path

    ):



        raise FileNotFoundError(

            "Model checkpoint does not exist: "

            f"{checkpoint_path}"

        )



    print(

        "Loading checkpoint:",

        checkpoint_path,

    )



    checkpoint = torch.load(

        checkpoint_path,

        map_location=device,

    )



    state_dict = (

        extract_model_state_dict(

            checkpoint

        )

    )



    model = SplendorNetwork()



    model.load_state_dict(

        state_dict,

        strict=True,

    )



    model = model.to(

        device

    )



    print(

        "Model loaded successfully."

    )



    return (

        model,

        checkpoint,

    )





# ============================================================

# HELPERS: OPTIMIZER

# ============================================================





def create_optimizer(

    model,

):

    return torch.optim.AdamW(

        model.parameters(),

        lr=LEARNING_RATE,

        weight_decay=WEIGHT_DECAY,

    )





def move_optimizer_state_to_device(

    optimizer,

    target_device,

):

    """

    Defensive helper for resumed checkpoints.



    Ensures Adam/AdamW moment tensors live on the same device

    as the Model 4 parameters.

    """



    for state in optimizer.state.values():



        for key, value in state.items():



            if torch.is_tensor(

                value

            ):



                state[

                    key

                ] = value.to(

                    target_device

                )





def maybe_restore_optimizer(

    optimizer,

    checkpoint,

):

    if not RESUME_TRAINING:



        print(

            "Fresh run: optimizer starts new."

        )



        return False



    if not isinstance(

        checkpoint,

        dict,

    ):



        print(

            "Resume checkpoint has no optimizer metadata. "

            "Starting a fresh optimizer."

        )



        return False



    optimizer_state = (

        checkpoint.get(

            "optimizer_state_dict"

        )

    )



    if optimizer_state is None:



        print(

            "Resume checkpoint has no optimizer_state_dict. "

            "Starting a fresh optimizer."

        )



        return False



    optimizer.load_state_dict(

        optimizer_state

    )



    move_optimizer_state_to_device(

        optimizer,

        device,

    )



    print(

        "Optimizer state restored."

    )



    return True





# ============================================================

# HELPERS: RICH REPLAY LOAD

# ============================================================





def load_rich_replay_buffer(

    replay_path,

):

    """

    Restore the rich replay format used by Model 4.



    Preserved fields:



        capacity

        buffer

        position

        metadata

        games

        game_sample_counts

        next_game_id



    Also accepts a directly pickled ReplayBuffer object.

    """



    if not os.path.exists(

        replay_path

    ):



        raise FileNotFoundError(

            "Replay buffer does not exist: "

            f"{replay_path}"

        )



    print(

        "Loading replay:",

        replay_path,

    )



    with open(

        replay_path,

        "rb",

    ) as file:



        data = pickle.load(

            file

        )



    if isinstance(

        data,

        ReplayBuffer,

    ):



        replay_buffer = data



    elif isinstance(

        data,

        dict,

    ):



        required_fields = (

            "capacity",

            "buffer",

            "position",

            "games",

            "game_sample_counts",

        )



        missing_fields = [

            field

            for field

            in required_fields

            if field not in data

        ]



        if missing_fields:



            raise RuntimeError(

                "Replay file is not the rich replay "

                "format required by Model 4. "

                f"Missing fields: {missing_fields}"

            )



        metadata = data.get(

            "metadata",

            {},

        )



        try:



            replay_buffer = ReplayBuffer(

                capacity=int(

                    data[

                        "capacity"

                    ]

                ),

                metadata=metadata,

            )



        except TypeError:



            # Backward-compatible fallback in case the local

            # ReplayBuffer constructor does not expose metadata.



            replay_buffer = ReplayBuffer(

                int(

                    data[

                        "capacity"

                    ]

                )

            )



            replay_buffer.metadata = (

                metadata

            )



        replay_buffer.buffer = (

            data[

                "buffer"

            ]

        )



        replay_buffer.position = int(

            data[

                "position"

            ]

        )



        replay_buffer.games = (

            data[

                "games"

            ]

        )



        replay_buffer.game_sample_counts = (

            data[

                "game_sample_counts"

            ]

        )



        replay_buffer.next_game_id = int(

            data.get(

                "next_game_id",

                len(

                    replay_buffer.games

                ),

            )

        )



        replay_buffer.metadata = (

            metadata

        )



    else:



        raise RuntimeError(

            "Unsupported replay file type: "

            f"{type(data).__name__}"

        )



    if not hasattr(

        replay_buffer,

        "games",

    ):



        raise RuntimeError(

            "Loaded replay has no game metadata."

        )



    if not hasattr(

        replay_buffer,

        "game_sample_counts",

    ):



        raise RuntimeError(

            "Loaded replay has no game_sample_counts."

        )



    print(

        "Replay loaded successfully."

    )



    print(

        "Replay positions:",

        f"{len(replay_buffer):,}",

    )



    print(

        "Replay game records:",

        f"{len(replay_buffer.games):,}",

    )



    print(

        "Replay next_game_id:",

        getattr(

            replay_buffer,

            "next_game_id",

            None,

        ),

    )



    return replay_buffer





# ============================================================

# HELPERS: FRESH REPLAY

# ============================================================





def create_fresh_replay_buffer():

    metadata = {

        "dataset_family":

            "training_v5",



        "schema_version":

            1,



        "model_generation":

            4,



        "architecture":

            "attention_wdl_legal_action_scorer",



        "source_model":

            "Model4",



        "source_checkpoint":

            START_CHECKPOINT_PATH,



        "run_type":

            "self_play_mcts_v5_pruning",



        "search_variant":

            "mcts_v5_pruning",

    }



    try:



        replay_buffer = ReplayBuffer(

            capacity=REPLAY_CAPACITY,

            metadata=metadata,

        )



    except TypeError:



        replay_buffer = ReplayBuffer(

            REPLAY_CAPACITY

        )



        replay_buffer.metadata = (

            metadata

        )



    print(

        "Created fresh Model 4 / MCTS V5 pruning replay buffer."

    )



    print(

        "Replay capacity:",

        f"{REPLAY_CAPACITY:,}",

    )



    return replay_buffer





# ============================================================

# HELPERS: RESUME COUNTERS

# ============================================================





def infer_games_played_from_checkpoint(

    checkpoint,

):

    """

    Self-play checkpoints created by training/checkpoint.py normally

    include games_played.



    For a fresh supervised checkpoint this correctly falls back to 0.

    """



    if not isinstance(

        checkpoint,

        dict,

    ):



        return 0



    value = checkpoint.get(

        "games_played"

    )



    if value is None:



        return 0



    return int(

        value

    )





def infer_games_attempted_from_replay(

    replay_buffer,

    fallback_games_played,

):

    """

    Successful Model 4 games store:



        game_metadata["game_index"]



    game_index is the ATTEMPT index, not just the successful-game count.



    Therefore:



        max(game_index) + 1



    restores seed progression even when earlier attempts failed.

    """



    highest_game_index = None



    for game in (

        replay_buffer

        .games

        .values()

    ):



        game_index = (

            game.get(

                "game_index"

            )

        )



        if game_index is None:

            continue



        game_index = int(

            game_index

        )



        if (

            highest_game_index

            is None

            or game_index

            > highest_game_index

        ):



            highest_game_index = (

                game_index

            )



    if highest_game_index is None:



        return int(

            fallback_games_played

        )



    return (

        highest_game_index

        + 1

    )





def count_v5_generated_games(

    replay_buffer,

):

    """

    Diagnostic only.



    Counts stored game records tagged as Model 4 self-play.

    This is NOT used as the authoritative resume counter because

    circular replay-buffer cleanup may eventually remove old games.

    """



    count = 0



    for game in (

        replay_buffer

        .games

        .values()

    ):



        if (

            game.get(

                "model_generation"

            )

            == 4

            and (

                game.get(

                    "trainer_version"

                )

                == 5

                or game.get(

                    "search",

                    {},

                ).get(

                    "search_variant"

                )

                == "mcts_v5_pruning"

            )

        ):



            count += 1



    return count





# ============================================================

# CREATE OUTPUT DIRECTORIES

# ============================================================



os.makedirs(

    OUTPUT_CHECKPOINT_DIR,

    exist_ok=True,

)



output_replay_directory = (

    os.path.dirname(

        OUTPUT_REPLAY_PATH

    )

)



if output_replay_directory:



    os.makedirs(

        output_replay_directory,

        exist_ok=True,

    )





# ============================================================

# ENVIRONMENT

# ============================================================



env = SplendorEnv()





# ============================================================

# LOAD MODEL + REPLAY MODE

# ============================================================



if RESUME_TRAINING:



    active_checkpoint_path = (

        RESUME_CHECKPOINT_PATH

    )



    (

        model,

        checkpoint,

    ) = load_model_checkpoint(

        active_checkpoint_path

    )



    replay_buffer = (

        load_rich_replay_buffer(

            RESUME_REPLAY_PATH

        )

    )



    starting_games_played = (

        infer_games_played_from_checkpoint(

            checkpoint

        )

    )



    starting_games_attempted = (

        infer_games_attempted_from_replay(

            replay_buffer=replay_buffer,

            fallback_games_played=(

                starting_games_played

            ),

        )

    )



else:



    active_checkpoint_path = (

        START_CHECKPOINT_PATH

    )



    (

        model,

        checkpoint,

    ) = load_model_checkpoint(

        active_checkpoint_path

    )



    replay_buffer = (

        create_fresh_replay_buffer()

    )



    starting_games_played = 0



    starting_games_attempted = 0





# ============================================================

# OPTIMIZER

# ============================================================



optimizer = create_optimizer(

    model

)



optimizer_restored = (

    maybe_restore_optimizer(

        optimizer,

        checkpoint,

    )

)





# ============================================================

# RESUME DIAGNOSTICS

# ============================================================



print()

print(

    "=" * 70

)



if RESUME_TRAINING:



    print(

        "MODEL 4 + MCTS V5 PRUNING RESUME MODE"

    )



else:



    print(

        "MODEL 4 + MCTS V5 PRUNING FRESH SELF-PLAY MODE"

    )



print(

    "=" * 70

)



print(

    "Active checkpoint:",

    active_checkpoint_path,

)



print(

    "Output replay:",

    OUTPUT_REPLAY_PATH,

)



print(

    "Starting successful games:",

    starting_games_played,

)



print(

    "Starting attempted games:",

    starting_games_attempted,

)



print(

    "Next environment seed:",

    (

        None

        if BASE_SEED is None

        else (

            BASE_SEED

            + starting_games_attempted

            if DYNAMIC_SEEDING

            else BASE_SEED

        )

    ),

)



print(

    "Optimizer restored:",

    optimizer_restored,

)



print(

    "Stored V5 pruning replay games:",

    count_v5_generated_games(

        replay_buffer

    ),

)



print(

    "Replay positions:",

    f"{len(replay_buffer):,}",

)



print(

    "Simulations:",

    SIMULATIONS,

)




print(

    "Adaptive simulations:",

    ADAPTIVE_SIMULATIONS,

)



print(

    "Minimum simulations:",

    MIN_SIMULATIONS,

)



print(

    "Check interval:",

    CHECK_INTERVAL,

)



print(

    "Target visits/action:",

    TARGET_VISITS_PER_ACTION,

)



print(

    "Single-action simulations:",

    SINGLE_ACTION_SIMULATIONS,

)



print(

    "Stability checks:",

    STABILITY_CHECKS,

)



print(

    "Batch size:",

    BATCH_SIZE,

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

    "Checkpoint every games:",

    CHECKPOINT_EVERY_GAMES,

)



print(

    "Max game steps:",

    MAX_GAME_STEPS,

)



print(

    "=" * 70

)



print()





# ============================================================

# RUN MODEL 4 + MCTS V5 PRUNING TRAINING

# ============================================================



history = run_training(

    env=env,

    model=model,

    optimizer=optimizer,

    replay_buffer=replay_buffer,



    # --------------------------------------------------------

    # RUN SIZE

    # --------------------------------------------------------



    num_iterations=NUM_ITERATIONS,



    self_play_games_per_iteration=(

        SELF_PLAY_GAMES_PER_ITERATION

    ),



    simulations=SIMULATIONS,



    batch_size=BATCH_SIZE,



    training_ratio=TRAINING_RATIO,



    # --------------------------------------------------------

    # CHECKPOINTS

    # --------------------------------------------------------



    checkpoint_every_games=(

        CHECKPOINT_EVERY_GAMES

    ),



    checkpoint_dir=(

        OUTPUT_CHECKPOINT_DIR

    ),



    starting_games_played=(

        starting_games_played

    ),



    starting_games_attempted=(

        starting_games_attempted

    ),



    # --------------------------------------------------------

    # SELF PLAY

    # --------------------------------------------------------



    seed=BASE_SEED,



    dynamic_seeding=(

        DYNAMIC_SEEDING

    ),



    teacher_mode=False,



    temperature=1.0,



    temperature_fn=(

        splendor_temperature

    ),



    add_root_noise=True,



    root_noise_fn=(

        splendor_root_noise

    ),



    model_generation=4,



    model_name="Model4",



    run_start_checkpoint=(

        active_checkpoint_path

    ),



    action_space_version=1,



    max_game_steps=(

        MAX_GAME_STEPS

    ),



    max_consecutive_failures=(

        MAX_CONSECUTIVE_FAILURES

    ),



    # --------------------------------------------------------

    # OPTIMIZATION

    # --------------------------------------------------------



    grad_clip=GRAD_CLIP,



    # --------------------------------------------------------

    # TRAIN / VALIDATION

    # --------------------------------------------------------



    validation_fraction=(

        VALIDATION_FRACTION

    ),



    split_seed=SPLIT_SEED,



    train_sample_split="train",



    val_sample_split="val",



    min_validation_games=(

        MIN_VALIDATION_GAMES

    ),



    min_validation_positions=(

        MIN_VALIDATION_POSITIONS

    ),



    validation_steps=(

        VALIDATION_STEPS

    ),



    validation_batch_size=(

        BATCH_SIZE

    ),



    # --------------------------------------------------------

    # REPLAY SAVE

    # --------------------------------------------------------



    replay_buffer_path=(

        OUTPUT_REPLAY_PATH

    ),



    # --------------------------------------------------------

    # MCTS

    # --------------------------------------------------------



        mcts_kwargs={

        "c_puct":

            3.0,



        "dirichlet_alpha":

            0.3,



        "dirichlet_epsilon":

            0.25,



        "adaptive_simulations":

            ADAPTIVE_SIMULATIONS,



        "min_simulations":

            MIN_SIMULATIONS,



        "check_interval":

            CHECK_INTERVAL,



        "target_visits_per_action":

            TARGET_VISITS_PER_ACTION,



        "single_action_simulations":

            SINGLE_ACTION_SIMULATIONS,



        "stability_checks":

            STABILITY_CHECKS,

    },

)





# ============================================================

# FINAL SUMMARY

# ============================================================



print()

print(

    "=" * 70

)



print(

    "MODEL 4 + MCTS V5 PRUNING TRAINING RUN COMPLETE"

)



print(

    "=" * 70

)



print(

    "Replay samples:",

    len(

        replay_buffer

    ),

)



print(

    "Active game records:",

    len(

        replay_buffer.games

    ),

)



print(

    "Replay next_game_id:",

    getattr(

        replay_buffer,

        "next_game_id",

        None,

    ),

)



print(

    "Replay saved to:",

    OUTPUT_REPLAY_PATH,

)



print(

    "Checkpoint directory:",

    OUTPUT_CHECKPOINT_DIR,

)



print(

    "Training iterations completed:",

    len(

        history

    ),

)



print()
