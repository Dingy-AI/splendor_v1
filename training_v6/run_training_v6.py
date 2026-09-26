"""
Model 4 / MCTS V6 training runner.

V6 keeps the Model 4 network, rich replay format, WDL training,
dynamic legal-action training batches, and MCTS V5 adaptive-search
algorithm from V5.

The V6 change is SELF-PLAY EXECUTION:

    V5:
        one game at a time
        direct batch-size-1 GPU inference

    V6:
        multiple independent CPU game/MCTS processes
        -> centralized GPU inference server
        -> dynamic cross-game neural batching
        -> completed whole-game replay commits in MAIN

Training and self-play remain sequential phases:

    parallel self-play block
        ↓
    stop GPU inference server
        ↓
    train parent Model 4
        ↓
    validation
        ↓
    save current weights
        ↓
    next parallel self-play block

Windows multiprocessing requires the __main__ guard at the bottom.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import pickle
import time

import torch

from splendor_v1.network.model_4_legal_scorer import (
    SplendorNetwork,
)

from splendor_v1.training_v2.replay_buffer import (
    ReplayBuffer,
)

from splendor_v1.training.checkpoint import (
    save_model_if_needed,
    save_checkpoint,
)

# Reuse V5's already-validated Model 4 optimization and split code.
from splendor_v1.training_v5.train_v5 import (
    resolve_game_seed,
    choose_game_split,
    get_split_stats,
    train_network,
    validate_network,
)

from splendor_v1.mcts_batched.multiprocess_self_play_worker import (
    SelfPlayGameJob,
    SelfPlaySearchConfig,
    SelfPlayWorkerConfig,
)

from splendor_v1.mcts_batched.multiprocess_self_play_coordinator import (
    MultiprocessSelfPlayCoordinator,
    MultiprocessSelfPlayCoordinatorConfig,
)


# ============================================================
# MODE
# ============================================================

# False:
#     Start a new Model 4 V6 self-play run from START_CHECKPOINT_PATH.
#
# True:
#     Continue a previous Model 4 self-play run from the selected
#     model checkpoint + rich replay.
RESUME_TRAINING = True


# ============================================================
# PATHS
# ============================================================

START_CHECKPOINT_PATH = (
    "splendor_v1/training_v5/data/"
    "model_1900_games.pt"
)

RESUME_CHECKPOINT_PATH = (
    "splendor_v1/training_v5/data/"
    "model_2000_games.pt"
)

OUTPUT_REPLAY_PATH = (
    "splendor_v1/training_v6/data/"
    "replay_buffer_model4_mcts_v6_multiprocess.pkl"
)

# V5 replay is forward-compatible with V6 because the rich replay
# schema and Model 4 training targets are unchanged.
RESUME_REPLAY_PATH = (
    "splendor_v1/training_v5/data/"
    "replay_2000_games.pkl"
)

OUTPUT_CHECKPOINT_DIR = (
    "splendor_v1/training_v6/data/"
)

# Lightweight weight snapshot read by the centralized GPU process.
# This is rewritten after every training iteration.
INFERENCE_SNAPSHOT_PATH = (
    "splendor_v1/training_v6/data/"
    "model4_v6_inference_latest.pt"
)


# ============================================================
# RUN SIZE
# ============================================================

NUM_ITERATIONS = 1

SELF_PLAY_GAMES_PER_ITERATION = 10


# ============================================================
# MULTIPROCESS SELF-PLAY
# ============================================================

# Start with one worker per physical core. Benchmark this later against
# 4 / 6 / 8 / 12 on the actual machine.
NUM_SELF_PLAY_WORKERS = 6

# One synchronous MCTS game can have only one outstanding NN request,
# so a batch cannot exceed the number of active workers.
GPU_MAX_BATCH_SIZE = NUM_SELF_PLAY_WORKERS

GPU_BATCH_WAIT_MS = 0.5

SELF_PLAY_STARTUP_TIMEOUT_S = 180.0
SELF_PLAY_GAME_RESULT_TIMEOUT_S = 900.0
SELF_PLAY_SHUTDOWN_TIMEOUT_S = 30.0


# ============================================================
# MCTS V5 ADAPTIVE SEARCH
# ============================================================

SIMULATIONS = 400

ADAPTIVE_SIMULATIONS = True

MIN_SIMULATIONS = 80

CHECK_INTERVAL = 20

TARGET_VISITS_PER_ACTION = 20.0

SINGLE_ACTION_SIMULATIONS = 4

STABILITY_CHECKS = 3

C_PUCT = 3.0

DIRICHLET_ALPHA = 0.3

DIRICHLET_EPSILON = 0.25

MAX_GAME_STEPS = 300


# ============================================================
# TRAINING CONFIG
# ============================================================

BATCH_SIZE = 256

TRAINING_RATIO = 1.5

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 0.0

GRAD_CLIP = 1.0

REPLAY_CAPACITY = 500_000

CHECKPOINT_EVERY_GAMES = 100


# ============================================================
# SEEDS / TRAIN-VALIDATION SPLIT
# ============================================================

BASE_SEED = 10000

DYNAMIC_SEEDING = True

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


# ============================================================
# CHECKPOINT HELPERS
# ============================================================


def extract_model_state_dict(
    checkpoint,
):
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

    model = SplendorNetwork()

    model.load_state_dict(
        extract_model_state_dict(
            checkpoint
        ),
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


def save_inference_snapshot(
    model,
    path,
    *,
    games_played,
):
    """
    Save only what the GPU inference process needs.

    Write to a temporary file first, then atomically replace the
    previous snapshot so a partially written checkpoint is never
    observed.
    """

    directory = os.path.dirname(
        path
    )

    if directory:
        os.makedirs(
            directory,
            exist_ok=True,
        )

    temp_path = (
        path
        + ".tmp"
    )

    state_dict_cpu = {
        key:
            value.detach()
            .cpu()
            .clone()

        for key, value
        in model.state_dict().items()
    }

    torch.save(
        {
            "model_state_dict":
                state_dict_cpu,

            "games_played":
                int(
                    games_played
                ),

            "trainer_version":
                6,
        },
        temp_path,
    )

    os.replace(
        temp_path,
        path,
    )


# ============================================================
# OPTIMIZER
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

    optimizer_state = checkpoint.get(
        "optimizer_state_dict"
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
# RICH REPLAY
# ============================================================


def load_rich_replay_buffer(
    replay_path,
):
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

        replay_buffer.buffer = data[
            "buffer"
        ]

        replay_buffer.position = int(
            data[
                "position"
            ]
        )

        replay_buffer.games = data[
            "games"
        ]

        replay_buffer.game_sample_counts = data[
            "game_sample_counts"
        ]

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


def create_fresh_replay_buffer():
    metadata = {
        "dataset_family":
            "training_v6",

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
            "self_play_mcts_v6_multiprocess",

        # Search algorithm is still V5 adaptive pruning.
        "search_variant":
            "mcts_v5_pruning",

        "self_play_execution":
            "multiprocess_cross_game_gpu_batching",
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
        "Created fresh Model 4 / V6 "
        "multiprocess replay buffer."
    )

    print(
        "Replay capacity:",
        f"{REPLAY_CAPACITY:,}",
    )

    return replay_buffer


# ============================================================
# RESUME COUNTERS
# ============================================================


def infer_games_played_from_checkpoint(
    checkpoint,
):
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
    highest_game_index = None

    for game in (
        replay_buffer
        .games
        .values()
    ):
        game_index = game.get(
            "game_index"
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


# ============================================================
# SELF-PLAY CONFIG / JOB FACTORY
# ============================================================


def make_worker_config(
    *,
    generating_model_id,
):
    return SelfPlayWorkerConfig(
        search=SelfPlaySearchConfig(
            simulations=SIMULATIONS,
            min_simulations=MIN_SIMULATIONS,
            check_interval=CHECK_INTERVAL,
            target_visits_per_action=(
                TARGET_VISITS_PER_ACTION
            ),
            single_action_simulations=(
                SINGLE_ACTION_SIMULATIONS
            ),
            stability_checks=(
                STABILITY_CHECKS
            ),
            c_puct=C_PUCT,
            dirichlet_alpha=(
                DIRICHLET_ALPHA
            ),
            dirichlet_epsilon=(
                DIRICHLET_EPSILON
            ),
            max_game_steps=(
                MAX_GAME_STEPS
            ),
        ),
        replay_capacity=10_000,
        request_timeout_s=(
            SELF_PLAY_GAME_RESULT_TIMEOUT_S
        ),
        request_put_timeout_s=30.0,
        model_generation=4,
        model_checkpoint_label=(
            generating_model_id
        ),
        # Every job overrides this with its deterministic split.
        split="train",
        return_samples=True,
    )


def make_job_factory(
    *,
    starting_attempt_index,
    generating_model_id,
    generating_model_self_play_games,
    run_start_checkpoint,
):
    """
    Produce deterministic V5-compatible seeds, whole-game splits,
    and replay metadata independent of completion order.
    """

    def build_job(
        offset,
        default_game_id,
        default_seed,
    ):
        del default_game_id
        del default_seed

        game_index = (
            int(
                starting_attempt_index
            )
            + int(
                offset
            )
        )

        game_seed = resolve_game_seed(
            base_seed=BASE_SEED,
            game_index=game_index,
            dynamic_seeding=(
                DYNAMIC_SEEDING
            ),
        )

        if game_seed is None:
            raise RuntimeError(
                "V6 multiprocess self-play currently "
                "requires BASE_SEED to be an integer."
            )

        game_split = choose_game_split(
            seed=game_seed,
            game_index=game_index,
            validation_fraction=(
                VALIDATION_FRACTION
            ),
            split_seed=(
                SPLIT_SEED
            ),
        )

        return SelfPlayGameJob(
            # Keep job id equal to attempted-game index. Completion
            # order may differ, but metadata remains deterministic.
            game_id=(
                game_index
            ),
            seed=int(
                game_seed
            ),
            split=(
                game_split
            ),
            extra_game_metadata={
                "game_index":
                    int(
                        game_index
                    ),

                "validation_fraction":
                    VALIDATION_FRACTION,

                "split_seed":
                    SPLIT_SEED,

                "model_name":
                    "Model4",

                "model_id":
                    generating_model_id,

                "self_play_games_at_generation":
                    int(
                        generating_model_self_play_games
                    ),

                "run_start_checkpoint":
                    run_start_checkpoint,

                "trainer_version":
                    6,

                "self_play_execution":
                    "multiprocess_cross_game_gpu_batching",

                "num_self_play_workers":
                    NUM_SELF_PLAY_WORKERS,
            },
        )

    return build_job


# ============================================================
# V6 TRAINING LOOP
# ============================================================


def run_training_v6(
    *,
    model,
    optimizer,
    replay_buffer,
    active_checkpoint_path,
    starting_games_played,
    starting_games_attempted,
):
    history = []

    games_played = int(
        starting_games_played
    )

    games_attempted = int(
        starting_games_attempted
    )

    if CHECKPOINT_EVERY_GAMES is not None:
        next_checkpoint = (
            (
                games_played
                // CHECKPOINT_EVERY_GAMES
            )
            + 1
        ) * CHECKPOINT_EVERY_GAMES

    else:
        next_checkpoint = None

    for iteration in range(
        NUM_ITERATIONS
    ):
        iteration_start = (
            time.perf_counter()
        )

        generating_model_self_play_games = int(
            games_played
        )

        generating_model_id = (
            f"Model4_"
            f"{generating_model_self_play_games}_games"
        )

        print()
        print(
            "=" * 78
        )
        print(
            f"V6 ITERATION {iteration + 1}/"
            f"{NUM_ITERATIONS}"
        )
        print(
            "=" * 78
        )
        print(
            "Generating model:",
            generating_model_id,
        )
        print(
            "Attempt index start:",
            games_attempted,
        )

        # ----------------------------------------------------
        # Snapshot current model for child GPU inference
        # ----------------------------------------------------

        model.eval()

        save_inference_snapshot(
            model,
            INFERENCE_SNAPSHOT_PATH,
            games_played=games_played,
        )

        # The parent does no neural training during self-play.
        # Move its model + optimizer moments off CUDA so the child
        # inference process has as much GPU memory as possible.
        cpu_device = torch.device(
            "cpu"
        )

        model.to(
            cpu_device
        )

        move_optimizer_state_to_device(
            optimizer,
            cpu_device,
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ----------------------------------------------------
        # Parallel self-play
        # ----------------------------------------------------

        worker_config = make_worker_config(
            generating_model_id=(
                generating_model_id
            )
        )

        coordinator_config = (
            MultiprocessSelfPlayCoordinatorConfig(
                num_workers=(
                    NUM_SELF_PLAY_WORKERS
                ),
                checkpoint_path=(
                    INFERENCE_SNAPSHOT_PATH
                ),
                worker_config=(
                    worker_config
                ),
                device=str(
                    device
                ),
                max_batch_size=(
                    GPU_MAX_BATCH_SIZE
                ),
                batch_wait_ms=(
                    GPU_BATCH_WAIT_MS
                ),
                startup_timeout_s=(
                    SELF_PLAY_STARTUP_TIMEOUT_S
                ),
                game_result_timeout_s=(
                    SELF_PLAY_GAME_RESULT_TIMEOUT_S
                ),
                shutdown_timeout_s=(
                    SELF_PLAY_SHUTDOWN_TIMEOUT_S
                ),
            )
        )

        coordinator = (
            MultiprocessSelfPlayCoordinator(
                replay_buffer=(
                    replay_buffer
                ),
                config=(
                    coordinator_config
                ),
                mp_context=(
                    mp.get_context(
                        "spawn"
                    )
                ),
            )
        )

        job_factory = make_job_factory(
            starting_attempt_index=(
                games_attempted
            ),
            generating_model_id=(
                generating_model_id
            ),
            generating_model_self_play_games=(
                generating_model_self_play_games
            ),
            run_start_checkpoint=(
                active_checkpoint_path
            ),
        )

        def progress_callback(
            progress,
        ):
            print(
                "Self-play "
                f"{progress['committed_games']}/"
                f"{progress['requested_games']} "
                f"- in flight "
                f"{progress['in_flight_games']} "
                f"- {progress['latest_positions']} positions "
                f"- {progress['games_per_hour']:.2f} games/hour "
                f"- replay {len(replay_buffer):,}"
            )

        self_play_summary = (
            coordinator.run_block(
                num_games=(
                    SELF_PLAY_GAMES_PER_ITERATION
                ),
                # job_factory owns the actual seed assignment.
                # This value is only a required coordinator default.
                seed_start=(
                    BASE_SEED
                    + games_attempted
                ),
                game_id_start=(
                    games_attempted
                ),
                progress_callback=(
                    progress_callback
                ),
                job_factory=(
                    job_factory
                ),
            )
        )

        # Every submitted game succeeded or run_block would raise.
        games_attempted += int(
            self_play_summary[
                "submitted_games"
            ]
        )

        games_played += int(
            self_play_summary[
                "committed_games"
            ]
        )

        # ----------------------------------------------------
        # Restore parent training state to training device
        # ----------------------------------------------------

        model.to(
            device
        )

        move_optimizer_state_to_device(
            optimizer,
            device,
        )

        # ----------------------------------------------------
        # New train/validation positions this iteration
        # ----------------------------------------------------

        split_games = (
            self_play_summary.get(
                "split_games",
                {},
            )
        )

        split_positions = (
            self_play_summary.get(
                "split_positions",
                {},
            )
        )

        train_games_this_iteration = int(
            split_games.get(
                "train",
                0,
            )
        )

        val_games_this_iteration = int(
            split_games.get(
                "val",
                0,
            )
        )

        train_positions_this_iteration = int(
            split_positions.get(
                "train",
                0,
            )
        )

        val_positions_this_iteration = int(
            split_positions.get(
                "val",
                0,
            )
        )

        # ----------------------------------------------------
        # Search diagnostics
        # ----------------------------------------------------

        print()
        print(
            "V6 self-play wall time:",
            f"{self_play_summary['wall_seconds']:.2f}s",
        )
        print(
            "V6 self-play throughput:",
            f"{self_play_summary['games_per_hour']:.2f} games/hour",
        )
        print(
            "V6 average game positions:",
            f"{self_play_summary['average_positions_per_game']:.2f}",
        )
        print(
            "MCTS average simulations:",
            self_play_summary.get(
                "average_actual_simulations"
            ),
        )

        savings = self_play_summary.get(
            "simulation_savings_fraction"
        )

        if savings is not None:
            print(
                "MCTS simulation savings:",
                f"{100.0 * savings:.2f}%",
            )

        print(
            "MCTS stop reasons:",
            self_play_summary.get(
                "stop_reasons"
            ),
        )

        gpu_stats = (
            self_play_summary.get(
                "gpu_server"
            )
            or {}
        )

        print(
            "GPU average batch size:",
            gpu_stats.get(
                "average_batch_size"
            ),
        )
        print(
            "GPU max batch size:",
            gpu_stats.get(
                "max_observed_batch_size"
            ),
        )
        print(
            "GPU inference positions/sec:",
            gpu_stats.get(
                "inference_positions_per_second"
            ),
        )

        # ----------------------------------------------------
        # Current replay split stats
        # ----------------------------------------------------

        train_stats = get_split_stats(
            replay_buffer=replay_buffer,
            split="train",
        )

        val_stats = get_split_stats(
            replay_buffer=replay_buffer,
            split="val",
        )

        print(
            "Iteration split:"
        )
        print(
            "  train games/positions:",
            train_games_this_iteration,
            train_positions_this_iteration,
        )
        print(
            "  val games/positions:",
            val_games_this_iteration,
            val_positions_this_iteration,
        )

        # ----------------------------------------------------
        # Training steps
        # ----------------------------------------------------

        if (
            train_positions_this_iteration
            > 0
        ):
            training_steps = max(
                1,
                round(
                    train_positions_this_iteration
                    * TRAINING_RATIO
                    / BATCH_SIZE
                ),
            )

        else:
            training_steps = 0

        print(
            "Number of Training Steps:",
            training_steps,
        )

        training_results = None
        validation_results = None

        if training_steps > 0:
            training_results = train_network(
                model=model,
                replay_buffer=(
                    replay_buffer
                ),
                optimizer=optimizer,
                batch_size=BATCH_SIZE,
                training_steps=(
                    training_steps
                ),
                scheduler=None,
                split="train",
                grad_clip=GRAD_CLIP,
            )

            history.append(
                training_results
            )

            print(
                "Training loss:",
                f"{training_results['average_total_loss']:.4f}",
            )
            print(
                "Training policy loss:",
                f"{training_results['average_policy_loss']:.4f}",
            )
            print(
                "Training WDL loss:",
                f"{training_results['average_wdl_loss']:.4f}",
            )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        enough_validation_data = (
            val_stats[
                "games"
            ]
            >= MIN_VALIDATION_GAMES
            and
            val_stats[
                "positions"
            ]
            >= MIN_VALIDATION_POSITIONS
        )

        if enough_validation_data:
            validation_results = validate_network(
                model=model,
                replay_buffer=replay_buffer,
                batch_size=BATCH_SIZE,
                validation_steps=(
                    VALIDATION_STEPS
                ),
                split="val",
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
                f"{MIN_VALIDATION_GAMES} games "
                f"- {val_stats['positions']} / "
                f"{MIN_VALIDATION_POSITIONS} positions"
            )

        # ----------------------------------------------------
        # Scheduled full model/replay checkpoints
        # ----------------------------------------------------

        if CHECKPOINT_EVERY_GAMES is not None:
            (
                next_checkpoint,
                checkpoint_saved,
            ) = save_model_if_needed(
                model=model,
                optimizer=optimizer,
                games_played=games_played,
                history=history,
                checkpoint_every_games=(
                    CHECKPOINT_EVERY_GAMES
                ),
                next_checkpoint=(
                    next_checkpoint
                ),
                checkpoint_dir=(
                    OUTPUT_CHECKPOINT_DIR
                ),
                replay_buffer=(
                    replay_buffer
                ),
                scheduler=None,
            )

            if checkpoint_saved:
                replay_buffer.save(
                    OUTPUT_REPLAY_PATH
                )

        # Save current trained weights now so the next iteration's
        # inference server receives the just-updated network.
        save_inference_snapshot(
            model,
            INFERENCE_SNAPSHOT_PATH,
            games_played=games_played,
        )

        iteration_seconds = (
            time.perf_counter()
            - iteration_start
        )

        print()
        print(
            "-" * 78
        )
        print(
            f"Iteration {iteration + 1} complete"
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
            "Replay positions:",
            f"{len(replay_buffer):,}",
        )
        print(
            "Iteration time:",
            f"{iteration_seconds:.2f}s",
        )
        print(
            "-" * 78
        )

    # ========================================================
    # FINAL SAVE
    # ========================================================

    os.makedirs(
        OUTPUT_CHECKPOINT_DIR,
        exist_ok=True,
    )

    final_checkpoint_path = (
        f"{OUTPUT_CHECKPOINT_DIR}/"
        f"model_{games_played}_games_last.pt"
    )

    save_checkpoint(
        path=final_checkpoint_path,
        model=model,
        optimizer=optimizer,
        games_played=games_played,
        history=history,
    )

    replay_buffer.save(
        OUTPUT_REPLAY_PATH
    )

    print(
        "Final replay buffer saved:",
        OUTPUT_REPLAY_PATH,
    )

    print(
        "Final checkpoint saved:",
        final_checkpoint_path,
    )

    return history


# ============================================================
# MAIN
# ============================================================


def main():
    os.makedirs(
        OUTPUT_CHECKPOINT_DIR,
        exist_ok=True,
    )

    replay_directory = os.path.dirname(
        OUTPUT_REPLAY_PATH
    )

    if replay_directory:
        os.makedirs(
            replay_directory,
            exist_ok=True,
        )

    print(
        "Device:",
        device,
    )

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
                replay_buffer=(
                    replay_buffer
                ),
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

    optimizer = create_optimizer(
        model
    )

    optimizer_restored = (
        maybe_restore_optimizer(
            optimizer,
            checkpoint,
        )
    )

    print()
    print(
        "=" * 78
    )

    if RESUME_TRAINING:
        print(
            "MODEL 4 + V6 MULTIPROCESS SELF-PLAY RESUME MODE"
        )

    else:
        print(
            "MODEL 4 + V6 MULTIPROCESS SELF-PLAY FRESH MODE"
        )

    print(
        "=" * 78
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
        "Optimizer restored:",
        optimizer_restored,
    )
    print(
        "Replay positions:",
        f"{len(replay_buffer):,}",
    )
    print(
        "CPU self-play workers:",
        NUM_SELF_PLAY_WORKERS,
    )
    print(
        "GPU max batch size:",
        GPU_MAX_BATCH_SIZE,
    )
    print(
        "GPU batch wait ms:",
        GPU_BATCH_WAIT_MS,
    )
    print(
        "MCTS simulations hard cap:",
        SIMULATIONS,
    )
    print(
        "MCTS minimum simulations:",
        MIN_SIMULATIONS,
    )
    print(
        "Target visits/action:",
        TARGET_VISITS_PER_ACTION,
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
        "=" * 78
    )
    print()

    history = run_training_v6(
        model=model,
        optimizer=optimizer,
        replay_buffer=replay_buffer,
        active_checkpoint_path=(
            active_checkpoint_path
        ),
        starting_games_played=(
            starting_games_played
        ),
        starting_games_attempted=(
            starting_games_attempted
        ),
    )

    print()
    print(
        "=" * 78
    )
    print(
        "MODEL 4 + V6 MULTIPROCESS TRAINING RUN COMPLETE"
    )
    print(
        "=" * 78
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


if __name__ == "__main__":
    mp.freeze_support()
    main()
