import os
import pickle
import time

import torch

from splendor_v1.env.env import SplendorEnv
from splendor_v1.mcts.mcts import MCTS
from splendor_v1.network.model import SplendorNetwork
from splendor_v1.training.model_replay_generator import ModelReplayGenerator
from splendor_v1.training.replay_buffer import ReplayBuffer


# ============================================================
# CONFIG
# ============================================================

CHECKPOINT_PATH = (
    "checkpoints/heuristic_pretrain/"
    "heuristic_pretrain_best_400sim.pt"
)

OUTPUT_PATH = (
    "splendor_v1/training/data/"
    "model_replay_buffer_m1_600.pkl"
)

NUM_GAMES = 1000
SIMULATIONS = 600
REPLAY_CAPACITY = 500_000

SAVE_EVERY_GAMES = 10


# ------------------------------------------------------------
# REPLAY START MODE
# ------------------------------------------------------------
#
# False:
#     Start with an EMPTY replay buffer.
#     completed_games = 0
#     next_seed = 0
#
# True:
#     Load OUTPUT_PATH and resume using:
#         games_completed
#         next_seed
#
# ------------------------------------------------------------

START_FROM_EXISTING = False


# ============================================================
# HELPERS
# ============================================================

def load_model(
    checkpoint_path,
    device,
):

    model = SplendorNetwork().to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    # Supports:
    #
    # 1. Full checkpoint
    # 2. Raw model.state_dict()

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        state_dict = checkpoint[
            "model_state_dict"
        ]

    else:
        state_dict = checkpoint

    model.load_state_dict(
        state_dict
    )

    # Fixed teacher for entire dataset.
    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    return model


# ============================================================
# LOAD EXISTING REPLAY
# ============================================================

def load_replay_buffer(path):

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Replay buffer does not exist: {path}"
        )

    with open(path, "rb") as f:
        data = pickle.load(f)

    replay_buffer = ReplayBuffer(
        data["capacity"]
    )

    replay_buffer.buffer = data["buffer"]
    replay_buffer.position = data["position"]

    # --------------------------------------------------------
    # Resume metadata
    # --------------------------------------------------------

    games_completed = data.get(
        "games_completed",
        0,
    )

    next_seed = data.get(
        "next_seed",
        games_completed,
    )

    return (
        replay_buffer,
        games_completed,
        next_seed,
    )


# ============================================================
# CREATE FRESH REPLAY
# ============================================================

def create_fresh_replay_buffer():

    replay_buffer = ReplayBuffer(
        REPLAY_CAPACITY
    )

    games_completed = 0
    next_seed = 0

    return (
        replay_buffer,
        games_completed,
        next_seed,
    )


# ============================================================
# SAVE
# ============================================================

def save_replay_buffer(
    replay_buffer,
    output_path,
    games_completed,
    next_seed,
):

    data = {
        "capacity": replay_buffer.capacity,
        "buffer": replay_buffer.buffer,
        "position": replay_buffer.position,

        # Resume information
        "games_completed": games_completed,
        "next_seed": next_seed,
    }

    output_directory = os.path.dirname(
        output_path
    )

    if output_directory:
        os.makedirs(
            output_directory,
            exist_ok=True,
        )

    # Write temporary file first so an interrupted
    # save doesn't destroy the existing replay.
    temp_path = (
        output_path
        + ".tmp"
    )

    with open(temp_path, "wb") as f:

        pickle.dump(
            data,
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    os.replace(
        temp_path,
        output_path,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 70)
    print("MODEL REPLAY GENERATION")
    print("=" * 70)

    print(
        f"Device:              {device}"
    )

    print(
        f"Checkpoint:          {CHECKPOINT_PATH}"
    )

    print(
        f"Target games:        {NUM_GAMES}"
    )

    print(
        f"Simulations:         {SIMULATIONS}"
    )

    print(
        f"Replay capacity:     {REPLAY_CAPACITY:,}"
    )

    print(
        f"Output:              {OUTPUT_PATH}"
    )

    print(
        f"Start from existing: {START_FROM_EXISTING}"
    )

    print()


    # ========================================================
    # ENVIRONMENT
    # ========================================================

    env = SplendorEnv()


    # ========================================================
    # FIXED MODEL
    # ========================================================

    model = load_model(
        CHECKPOINT_PATH,
        device,
    )

    print(
        "Loaded frozen model."
    )


    # ========================================================
    # MCTS TEACHER
    # ========================================================

    mcts = MCTS(
        simulations=SIMULATIONS,
        rollout_type="neural",
        selection_type="puct",
        model=model,
    )


    # ========================================================
    # REPLAY BUFFER
    # ========================================================

    if START_FROM_EXISTING:

        (
            replay_buffer,
            completed_games,
            next_seed,
        ) = load_replay_buffer(
            OUTPUT_PATH
        )

        print()
        print(
            "Loaded existing replay buffer."
        )

        print(
            f"Positions:       "
            f"{len(replay_buffer.buffer):,}"
        )

        print(
            f"Games completed: "
            f"{completed_games}"
        )

        print(
            f"Next seed:       "
            f"{next_seed}"
        )

    else:

        (
            replay_buffer,
            completed_games,
            next_seed,
        ) = create_fresh_replay_buffer()

        print()
        print(
            "Created fresh replay buffer."
        )

        print(
            "Games completed: 0"
        )

        print(
            "Next seed:       0"
        )


    # ========================================================
    # GENERATOR
    # ========================================================

    generator = ModelReplayGenerator(
        env=env,
        mcts=mcts,
        replay_buffer=replay_buffer,

        # Fixed deterministic teacher.
        add_root_noise=False,
    )


    # ========================================================
    # GENERATE
    # ========================================================

    start_time = time.perf_counter()

    positions_generated_this_run = 0
    games_generated_this_run = 0


    while completed_games < NUM_GAMES:

        current_seed = next_seed

        try:

            num_positions = (
                generator.generate_game(
                    seed=current_seed,
                )
            )

        except RuntimeError as e:

            print()
            print(
                f"FAILED seed {current_seed}: "
                f"{e}"
            )

            print(
                "Skipping seed and continuing."
            )

            print()

            next_seed += 1

            continue


        # ----------------------------------------------------
        # Successful game
        # ----------------------------------------------------

        completed_games += 1
        games_generated_this_run += 1

        positions_generated_this_run += (
            num_positions
        )

        next_seed += 1


        elapsed = (
            time.perf_counter()
            - start_time
        )

        average_seconds = (
            elapsed
            / games_generated_this_run
        )


        print(
            f"Game "
            f"{completed_games}/{NUM_GAMES} "
            f"- seed {current_seed} "
            f"- {num_positions} positions "
            f"- buffer size: "
            f"{len(replay_buffer.buffer):,} "
            f"- avg: "
            f"{average_seconds:.2f}s/game"
        )


        # ----------------------------------------------------
        # Periodic save
        # ----------------------------------------------------

        if (
            completed_games
            % SAVE_EVERY_GAMES
            == 0
        ):

            save_replay_buffer(
                replay_buffer=replay_buffer,
                output_path=OUTPUT_PATH,
                games_completed=completed_games,
                next_seed=next_seed,
            )

            print(
                f"Saved after "
                f"{completed_games} "
                f"completed games."
            )


    # ========================================================
    # FINAL SAVE
    # ========================================================

    save_replay_buffer(
        replay_buffer=replay_buffer,
        output_path=OUTPUT_PATH,
        games_completed=completed_games,
        next_seed=next_seed,
    )


    # ========================================================
    # SUMMARY
    # ========================================================

    elapsed = (
        time.perf_counter()
        - start_time
    )

    print()
    print("=" * 70)
    print("GENERATION COMPLETE")
    print("=" * 70)

    print(
        f"Total games:              "
        f"{completed_games}"
    )

    print(
        f"Games generated this run: "
        f"{games_generated_this_run}"
    )

    print(
        f"Total replay positions:   "
        f"{len(replay_buffer.buffer):,}"
    )

    print(
        f"Positions generated now:  "
        f"{positions_generated_this_run:,}"
    )

    if games_generated_this_run > 0:

        print(
            f"Avg positions/game:       "
            f"{positions_generated_this_run / games_generated_this_run:.2f}"
        )

    print(
        f"Elapsed minutes:          "
        f"{elapsed / 60:.2f}"
    )

    print(
        f"Next seed:                "
        f"{next_seed}"
    )

    print(
        f"Saved to:                 "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()