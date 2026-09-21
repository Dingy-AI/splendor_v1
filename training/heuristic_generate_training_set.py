from splendor_v1.training.heuristic_replay_generator import HeuristicReplayGenerator
from splendor_v1.env.env import SplendorEnv
from splendor_v1.training.replay_buffer import ReplayBuffer
from splendor_v1.agents.heuristic_agent_16.heuristic_agent_16 import HeuristicAgent16

import pickle
import os
import traceback
import time
import random


start = time.perf_counter()


# ============================================================
# CONFIG
# ============================================================

OUTPUT_PATH = (
    "splendor_v1/training/data/"
    "h16_replay_data_j_game_split_comp_a.pkl"
)

NUM_GAMES = 1000
SAVE_EVERY = 10

# Validation is assigned at the GAME level, not the position level.
VALIDATION_FRACTION = 0.10
SPLIT_SEED = 20260917


# ============================================================
# SETUP
# ============================================================

env = SplendorEnv()

agent = HeuristicAgent16(
    num_rollouts=8,
)

replay_buffer = ReplayBuffer(
    500_000
)

generator = HeuristicReplayGenerator(
    env=env,
    agent=agent,
    replay_buffer=replay_buffer,
)

failed_seeds = []
successful_games = 0

# One record per successfully generated game.
#
# start_index is inclusive.
# end_index is exclusive.
#
# Example:
# {
#     "game_id": 12,
#     "seed": 5012,
#     "start_index": 731,
#     "end_index": 794,
#     "num_positions": 63,
#     "split": "train",
# }
game_records = []

train_game_ids = []
val_game_ids = []


# ============================================================
# SPLIT HELPER
# ============================================================

def choose_game_split(seed):
    """
    Deterministically assign an entire game to train or validation.

    The assignment depends only on the game's seed and SPLIT_SEED, so every
    position from the same game always lands in the same split.

    This avoids trajectory leakage where positions from one game appear in
    both training and validation.
    """

    split_rng = random.Random(
        (seed << 32) ^ SPLIT_SEED
    )

    if split_rng.random() < VALIDATION_FRACTION:
        return "val"

    return "train"


# ============================================================
# SAVE HELPER
# ============================================================

def save_replay_buffer():

    data = {
        # ----------------------------------------------------
        # Existing replay-buffer fields.
        # ----------------------------------------------------
        "capacity": replay_buffer.capacity,
        "buffer": replay_buffer.buffer,
        "position": replay_buffer.position,

        "successful_games": successful_games,
        "failed_seeds": failed_seeds,

        # ----------------------------------------------------
        # New game-level metadata.
        #
        # The buffer remains a flat list of:
        #     (observation, policy, value)
        #
        # so existing replay code can still read the samples.
        #
        # game_records tells us which slice belongs to each
        # game and whether that whole game is train or val.
        # ----------------------------------------------------
        "format_version": 2,
        "validation_fraction": VALIDATION_FRACTION,
        "split_seed": SPLIT_SEED,

        "game_records": game_records,
        "train_game_ids": train_game_ids,
        "val_game_ids": val_game_ids,
    }

    temp_path = OUTPUT_PATH + ".tmp"

    output_dir = os.path.dirname(
        OUTPUT_PATH
    )

    if output_dir:
        os.makedirs(
            output_dir,
            exist_ok=True,
        )

    with open(temp_path, "wb") as f:
        pickle.dump(
            data,
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    os.replace(
        temp_path,
        OUTPUT_PATH,
    )

    train_positions = sum(
        record["num_positions"]
        for record in game_records
        if record["split"] == "train"
    )

    val_positions = sum(
        record["num_positions"]
        for record in game_records
        if record["split"] == "val"
    )

    print(
        f"\nSaved "
        f"{successful_games} successful games "
        f"- {len(replay_buffer.buffer):,} positions "
        f"- {len(failed_seeds)} failed seeds"
    )

    print(
        f"Game-level split: "
        f"{len(train_game_ids)} train games / "
        f"{len(val_game_ids)} val games"
    )

    print(
        f"Position counts: "
        f"{train_positions:,} train / "
        f"{val_positions:,} val\n"
    )


# ============================================================
# GENERATE
# ============================================================

for game in range(NUM_GAMES):

    seed = game + 8000

    # Snapshot the replay buffer before generating the game.
    #
    # If generation fails after partially adding samples, we
    # roll those samples back so no partial trajectory remains.
    buffer_len_before = len(
        replay_buffer.buffer
    )

    buffer_position_before = (
        replay_buffer.position
    )

    try:

        reported_num_positions = (
            generator.generate_game(
                seed=seed,
            )
        )

        buffer_len_after = len(
            replay_buffer.buffer
        )

        actual_num_positions = (
            buffer_len_after
            - buffer_len_before
        )

        # ----------------------------------------------------
        # Safety check.
        #
        # Game-boundary slicing assumes the replay buffer has
        # not started overwriting old samples.
        # ----------------------------------------------------
        if buffer_len_after >= replay_buffer.capacity:
            raise RuntimeError(
                "Replay buffer reached capacity. "
                "Game-boundary metadata assumes no circular "
                "overwrite has occurred. Increase capacity "
                "before continuing generation."
            )

        if (
            reported_num_positions
            != actual_num_positions
        ):
            print(
                f"WARNING seed {seed}: "
                f"generator reported "
                f"{reported_num_positions} positions, "
                f"but buffer grew by "
                f"{actual_num_positions}. "
                f"Using actual buffer growth."
            )

        if actual_num_positions <= 0:
            raise RuntimeError(
                f"Seed {seed} generated no replay positions."
            )

        # ----------------------------------------------------
        # Assign one ID and one split to the ENTIRE game.
        # ----------------------------------------------------
        game_id = successful_games

        split = choose_game_split(
            seed
        )

        record = {
            "game_id": game_id,
            "seed": seed,
            "start_index": buffer_len_before,
            "end_index": buffer_len_after,
            "num_positions": actual_num_positions,
            "split": split,
        }

        game_records.append(
            record
        )

        if split == "train":
            train_game_ids.append(
                game_id
            )
        else:
            val_game_ids.append(
                game_id
            )

        successful_games += 1

        print(
            f"Seed {seed} "
            f"- successful game {successful_games} "
            f"- game_id {game_id} "
            f"- split={split} "
            f"- {actual_num_positions} positions "
            f"- buffer size: "
            f"{len(replay_buffer.buffer):,}"
        )

        # Save every N SUCCESSFUL games.
        if (
            successful_games
            % SAVE_EVERY
            == 0
        ):
            save_replay_buffer()

    except Exception as e:

        # ----------------------------------------------------
        # Roll back any partially-added samples from a failed
        # game. This keeps game boundaries exact.
        #
        # This is safe while the buffer has not wrapped.
        # ----------------------------------------------------
        if (
            len(replay_buffer.buffer)
            > buffer_len_before
        ):
            del replay_buffer.buffer[
                buffer_len_before:
            ]

        replay_buffer.position = (
            buffer_position_before
        )

        failed_seeds.append(
            seed
        )

        print(
            f"\nFAILED seed {seed}"
        )

        print(
            f"{type(e).__name__}: {e}"
        )

        traceback.print_exc()

        print(
            "Rolled back any partial replay samples."
        )

        print(
            "Skipping to next seed...\n"
        )

        continue


# ============================================================
# TIMING
# ============================================================

elapsed = (
    time.perf_counter()
    - start
)

print(
    "Attempted games:",
    NUM_GAMES,
)

print(
    "Successful games:",
    successful_games,
)

print(
    "Total seconds:",
    elapsed,
)

if successful_games > 0:

    print(
        "Seconds/successful game:",
        elapsed / successful_games,
    )

    print(
        "Successful games/hour:",
        successful_games / elapsed * 3600,
    )


# ============================================================
# FINAL SAVE
# ============================================================

save_replay_buffer()


# ============================================================
# FINAL SUMMARY
# ============================================================

print(
    "=" * 70
)

print(
    "GENERATION COMPLETE"
)

print(
    "=" * 70
)

print(
    f"Successful games: "
    f"{successful_games}"
)

print(
    f"Failed games:     "
    f"{len(failed_seeds)}"
)

print(
    f"Failed seeds:     "
    f"{failed_seeds}"
)

print(
    f"Train games:      "
    f"{len(train_game_ids)}"
)

print(
    f"Validation games: "
    f"{len(val_game_ids)}"
)

print(
    f"Total positions:  "
    f"{len(replay_buffer.buffer):,}"
)

train_positions = sum(
    record["num_positions"]
    for record in game_records
    if record["split"] == "train"
)

val_positions = sum(
    record["num_positions"]
    for record in game_records
    if record["split"] == "val"
)

print(
    f"Train positions:  "
    f"{train_positions:,}"
)

print(
    f"Val positions:    "
    f"{val_positions:,}"
)
