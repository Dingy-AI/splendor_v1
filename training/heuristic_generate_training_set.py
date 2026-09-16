from splendor_v1.training.heuristic_replay_generator import HeuristicReplayGenerator
from splendor_v1.env.env import SplendorEnv
from splendor_v1.training.replay_buffer import ReplayBuffer
from splendor_v1.agents.heuristic_agent_12 import HeuristicAgent12

import pickle
import os
import traceback


# ============================================================
# CONFIG
# ============================================================

OUTPUT_PATH = (
    "splendor_v1/training/data/"
    "h12_replay_data.pkl"
)

NUM_GAMES = 1000
SAVE_EVERY = 10


# ============================================================
# SETUP
# ============================================================

env = SplendorEnv()

agent = HeuristicAgent12(
    num_rollouts=16,
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


# ============================================================
# SAVE HELPER
# ============================================================

def save_replay_buffer():

    data = {
        "capacity": replay_buffer.capacity,
        "buffer": replay_buffer.buffer,
        "position": replay_buffer.position,

        "successful_games": successful_games,
        "failed_seeds": failed_seeds,
    }

    temp_path = OUTPUT_PATH + ".tmp"

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

    print(
        f"\nSaved "
        f"{successful_games} successful games "
        f"- {len(replay_buffer.buffer):,} positions "
        f"- {len(failed_seeds)} failed seeds\n"
    )


# ============================================================
# GENERATE
# ============================================================

for game in range(NUM_GAMES):

    seed = game

    try:

        num_positions = generator.generate_game(
            seed=seed,
        )

        successful_games += 1

        print(
            f"Seed {seed} "
            f"- successful game {successful_games} "
            f"- {num_positions} positions "
            f"- buffer size: "
            f"{len(replay_buffer.buffer):,}"
        )

        # Save every 10 SUCCESSFUL games
        if successful_games % SAVE_EVERY == 0:
            save_replay_buffer()

    except Exception as e:

        failed_seeds.append(
            seed
        )

        print(
            f"\nFAILED seed {seed}"
        )

        print(
            f"{type(e).__name__}: {e}"
        )

        # Optional: full stack trace
        traceback.print_exc()

        print(
            "Skipping to next seed...\n"
        )

        # Move on to next game
        continue


# ============================================================
# FINAL SAVE
# ============================================================

save_replay_buffer()


print("=" * 70)
print("GENERATION COMPLETE")
print("=" * 70)

print(
    f"Successful games: {successful_games}"
)

print(
    f"Failed games:     {len(failed_seeds)}"
)

print(
    f"Failed seeds:     {failed_seeds}"
)

print(
    f"Total positions:  "
    f"{len(replay_buffer.buffer):,}"
)