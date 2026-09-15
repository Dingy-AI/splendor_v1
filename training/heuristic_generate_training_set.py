from splendor_v1.training.heuristic_replay_generator import HeuristicReplayGenerator
from splendor_v1.env.env import SplendorEnv
from splendor_v1.training.replay_buffer import ReplayBuffer
from splendor_v1.agents.heuristic_agent_6 import HeuristicAgent6

import pickle
import os


# ============================================================
# CONFIG
# ============================================================

OUTPUT_PATH = (
    "splendor_v1/training/data/"
    "h6_replay_data.pkl"
)

NUM_GAMES = 1000
SAVE_EVERY = 10


# ============================================================
# SETUP
# ============================================================

env = SplendorEnv()

agent = HeuristicAgent6(
    num_rollouts=28,
)

replay_buffer = ReplayBuffer(
    500_000
)

generator = HeuristicReplayGenerator(
    env=env,
    agent=agent,
    replay_buffer=replay_buffer,
)


# ============================================================
# SAVE HELPER
# ============================================================

def save_replay_buffer(
    path,
    replay_buffer,
    games_completed,
):

    data = {
        "capacity": replay_buffer.capacity,
        "buffer": replay_buffer.buffer,
        "position": replay_buffer.position,

        # Useful metadata for resuming / debugging
        "games_completed": games_completed,
    }

    # Save to temporary file first so a crash during
    # pickle.dump does not corrupt the previous save.
    temp_path = path + ".tmp"

    with open(temp_path, "wb") as f:
        pickle.dump(
            data,
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    os.replace(
        temp_path,
        path,
    )

    print(
        f"Saved after {games_completed} games "
        f"- {len(replay_buffer.buffer):,} positions"
    )


# ============================================================
# GENERATE GAMES
# ============================================================

for game in range(NUM_GAMES):

    num_positions = generator.generate_game(
        seed=game,
    )

    games_completed = game + 1

    print(
        f"Game {games_completed}/{NUM_GAMES} "
        f"- {num_positions} positions "
        f"- buffer size: {len(replay_buffer.buffer):,}"
    )

    # --------------------------------------------------------
    # SAVE EVERY 10 GAMES
    # --------------------------------------------------------

    if games_completed % SAVE_EVERY == 0:

        save_replay_buffer(
            OUTPUT_PATH,
            replay_buffer,
            games_completed,
        )


# ============================================================
# FINAL SAVE
# ============================================================

save_replay_buffer(
    OUTPUT_PATH,
    replay_buffer,
    NUM_GAMES,
)


print(
    f"Finished generating {NUM_GAMES} games."
)