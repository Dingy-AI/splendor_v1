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

SIMULATIONS = 800
REPLAY_CAPACITY = 500_000
SAVE_EVERY_SUCCESSFUL_GAMES = 10

# Fixed seed shard. Python-style half-open range:
# START_SEED = 0, END_SEED = 100 -> seeds 0..99
START_SEED = 400
END_SEED = 500

# False = start shard from scratch.
# True  = resume this exact shard from its saved replay file.
START_FROM_EXISTING = False

OUTPUT_PATH = (
    "splendor_v1/training/data/"
    f"model_replay_buffer_m1_{SIMULATIONS}sim_"
    f"seed_{START_SEED:04d}_{END_SEED - 1:04d}.pkl"
)


def load_model(checkpoint_path, device):

    model = SplendorNetwork().to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)

    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    return model


def create_fresh_replay_buffer():

    replay_buffer = ReplayBuffer(
        REPLAY_CAPACITY
    )

    games_completed = 0
    seeds_attempted = 0
    failed_seeds = []
    next_seed = START_SEED

    return (
        replay_buffer,
        games_completed,
        seeds_attempted,
        failed_seeds,
        next_seed,
    )


def load_replay_buffer(path):

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Replay buffer does not exist: {path}"
        )

    with open(path, "rb") as f:
        data = pickle.load(f)

    saved_start_seed = data.get(
        "start_seed",
        START_SEED,
    )

    saved_end_seed = data.get(
        "end_seed",
        END_SEED,
    )

    if (
        saved_start_seed != START_SEED
        or saved_end_seed != END_SEED
    ):
        raise ValueError(
            "Replay shard does not match configured seed range.\n"
            f"File range:   [{saved_start_seed}, {saved_end_seed})\n"
            f"Config range: [{START_SEED}, {END_SEED})"
        )

    replay_buffer = ReplayBuffer(
        data["capacity"]
    )

    replay_buffer.buffer = data["buffer"]
    replay_buffer.position = data["position"]

    games_completed = data.get(
        "games_completed",
        0,
    )

    seeds_attempted = data.get(
        "seeds_attempted",
        0,
    )

    failed_seeds = data.get(
        "failed_seeds",
        [],
    )

    next_seed = data.get(
        "next_seed",
        START_SEED,
    )

    return (
        replay_buffer,
        games_completed,
        seeds_attempted,
        failed_seeds,
        next_seed,
    )


def save_replay_buffer(
    replay_buffer,
    output_path,
    games_completed,
    seeds_attempted,
    failed_seeds,
    next_seed,
):

    data = {
        "capacity": replay_buffer.capacity,
        "buffer": replay_buffer.buffer,
        "position": replay_buffer.position,
        "start_seed": START_SEED,
        "end_seed": END_SEED,
        "games_completed": games_completed,
        "seeds_attempted": seeds_attempted,
        "failed_seeds": failed_seeds,
        "next_seed": next_seed,
        "simulations": SIMULATIONS,
        "checkpoint_path": CHECKPOINT_PATH,
    }

    output_directory = os.path.dirname(
        output_path
    )

    if output_directory:
        os.makedirs(
            output_directory,
            exist_ok=True,
        )

    temp_path = output_path + ".tmp"

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


def main():

    if END_SEED <= START_SEED:
        raise ValueError(
            "END_SEED must be greater than START_SEED."
        )

    total_seeds_in_shard = (
        END_SEED - START_SEED
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 70)
    print("MODEL REPLAY GENERATION - FIXED SEED SHARD")
    print("=" * 70)
    print(f"Device:              {device}")
    print(f"Checkpoint:          {CHECKPOINT_PATH}")
    print(f"Simulations:         {SIMULATIONS}")
    print(f"Seed range:          [{START_SEED}, {END_SEED})")
    print(f"Seeds in shard:      {total_seeds_in_shard}")
    print(f"Replay capacity:     {REPLAY_CAPACITY:,}")
    print(f"Output:              {OUTPUT_PATH}")
    print(f"Start from existing: {START_FROM_EXISTING}")
    print()

    env = SplendorEnv()

    model = load_model(
        CHECKPOINT_PATH,
        device,
    )

    print("Loaded frozen model.")

    mcts = MCTS(
        simulations=SIMULATIONS,
        rollout_type="neural",
        selection_type="puct",
        model=model,
    )

    if START_FROM_EXISTING:

        (
            replay_buffer,
            games_completed,
            seeds_attempted,
            failed_seeds,
            next_seed,
        ) = load_replay_buffer(
            OUTPUT_PATH
        )

        print()
        print("Loaded existing shard.")
        print(
            f"Positions:       "
            f"{len(replay_buffer.buffer):,}"
        )
        print(
            f"Games completed: "
            f"{games_completed}"
        )
        print(
            f"Seeds attempted: "
            f"{seeds_attempted}"
        )
        print(
            f"Failed seeds:    "
            f"{len(failed_seeds)}"
        )
        print(
            f"Next seed:       "
            f"{next_seed}"
        )

    else:

        (
            replay_buffer,
            games_completed,
            seeds_attempted,
            failed_seeds,
            next_seed,
        ) = create_fresh_replay_buffer()

        print()
        print("Created fresh replay shard.")
        print("Games completed: 0")
        print("Seeds attempted: 0")
        print(
            f"Next seed:       "
            f"{next_seed}"
        )

    generator = ModelReplayGenerator(
        env=env,
        mcts=mcts,
        replay_buffer=replay_buffer,
        add_root_noise=False,
    )

    start_time = time.perf_counter()

    games_generated_this_run = 0
    positions_generated_this_run = 0

    while next_seed < END_SEED:

        current_seed = next_seed

        # Advance the resume cursor before running the seed.
        next_seed += 1
        seeds_attempted += 1

        try:

            num_positions = generator.generate_game(
                seed=current_seed,
            )

        except RuntimeError as e:

            failed_seeds.append(
                current_seed
            )

            print()
            print(
                f"FAILED seed {current_seed}: {e}"
            )
            print(
                "Skipping this seed. "
                "Shard boundary will not change."
            )
            print()

            save_replay_buffer(
                replay_buffer=replay_buffer,
                output_path=OUTPUT_PATH,
                games_completed=games_completed,
                seeds_attempted=seeds_attempted,
                failed_seeds=failed_seeds,
                next_seed=next_seed,
            )

            continue

        games_completed += 1
        games_generated_this_run += 1

        positions_generated_this_run += (
            num_positions
        )

        elapsed = (
            time.perf_counter()
            - start_time
        )

        average_seconds = (
            elapsed
            / games_generated_this_run
        )

        print(
            f"Seed {current_seed} "
            f"- successful game {games_completed} "
            f"- {num_positions} positions "
            f"- buffer size: "
            f"{len(replay_buffer.buffer):,} "
            f"- avg: "
            f"{average_seconds:.2f}s/game"
        )

        if (
            games_completed
            % SAVE_EVERY_SUCCESSFUL_GAMES
            == 0
        ):

            save_replay_buffer(
                replay_buffer=replay_buffer,
                output_path=OUTPUT_PATH,
                games_completed=games_completed,
                seeds_attempted=seeds_attempted,
                failed_seeds=failed_seeds,
                next_seed=next_seed,
            )

            print(
                f"Saved shard after "
                f"{games_completed} successful games."
            )

    save_replay_buffer(
        replay_buffer=replay_buffer,
        output_path=OUTPUT_PATH,
        games_completed=games_completed,
        seeds_attempted=seeds_attempted,
        failed_seeds=failed_seeds,
        next_seed=next_seed,
    )

    elapsed = (
        time.perf_counter()
        - start_time
    )

    print()
    print("=" * 70)
    print("SHARD GENERATION COMPLETE")
    print("=" * 70)

    print(
        f"Seed range:               "
        f"[{START_SEED}, {END_SEED})"
    )

    print(
        f"Seeds attempted:          "
        f"{seeds_attempted}/{total_seeds_in_shard}"
    )

    print(
        f"Successful games:         "
        f"{games_completed}"
    )

    print(
        f"Failed seeds:             "
        f"{len(failed_seeds)}"
    )

    if failed_seeds:
        print(
            f"Failed seed list:         "
            f"{failed_seeds}"
        )

    print(
        f"Total replay positions:   "
        f"{len(replay_buffer.buffer):,}"
    )

    print(
        f"Games generated this run: "
        f"{games_generated_this_run}"
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
        f"Final next seed:          "
        f"{next_seed}"
    )

    print(
        f"Saved to:                 "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
