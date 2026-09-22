import os
import torch

from splendor_v1.env.env import SplendorEnv
from splendor_v1.network.model_2_attention import SplendorNetwork

from splendor_v1.training_v2.replay_buffer import ReplayBuffer
from splendor_v1.training_v2.train import run_training

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
    return (
        turn_number < 40
    )

# ============================================================
# MINI TEST CONFIG
# ============================================================

CHECKPOINT_PATH = (
    "checkpoints/"
    "gen_2/gen_2_h12.pt"
)

OUTPUT_CHECKPOINT_DIR = (
    "splendor_v1/training_v2/"
    "model"
)

OUTPUT_REPLAY_PATH = (
    "splendor_v1/training_v2/data/"
    "mini_test_model2_replay.pkl"
)

# Small enough to run quickly, large enough to exercise
# several complete games and optimizer steps.
NUM_ITERATIONS = 1
SELF_PLAY_GAMES_PER_ITERATION = 1
SIMULATIONS = 20
BATCH_SIZE = 32
TRAINING_RATIO = 1.5

# Reproducible but different Splendor setup each game:
# 10000, 10001, 10002, ...
BASE_SEED = 10019
DYNAMIC_SEEDING = True

# Real intended train/validation split.
VALIDATION_FRACTION = 0.10
SPLIT_SEED = 20260917


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)


# ============================================================
# ENVIRONMENT
# ============================================================

env = SplendorEnv()


# ============================================================
# MODEL 2 / G2H12
# ============================================================

model = SplendorNetwork()
model = model.to(device)

print("Loading checkpoint:", CHECKPOINT_PATH)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device,
)

# Support either the normal checkpoint dictionary or a raw
# state_dict so this launcher is convenient for quick tests.
if (
    isinstance(checkpoint, dict)
    and "model_state_dict" in checkpoint
):
    model_state_dict = checkpoint["model_state_dict"]
else:
    model_state_dict = checkpoint

model.load_state_dict(model_state_dict)

print("Model loaded successfully.")


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3,
)


# ============================================================
# REPLAY BUFFER V2
# ============================================================

replay_buffer = ReplayBuffer(
    capacity=500_000,
    metadata={
        "dataset_family": "training_v2",
        "schema_version": 1,
        "model_generation": 2,
        "source_model": "G2H12",
        "source_checkpoint": CHECKPOINT_PATH,
        "run_type": "mini_test",
    },
)


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

os.makedirs(
    OUTPUT_CHECKPOINT_DIR,
    exist_ok=True,
)

os.makedirs(
    os.path.dirname(OUTPUT_REPLAY_PATH),
    exist_ok=True,
)


# ============================================================
# RUN MINI TRAINING TEST
# ============================================================

history = run_training(
    env=env,
    model=model,
    optimizer=optimizer,
    replay_buffer=replay_buffer,

    # --------------------------------------------------------
    # MINI RUN
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

    # No intermediate checkpoint needed for a 4-game smoke test.
    checkpoint_every_games=25,
    checkpoint_dir=OUTPUT_CHECKPOINT_DIR,
    starting_games_played=0,

    # --------------------------------------------------------
    # SELF PLAY
    # --------------------------------------------------------

    seed=BASE_SEED,
    dynamic_seeding=DYNAMIC_SEEDING,
    teacher_mode=False,

    temperature=1.0,
    temperature_fn=splendor_temperature,

    add_root_noise=True,
    root_noise_fn=splendor_root_noise,

    model_generation=2,
    model_name="Blake",
    run_start_checkpoint=CHECKPOINT_PATH,
    action_space_version=1,

    # --------------------------------------------------------
    # TRAIN / VALIDATION
    # --------------------------------------------------------

    validation_fraction=VALIDATION_FRACTION,
    split_seed=SPLIT_SEED,
    train_sample_split="train",
    val_sample_split="val",

    # Validation probably will not activate in only four games,
    # which is expected. These are the real defaults we intend
    # to use once enough held-out data exists.
    min_validation_games=2,
    min_validation_positions=256,
    validation_steps=10,
    validation_batch_size=BATCH_SIZE,

    # --------------------------------------------------------
    # REPLAY SAVE
    # --------------------------------------------------------

    replay_buffer_path=OUTPUT_REPLAY_PATH,

    mcts_kwargs={
        "c_puct": 3.0,
        "dirichlet_alpha": 0.3,
        "dirichlet_epsilon": 0.25,
    },    
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("TRAINING V2 MINI TEST COMPLETE")
print("=" * 70)
print("Replay samples:", len(replay_buffer))
print("Games stored:", len(replay_buffer.games))
print("Replay saved to:", OUTPUT_REPLAY_PATH)
print("Checkpoint directory:", OUTPUT_CHECKPOINT_DIR)
print()
print("Training history:")
print(history)
