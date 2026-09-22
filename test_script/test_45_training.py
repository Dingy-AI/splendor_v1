import torch

from splendor_v1.env.env import SplendorEnv

from splendor_v1.network.model_2_attention import SplendorNetwork
# ^ change this import to your actual model class

from splendor_v1.training_v2.replay_buffer import (
    ReplayBuffer
)

from splendor_v1.training_v2.train import (
    run_training
)


# ============================================================
# ENVIRONMENT
# ============================================================

env = SplendorEnv()


# ============================================================
# MODEL
# ============================================================

model = SplendorNetwork()


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

model = model.to(
    device
)


# ============================================================
# OPTIONAL: LOAD MODEL 2 CHECKPOINT
# ============================================================

checkpoint_path = (
    "checkpoints/"
    "gen_2/gen_2_h12.pt"
)

checkpoint = torch.load(
    checkpoint_path,
    map_location=device,
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)


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
    },
)


# ============================================================
# SMALL TEST RUN
# ============================================================

history = run_training(

    env=env,
    model=model,
    optimizer=optimizer,
    replay_buffer=replay_buffer,

    # --------------------------------------------------------
    # SMALL INTEGRATION TEST
    # --------------------------------------------------------

    num_iterations=1,
    self_play_games_per_iteration=2,

    simulations=20,
    batch_size=32,
    training_ratio=1.5,

    # --------------------------------------------------------
    # CHECKPOINTS
    # --------------------------------------------------------

    checkpoint_every_games=None,

    checkpoint_dir=(
        "splendor_v1/training_v2/"
        "checkpoints_test"
    ),

    starting_games_played=0,

    # --------------------------------------------------------
    # SELF PLAY
    # --------------------------------------------------------

    seed=10000,
    dynamic_seeding=True,

    teacher_mode=False,

    temperature=1.0,
    add_root_noise=True,

    model_generation=2,
    action_space_version=1,

    # --------------------------------------------------------
    # TRAIN / VALIDATION
    # --------------------------------------------------------

    validation_fraction=0.10,
    split_seed=20260917,

    train_sample_split="train",
    val_sample_split="val",

    min_validation_games=5,
    min_validation_positions=256,

    validation_steps=10,
    validation_batch_size=32,

    # --------------------------------------------------------
    # SAVE RICH REPLAY
    # --------------------------------------------------------

    replay_buffer_path=(
        "splendor_v1/training_v2/data/"
        "test_model2_replay.pkl"
    ),
)

print()
print("Training V2 test complete.")
print("History:")
print(history)