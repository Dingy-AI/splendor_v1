import torch
import numpy as np
from splendor_v1.env.env import SplendorEnv
from splendor_v1.env.core.constants import OBSERVATION_SIZE

from splendor_v1.env.core.action_constants import ACTION_SPACE_SIZE
from splendor_v1.network.model import SplendorNetwork
from splendor_v1.training.replay_buffer import ReplayBuffer
from splendor_v1.training.train import run_training

from splendor_v1.training.checkpoint import load_checkpoint
def main():

    env = SplendorEnv()

    model = SplendorNetwork(
        OBSERVATION_SIZE,
        ACTION_SPACE_SIZE,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )



    starting_games_played = 0


    checkpoint_path = None

    # checkpoint_path = ("checkpoints/model_4_games.pt")

    if checkpoint_path is not None:

        checkpoint_info = load_checkpoint(
            path=checkpoint_path,
            model=model,
            optimizer=optimizer,
        )

        starting_games_played = (
            checkpoint_info["games_played"]
        )

        print(
            f"Loaded checkpoint: "
        )

    print(f"{starting_games_played} games")

    replay_buffer = ReplayBuffer(
        max_size=100_000,
    )
    # policy_debug_samples = []
    policy_debug_samples = None

    history = run_training(
        env=env,
        model=model,
        optimizer=optimizer,
        replay_buffer=replay_buffer,

        num_iterations=10,
        self_play_games_per_iteration=10,

        simulations=200,

        batch_size=32,
        training_steps=100,

        checkpoint_every_games=10,

        starting_games_played=starting_games_played,
        policy_debug_samples=policy_debug_samples,        
    )

    for i, sample in enumerate(
        policy_debug_samples
    ):
        print(
            f"\n========== SAMPLE {i} =========="
        )

        compare_policy_target_to_prediction(
            model,
            sample,
        )

    summarize_policy_debug_by_game(
        model,
        policy_debug_samples,
    )



    print("\nTraining complete.")
    print(
        f"Replay buffer size: "
        f"{len(replay_buffer)}"
    )

    for iteration, stats in enumerate(
        history,
        start=1,
    ):
        print(
            f"\nIteration {iteration}"
        )
        print(
            f"Loss: "
            f"{stats['loss']:.4f}"
        )
        print(
            f"Policy loss: "
            f"{stats['policy_loss']:.4f}"
        )
        print(
            f"Value loss: "
            f"{stats['value_loss']:.4f}"
        )


#DEBUG  HELPER
def compare_policy_target_to_prediction(
    model,
    sample,
    top_k=10,
):
    observation = sample[
        "observation"
    ]

    target_policy = sample[
        "target_policy"
    ]

    legal_action_ids = sample[
        "legal_action_ids"
    ]

    obs_tensor = torch.as_tensor(
        observation,
        dtype=torch.float32,
    ).unsqueeze(0)

    model.eval()

    with torch.inference_mode():

        policy_logits, _ = model(
            obs_tensor
        )

    legal_ids_tensor = torch.as_tensor(
        legal_action_ids,
        dtype=torch.long,
        device=policy_logits.device,
    )

    legal_logits = policy_logits[
        0,
        legal_ids_tensor,
    ]

    predicted_probs = torch.softmax(
        legal_logits,
        dim=0,
    ).cpu().numpy()

    target_probs = np.array([
        target_policy[action_id]
        for action_id in legal_action_ids
    ])

    print("\nPOLICY TARGET VS PREDICTION")
    print(
        "legal actions:",
        len(legal_action_ids),
    )

    print(
        "target sum:",
        target_probs.sum(),
    )

    print(
        "prediction sum:",
        predicted_probs.sum(),
    )

    print(
        "target max:",
        target_probs.max(),
    )

    print(
        "prediction max:",
        predicted_probs.max(),
    )

    order = np.argsort(
        -target_probs
    )

    print("\nTop target actions:")
    print(
        f"{'ID':>6} "
        f"{'TARGET':>10} "
        f"{'PREDICTED':>10}"
    )

    for i in order[:top_k]:

        print(
            f"{legal_action_ids[i]:>6} "
            f"{target_probs[i]:>10.4f} "
            f"{predicted_probs[i]:>10.4f}"
        )

    l1_error = np.abs(
        target_probs
        - predicted_probs
    ).sum()

    epsilon = 1e-12

    kl_divergence = np.sum(
        target_probs
        * (
            np.log(
                target_probs
                + epsilon
            )
            - np.log(
                predicted_probs
                + epsilon
            )
        )
    )

    target_best_index = np.argmax(
        target_probs
    )

    predicted_best_index = np.argmax(
        predicted_probs
    )

    print(
        "\nL1 error:",
        l1_error,
    )

    print(
        "KL divergence:",
        kl_divergence,
    )

    print(
        "target best action:",
        legal_action_ids[
            target_best_index
        ],
    )

    print(
        "predicted best action:",
        legal_action_ids[
            predicted_best_index
        ],
    )

    print(
        "same best action:",
        target_best_index
        == predicted_best_index,
    )

def summarize_policy_debug_by_game(
    model,
    policy_debug_samples,
):

    game_results = {}

    model.eval()

    for sample in policy_debug_samples:

        game_index = sample["game_index"]

        observation = sample["observation"]
        target_policy = sample["target_policy"]
        legal_action_ids = sample["legal_action_ids"]

        obs_tensor = torch.as_tensor(
            observation,
            dtype=torch.float32,
        ).unsqueeze(0)

        with torch.inference_mode():
            policy_logits, _ = model(
                obs_tensor
            )

        legal_ids_tensor = torch.as_tensor(
            legal_action_ids,
            dtype=torch.long,
            device=policy_logits.device,
        )

        legal_logits = policy_logits[
            0,
            legal_ids_tensor,
        ]

        predicted_probs = torch.softmax(
            legal_logits,
            dim=0,
        ).cpu().numpy()

        target_probs = np.array([
            target_policy[action_id]
            for action_id in legal_action_ids
        ])

        # L1
        l1 = np.abs(
            target_probs
            - predicted_probs
        ).sum()

        # KL
        epsilon = 1e-12

        kl = np.sum(
            target_probs
            * (
                np.log(target_probs + epsilon)
                - np.log(predicted_probs + epsilon)
            )
        )

        # Best action
        teacher_best_index = np.argmax(
            target_probs
        )

        predicted_best_index = np.argmax(
            predicted_probs
        )

        same_best = (
            teacher_best_index
            == predicted_best_index
        )

        teacher_best_target = target_probs[
            teacher_best_index
        ]

        teacher_best_prediction = predicted_probs[
            teacher_best_index
        ]

        teacher_best_ratio = (
            teacher_best_prediction
            / teacher_best_target
        )

        if game_index not in game_results:
            game_results[game_index] = {
                "l1": [],
                "kl": [],
                "same_best": [],
                "teacher_best_ratio": [],
            }

        game_results[
            game_index
        ]["l1"].append(l1)

        game_results[
            game_index
        ]["kl"].append(kl)

        game_results[
            game_index
        ]["same_best"].append(same_best)

        game_results[
            game_index
        ]["teacher_best_ratio"].append(
            teacher_best_ratio
        )

    print(
        "\n"
        "========== POLICY DEBUG BY GAME =========="
    )

    print(
        f"{'GAME':>5} "
        f"{'N':>5} "
        f"{'L1':>8} "
        f"{'KL':>8} "
        f"{'BEST%':>8} "
        f"{'RATIO':>8} "
        f"{'<0.5':>8} "
        f"{'<0.25':>8} "
        f"{'<0.10':>8}"
    )

    for game_index in sorted(
        game_results
    ):

        results = game_results[
            game_index
        ]

        l1 = np.array(
            results["l1"]
        )

        kl = np.array(
            results["kl"]
        )

        same_best = np.array(
            results["same_best"]
        )

        ratios = np.array(
            results["teacher_best_ratio"]
        )

        print(
            f"{game_index:>5} "
            f"{len(l1):>5} "
            f"{l1.mean():>8.3f} "
            f"{kl.mean():>8.3f} "
            f"{same_best.mean() * 100:>7.1f}% "
            f"{ratios.mean():>8.3f} "
            f"{(ratios < .50).mean() * 100:>7.1f}% "
            f"{(ratios < .25).mean() * 100:>7.1f}% "
            f"{(ratios < .10).mean() * 100:>7.1f}%"
        )


if __name__ == "__main__":
    main()