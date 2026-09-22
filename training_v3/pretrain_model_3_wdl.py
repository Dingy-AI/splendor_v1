import argparse
import os
import pickle
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F

from splendor_v1.network.model_3_wdl_output import (
    SplendorNetwork,
    WDL_LOSS,
    WDL_DRAW,
    WDL_WIN,
)

from splendor_v1.training_v2.replay_buffer import (
    ReplayBuffer,
)

from splendor_v1.training_v3.train import (
    build_training_batch,
    build_wdl_target,
    train_network,
    validate_network,
)


WDL_NAMES = {
    WDL_LOSS: "LOSS",
    WDL_DRAW: "DRAW",
    WDL_WIN: "WIN",
}


# ============================================================
# CHECKPOINT HELPERS
# ============================================================


def extract_model_state_dict(
    checkpoint,
):
    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        return checkpoint[
            "model_state_dict"
        ]

    if (
        isinstance(checkpoint, dict)
        and checkpoint
        and all(
            torch.is_tensor(value)
            for value in checkpoint.values()
        )
    ):
        return checkpoint

    raise RuntimeError(
        "Checkpoint must either be a raw model state_dict "
        "or contain 'model_state_dict'."
    )


def load_model3(
    checkpoint_path,
    device,
):
    if not os.path.exists(
        checkpoint_path
    ):
        raise FileNotFoundError(
            f"Model 3 checkpoint does not exist: "
            f"{checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
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

    return model, checkpoint


def save_model3_checkpoint(
    path,
    model,
    source_checkpoint,
    replay_path,
    phase,
    phase_metrics,
    config,
):
    output_dir = os.path.dirname(
        path
    )

    if output_dir:
        os.makedirs(
            output_dir,
            exist_ok=True,
        )

    payload = {
        "model_state_dict":
            model.state_dict(),

        "model_generation":
            3,

        "architecture":
            "attention_wdl",

        "source_checkpoint":
            source_checkpoint,

        "pretraining_replay":
            replay_path,

        "pretraining_phase":
            phase,

        "phase_metrics":
            phase_metrics,

        "pretraining_config":
            config,
    }

    temp_path = (
        path + ".tmp"
    )

    torch.save(
        payload,
        temp_path,
    )

    os.replace(
        temp_path,
        path,
    )


# ============================================================
# REPLAY LOADING
# ============================================================


def load_replay_buffer(
    path,
):
    """
    Restore the current rich replay-buffer format.

    Required for WDL pretraining:
        buffer
        games

    Restores current format-3 metadata when present:
        capacity
        position
        metadata
        games
        game_sample_counts
        next_game_id
    """

    if not os.path.exists(
        path
    ):
        raise FileNotFoundError(
            f"Replay buffer does not exist: {path}"
        )

    with open(
        path,
        "rb",
    ) as file:
        data = pickle.load(
            file
        )

    # Support a directly pickled ReplayBuffer too.
    if isinstance(
        data,
        ReplayBuffer,
    ):
        replay_buffer = data

    elif isinstance(
        data,
        dict,
    ):
        if "capacity" not in data:
            raise RuntimeError(
                "Replay file is missing 'capacity'."
            )

        metadata = data.get(
            "metadata",
            {},
        )

        try:
            replay_buffer = ReplayBuffer(
                capacity=int(
                    data["capacity"]
                ),
                metadata=metadata,
            )
        except TypeError:
            # Compatibility fallback if constructor signature
            # does not expose metadata.
            replay_buffer = ReplayBuffer(
                int(
                    data["capacity"]
                )
            )

            replay_buffer.metadata = (
                metadata
            )

        required_fields = (
            "buffer",
            "position",
            "games",
            "game_sample_counts",
        )

        missing = [
            field
            for field in required_fields
            if field not in data
        ]

        if missing:
            raise RuntimeError(
                "Replay is not the rich Model-2/Model-3 "
                "format required for WDL pretraining. "
                f"Missing: {missing}"
            )

        replay_buffer.buffer = data[
            "buffer"
        ]

        replay_buffer.position = int(
            data["position"]
        )

        replay_buffer.games = data[
            "games"
        ]

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

    if len(
        replay_buffer
    ) == 0:
        raise RuntimeError(
            "Replay buffer is empty."
        )

    if not getattr(
        replay_buffer,
        "games",
        None,
    ):
        raise RuntimeError(
            "Replay contains no game metadata. "
            "WDL targets require winner_ids."
        )

    return replay_buffer


# ============================================================
# REPLAY DIAGNOSTICS
# ============================================================


def replay_split_stats(
    replay_buffer,
):
    stats = {
        "train_games": 0,
        "val_games": 0,
        "train_positions": 0,
        "val_positions": 0,
    }

    for (
        game_id,
        game,
    ) in replay_buffer.games.items():

        split = game.get(
            "split"
        )

        surviving_positions = int(
            replay_buffer
            .game_sample_counts
            .get(
                game_id,
                0,
            )
        )

        if surviving_positions <= 0:
            continue

        if split == "train":
            stats[
                "train_games"
            ] += 1
            stats[
                "train_positions"
            ] += (
                surviving_positions
            )

        elif split == "val":
            stats[
                "val_games"
            ] += 1
            stats[
                "val_positions"
            ] += (
                surviving_positions
            )

    return stats


def wdl_distribution(
    replay_buffer,
    split=None,
):
    """
    Count active replay positions by WDL target.
    """

    counts = Counter()

    for sample in replay_buffer.buffer:

        if sample is None:
            continue

        game_id = sample.get(
            "game_id"
        )

        if game_id is None:
            continue

        game = replay_buffer.games.get(
            game_id
        )

        if game is None:
            continue

        if (
            split is not None
            and game.get(
                "split"
            )
            != split
        ):
            continue

        target = build_wdl_target(
            sample,
            replay_buffer,
        )

        counts[
            int(target)
        ] += 1

    total = sum(
        counts.values()
    )

    return {
        "total":
            total,

        "loss":
            counts[
                WDL_LOSS
            ],

        "draw":
            counts[
                WDL_DRAW
            ],

        "win":
            counts[
                WDL_WIN
            ],
    }


def print_wdl_distribution(
    name,
    distribution,
):
    total = distribution[
        "total"
    ]

    print(
        f"{name} WDL positions: {total:,}"
    )

    for key in (
        "loss",
        "draw",
        "win",
    ):
        count = distribution[
            key
        ]

        fraction = (
            count / total
            if total > 0
            else 0.0
        )

        print(
            f"  {key.upper():4s}: "
            f"{count:>8,} "
            f"({fraction:7.2%})"
        )


# ============================================================
# PHASE A: WDL HEAD ONLY
# ============================================================


def freeze_except_wdl_head(
    model,
):
    for (
        name,
        parameter,
    ) in model.named_parameters():

        parameter.requires_grad_(
            name.startswith(
                "wdl_head."
            )
        )


def unfreeze_all(
    model,
):
    for parameter in (
        model.parameters()
    ):
        parameter.requires_grad_(
            True
        )


def train_wdl_head_only(
    model,
    replay_buffer,
    steps,
    batch_size,
    learning_rate,
    weight_decay=0.0,
    split="train",
    log_every=100,
):
    """
    Warm up only the new WDL head.

    The transferred trunk and policy head remain frozen.
    """

    freeze_except_wdl_head(
        model
    )

    optimizer = torch.optim.AdamW(
        model.wdl_head.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    device = next(
        model.parameters()
    ).device

    model.train()

    losses = []
    accuracies = []

    for step in range(
        1,
        steps + 1,
    ):
        batch = replay_buffer.sample(
            batch_size,
            split=split,
        )

        if not batch:
            raise RuntimeError(
                "ReplayBuffer returned an empty "
                "head-warmup batch."
            )

        (
            observations_np,
            _,
            target_wdl_np,
        ) = build_training_batch(
            batch=batch,
            replay_buffer=replay_buffer,
        )

        observations = (
            torch.as_tensor(
                observations_np,
                dtype=torch.float32,
                device=device,
            )
        )

        target_wdl = (
            torch.as_tensor(
                target_wdl_np,
                dtype=torch.long,
                device=device,
            )
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        # The trunk is frozen, so no gradient is needed through it.
        with torch.no_grad():
            features = (
                model._forward_features(
                    observations
                )
            )

        wdl_logits = model.wdl_head(
            features
        )

        loss = F.cross_entropy(
            wdl_logits,
            target_wdl,
        )

        loss.backward()

        optimizer.step()

        with torch.no_grad():
            accuracy = (
                wdl_logits.argmax(
                    dim=-1
                )
                == target_wdl
            ).float().mean().item()

        losses.append(
            loss.item()
        )

        accuracies.append(
            accuracy
        )

        if (
            log_every > 0
            and (
                step == 1
                or step % log_every == 0
                or step == steps
            )
        ):
            recent_count = min(
                log_every,
                len(
                    losses
                ),
            )

            print(
                f"[WDL head] "
                f"step {step:>6}/{steps} "
                f"loss="
                f"{np.mean(losses[-recent_count:]):.4f} "
                f"accuracy="
                f"{np.mean(accuracies[-recent_count:]):.3f}"
            )

    return {
        "steps":
            int(
                steps
            ),

        "average_wdl_loss":
            float(
                np.mean(
                    losses
                )
            ),

        "final_window_wdl_loss":
            float(
                np.mean(
                    losses[
                        -min(
                            log_every,
                            len(
                                losses
                            ),
                        ):
                    ]
                )
            ),

        "average_wdl_accuracy":
            float(
                np.mean(
                    accuracies
                )
            ),
    }


# ============================================================
# VALIDATION PRINTING
# ============================================================


def run_validation(
    model,
    replay_buffer,
    batch_size,
    validation_steps,
):
    results = validate_network(
        model=model,
        replay_buffer=replay_buffer,
        batch_size=batch_size,
        validation_steps=validation_steps,
        split="val",
    )

    if results is None:
        raise RuntimeError(
            "Validation produced no batches. "
            "Make sure the replay contains a 'val' split."
        )

    return results


def print_validation(
    label,
    results,
):
    print()
    print(
        "-" * 70
    )
    print(
        label
    )
    print(
        "-" * 70
    )

    print(
        "Total loss:       ",
        f"{results['average_total_loss']:.4f}",
    )

    print(
        "Policy loss:      ",
        f"{results['average_policy_loss']:.4f}",
    )

    print(
        "Policy KL:        ",
        f"{results['average_policy_kl']:.4f}",
    )

    print(
        "WDL loss:         ",
        f"{results['average_wdl_loss']:.4f}",
    )

    print(
        "WDL accuracy:     ",
        f"{results['average_wdl_accuracy']:.3f}",
    )

    print(
        "Predicted value:  ",
        f"{results['average_predicted_value']:+.4f}",
    )

    print(
        "Target value:     ",
        f"{results['average_target_value']:+.4f}",
    )

    print(
        "Predicted L/D/W:  ",
        (
            f"{results['average_loss_probability']:.3f} / "
            f"{results['average_draw_probability']:.3f} / "
            f"{results['average_win_probability']:.3f}"
        ),
    )

    print(
        "Target L/D/W:     ",
        (
            f"{results['average_target_loss_fraction']:.3f} / "
            f"{results['average_target_draw_fraction']:.3f} / "
            f"{results['average_target_win_fraction']:.3f}"
        ),
    )


# ============================================================
# MAIN
# ============================================================


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Pretrain Model 3's WDL head on an existing "
            "rich Model 2 replay buffer, then optionally "
            "jointly fine-tune the full network."
        )
    )

    parser.add_argument(
        "--checkpoint",
        default=(
            "checkpoints/gen_3/"
            "gen_3_wdl_transfer_g2h12.pt"
        ),
        help=(
            "Transferred Model 3 checkpoint."
        ),
    )

    parser.add_argument(
        "--replay",
        required=True,
        help=(
            "Existing rich replay-buffer .pkl file."
        ),
    )

    parser.add_argument(
        "--output",
        default=(
            "checkpoints/gen_3/"
            "gen_3_wdl_pretrained.pt"
        ),
        help=(
            "Final pretrained Model 3 checkpoint."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--head-steps",
        type=int,
        default=1000,
        help=(
            "Phase A optimizer steps with only wdl_head trainable."
        ),
    )

    parser.add_argument(
        "--head-lr",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--joint-steps",
        type=int,
        default=2000,
        help=(
            "Phase B optimizer steps with the full network trainable. "
            "Set 0 to skip."
        ),
    )

    parser.add_argument(
        "--joint-lr",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--validation-steps",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--log-every",
        type=int,
        default=100,
    )

    args = parser.parse_args()

    if args.head_steps < 0:
        raise ValueError(
            "--head-steps must be >= 0."
        )

    if args.joint_steps < 0:
        raise ValueError(
            "--joint-steps must be >= 0."
        )

    if args.batch_size < 1:
        raise ValueError(
            "--batch-size must be >= 1."
        )

    if args.validation_steps < 1:
        raise ValueError(
            "--validation-steps must be >= 1."
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print(
        "=" * 70
    )
    print(
        "MODEL 3 WDL PRETRAINING"
    )
    print(
        "=" * 70
    )

    print(
        "Device:       ",
        device,
    )

    print(
        "Checkpoint:   ",
        args.checkpoint,
    )

    print(
        "Replay:       ",
        args.replay,
    )

    print(
        "Output:       ",
        args.output,
    )

    print(
        "Batch size:   ",
        args.batch_size,
    )

    print(
        "Head steps:   ",
        args.head_steps,
    )

    print(
        "Joint steps:  ",
        args.joint_steps,
    )

    print()

    # ========================================================
    # LOAD REPLAY
    # ========================================================

    replay_buffer = (
        load_replay_buffer(
            args.replay
        )
    )

    split_stats = (
        replay_split_stats(
            replay_buffer
        )
    )

    print(
        "Replay positions:",
        f"{len(replay_buffer):,}",
    )

    print(
        "Replay games:    ",
        f"{len(replay_buffer.games):,}",
    )

    print(
        "Train games:     ",
        split_stats[
            "train_games"
        ],
    )

    print(
        "Train positions: ",
        f"{split_stats['train_positions']:,}",
    )

    print(
        "Val games:       ",
        split_stats[
            "val_games"
        ],
    )

    print(
        "Val positions:   ",
        f"{split_stats['val_positions']:,}",
    )

    if (
        split_stats[
            "train_positions"
        ]
        <= 0
    ):
        raise RuntimeError(
            "Replay contains no train positions."
        )

    if (
        split_stats[
            "val_positions"
        ]
        <= 0
    ):
        raise RuntimeError(
            "Replay contains no validation positions."
        )

    print()

    print_wdl_distribution(
        "TRAIN",
        wdl_distribution(
            replay_buffer,
            split="train",
        ),
    )

    print()

    print_wdl_distribution(
        "VAL",
        wdl_distribution(
            replay_buffer,
            split="val",
        ),
    )

    # ========================================================
    # LOAD MODEL
    # ========================================================

    model, source_checkpoint = (
        load_model3(
            args.checkpoint,
            device,
        )
    )

    print()
    print(
        "Model 3 loaded successfully."
    )

    # ========================================================
    # BASELINE VALIDATION
    # ========================================================

    initial_validation = (
        run_validation(
            model=model,
            replay_buffer=replay_buffer,
            batch_size=args.batch_size,
            validation_steps=(
                args.validation_steps
            ),
        )
    )

    print_validation(
        "BEFORE PRETRAINING",
        initial_validation,
    )

    config = {
        "batch_size":
            args.batch_size,

        "head_steps":
            args.head_steps,

        "head_lr":
            args.head_lr,

        "joint_steps":
            args.joint_steps,

        "joint_lr":
            args.joint_lr,

        "weight_decay":
            args.weight_decay,

        "validation_steps":
            args.validation_steps,
    }

    # ========================================================
    # PHASE A
    # ========================================================

    head_metrics = None
    head_validation = None

    if args.head_steps > 0:

        print()
        print(
            "=" * 70
        )

        print(
            "PHASE A: WDL HEAD WARMUP"
        )

        print(
            "=" * 70
        )

        head_metrics = (
            train_wdl_head_only(
                model=model,
                replay_buffer=replay_buffer,
                steps=args.head_steps,
                batch_size=args.batch_size,
                learning_rate=(
                    args.head_lr
                ),
                weight_decay=(
                    args.weight_decay
                ),
                split="train",
                log_every=(
                    args.log_every
                ),
            )
        )

        head_validation = (
            run_validation(
                model=model,
                replay_buffer=replay_buffer,
                batch_size=args.batch_size,
                validation_steps=(
                    args.validation_steps
                ),
            )
        )

        print_validation(
            "AFTER WDL HEAD WARMUP",
            head_validation,
        )

        base, extension = (
            os.path.splitext(
                args.output
            )
        )

        head_checkpoint_path = (
            base
            + "_head_warmup"
            + (
                extension
                if extension
                else ".pt"
            )
        )

        save_model3_checkpoint(
            path=(
                head_checkpoint_path
            ),
            model=model,
            source_checkpoint=(
                args.checkpoint
            ),
            replay_path=args.replay,
            phase="wdl_head_warmup",
            phase_metrics={
                "training":
                    head_metrics,

                "validation":
                    head_validation,
            },
            config=config,
        )

        print(
            "Saved head-warmup checkpoint:",
            head_checkpoint_path,
        )

    # ========================================================
    # PHASE B
    # ========================================================

    joint_metrics = None
    final_validation = (
        head_validation
        if head_validation
        is not None
        else initial_validation
    )

    if args.joint_steps > 0:

        print()
        print(
            "=" * 70
        )

        print(
            "PHASE B: JOINT POLICY + WDL FINE-TUNING"
        )

        print(
            "=" * 70
        )

        unfreeze_all(
            model
        )

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.joint_lr,
            weight_decay=(
                args.weight_decay
            ),
        )

        joint_metrics = (
            train_network(
                model=model,
                replay_buffer=(
                    replay_buffer
                ),
                optimizer=optimizer,
                batch_size=(
                    args.batch_size
                ),
                training_steps=(
                    args.joint_steps
                ),
                scheduler=None,
                split="train",
            )
        )

        final_validation = (
            run_validation(
                model=model,
                replay_buffer=replay_buffer,
                batch_size=args.batch_size,
                validation_steps=(
                    args.validation_steps
                ),
            )
        )

        print_validation(
            "AFTER JOINT FINE-TUNING",
            final_validation,
        )

    # ========================================================
    # FINAL SAVE
    # ========================================================

    unfreeze_all(
        model
    )

    save_model3_checkpoint(
        path=args.output,
        model=model,
        source_checkpoint=(
            args.checkpoint
        ),
        replay_path=args.replay,
        phase=(
            "joint_pretrained"
            if args.joint_steps > 0
            else "wdl_head_pretrained"
        ),
        phase_metrics={
            "initial_validation":
                initial_validation,

            "head_training":
                head_metrics,

            "head_validation":
                head_validation,

            "joint_training":
                joint_metrics,

            "final_validation":
                final_validation,
        },
        config=config,
    )

    print()
    print(
        "=" * 70
    )

    print(
        "MODEL 3 PRETRAINING COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        "Saved final checkpoint:",
        args.output,
    )

    print()

    print(
        "Initial val WDL loss:",
        f"{initial_validation['average_wdl_loss']:.4f}",
    )

    print(
        "Final val WDL loss:  ",
        f"{final_validation['average_wdl_loss']:.4f}",
    )

    print(
        "Initial val WDL acc: ",
        f"{initial_validation['average_wdl_accuracy']:.3f}",
    )

    print(
        "Final val WDL acc:   ",
        f"{final_validation['average_wdl_accuracy']:.3f}",
    )

    print()


if __name__ == "__main__":
    main()
