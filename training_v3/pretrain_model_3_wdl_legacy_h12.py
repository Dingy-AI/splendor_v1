import argparse
import os
import pickle
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from splendor_v1.network.model_3_wdl_output import (
    SplendorNetwork,
    WDL_LOSS,
    WDL_DRAW,
    WDL_WIN,
)

from splendor_v1.network.losses_3_wdl import (
    policy_wdl_loss,
)


# ============================================================
# DATASET
# ============================================================


class OldReplayDataset(Dataset):
    """
    Legacy replay format:

        data = {
            "capacity": ...,
            "buffer": [
                (
                    observation,   # float32 [258]
                    policy,        # float32 [1139]
                    value,         # scalar -1 / 0 / +1
                ),
                ...
            ],
            "position": ...,
            "successful_games": ...,
            "failed_seeds": ...,
        }

    Model 3 converts the old scalar target into:

        -1 -> LOSS (0)
         0 -> DRAW (1)
        +1 -> WIN  (2)
    """

    def __init__(
        self,
        replay_path,
    ):
        self.replay_path = Path(
            replay_path
        )

        if not self.replay_path.exists():
            raise FileNotFoundError(
                f"Replay file not found: "
                f"{self.replay_path}"
            )

        with self.replay_path.open(
            "rb"
        ) as file:
            data = pickle.load(
                file
            )

        if not isinstance(
            data,
            dict,
        ):
            raise TypeError(
                "Expected legacy replay .pkl "
                "to contain a dictionary."
            )

        if "buffer" not in data:
            raise KeyError(
                "Replay dictionary does not "
                "contain 'buffer'."
            )

        self.data = data
        self.samples = data[
            "buffer"
        ]

        if not isinstance(
            self.samples,
            (list, tuple),
        ):
            raise TypeError(
                "Replay data['buffer'] must "
                "be a list or tuple."
            )

        if len(
            self.samples
        ) == 0:
            raise ValueError(
                "Replay buffer is empty."
            )

        self._validate_example(
            self.samples[0]
        )

    @staticmethod
    def scalar_to_wdl(
        value,
    ):
        value = float(
            value
        )

        if np.isclose(
            value,
            -1.0,
            atol=1e-6,
        ):
            return WDL_LOSS

        if np.isclose(
            value,
            0.0,
            atol=1e-6,
        ):
            return WDL_DRAW

        if np.isclose(
            value,
            1.0,
            atol=1e-6,
        ):
            return WDL_WIN

        raise ValueError(
            "Legacy value target must be "
            "-1, 0, or +1. "
            f"Got {value}."
        )

    @staticmethod
    def _validate_example(
        sample,
    ):
        if (
            not isinstance(
                sample,
                (tuple, list),
            )
            or len(sample) != 3
        ):
            raise TypeError(
                "Expected each legacy replay "
                "sample to be a 3-tuple: "
                "(observation, policy, value)."
            )

        (
            observation,
            policy,
            value,
        ) = sample

        observation = np.asarray(
            observation
        )

        policy = np.asarray(
            policy
        )

        if observation.shape != (
            258,
        ):
            raise ValueError(
                "Expected observation shape "
                f"(258,), got "
                f"{observation.shape}."
            )

        if policy.shape != (
            1139,
        ):
            raise ValueError(
                "Expected policy shape "
                f"(1139,), got "
                f"{policy.shape}."
            )

        OldReplayDataset.scalar_to_wdl(
            value
        )

    def __len__(
        self,
    ):
        return len(
            self.samples
        )

    def __getitem__(
        self,
        index,
    ):
        (
            observation,
            policy,
            scalar_value,
        ) = self.samples[
            index
        ]

        observation = torch.as_tensor(
            observation,
            dtype=torch.float32,
        )

        policy = torch.as_tensor(
            policy,
            dtype=torch.float32,
        )

        target_wdl = torch.tensor(
            self.scalar_to_wdl(
                scalar_value
            ),
            dtype=torch.long,
        )

        scalar_value = torch.tensor(
            float(
                scalar_value
            ),
            dtype=torch.float32,
        )

        return (
            observation,
            policy,
            target_wdl,
            scalar_value,
        )


# ============================================================
# REPRODUCIBILITY
# ============================================================


def set_seed(
    seed,
):
    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
        )


# ============================================================
# MODEL LOADING / SAVING
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
        "Checkpoint must either be a "
        "raw model state_dict or contain "
        "'model_state_dict'."
    )


def load_model3(
    checkpoint_path,
    device,
):
    checkpoint_path = Path(
        checkpoint_path
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Model 3 checkpoint not found: "
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
        device=device,
        dtype=torch.float32,
    )

    return (
        model,
        checkpoint,
    )


def save_checkpoint(
    path,
    model,
    source_checkpoint,
    replay_path,
    phase,
    epoch,
    metrics,
    config,
):
    path = Path(
        path
    )

    path.parent.mkdir(
        parents=True,
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
            str(
                source_checkpoint
            ),

        "pretraining_replay":
            str(
                replay_path
            ),

        "pretraining_phase":
            phase,

        "pretraining_epoch":
            int(
                epoch
            ),

        "metrics":
            metrics,

        "config":
            config,
    }

    temp_path = Path(
        str(path) + ".tmp"
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
# SPLIT
# ============================================================


def build_contiguous_split(
    dataset,
    val_fraction,
):
    """
    Preserve the old H12 split behavior.

    Legacy games were appended sequentially, but the old
    files contain no game IDs. Therefore use a contiguous
    90/10 position split:

        [0, split_index)   -> train
        [split_index, end) -> validation

    This minimizes trajectory leakage compared with a
    random position-level split.
    """

    total_size = len(
        dataset
    )

    val_size = max(
        1,
        int(
            total_size
            * val_fraction
        ),
    )

    train_size = (
        total_size
        - val_size
    )

    if train_size <= 0:
        raise ValueError(
            "Training split would be empty."
        )

    split_index = (
        train_size
    )

    train_dataset = Subset(
        dataset,
        range(
            0,
            split_index,
        ),
    )

    val_dataset = Subset(
        dataset,
        range(
            split_index,
            total_size,
        ),
    )

    return (
        train_dataset,
        val_dataset,
        split_index,
    )


# ============================================================
# LABEL DISTRIBUTION
# ============================================================


def get_wdl_distribution(
    dataset,
):
    counts = {
        WDL_LOSS: 0,
        WDL_DRAW: 0,
        WDL_WIN: 0,
    }

    for index in range(
        len(dataset)
    ):
        (
            _,
            _,
            target_wdl,
            _,
        ) = dataset[
            index
        ]

        counts[
            int(
                target_wdl.item()
            )
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


def print_distribution(
    name,
    distribution,
):
    total = distribution[
        "total"
    ]

    print(
        f"{name} WDL distribution"
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
# FREEZE / UNFREEZE
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


# ============================================================
# METRICS
# ============================================================


def compute_policy_kl(
    policy_logits,
    target_policy,
):
    log_probs = F.log_softmax(
        policy_logits,
        dim=-1,
    )

    target_log_probs = torch.log(
        target_policy.clamp_min(
            1e-8
        )
    )

    return (
        target_policy
        * (
            target_log_probs
            - log_probs
        )
    ).sum(
        dim=-1
    ).mean()


def evaluate(
    model,
    data_loader,
    device,
):
    was_training = (
        model.training
    )

    model.eval()

    total_losses = []
    policy_losses = []
    wdl_losses = []
    policy_kls = []
    wdl_accuracies = []
    value_maes = []

    loss_probabilities = []
    draw_probabilities = []
    win_probabilities = []

    target_loss_fractions = []
    target_draw_fractions = []
    target_win_fractions = []

    with torch.inference_mode():

        for (
            observations,
            target_policies,
            target_wdl,
            scalar_values,
        ) in data_loader:

            observations = (
                observations.to(
                    device,
                    non_blocking=True,
                )
            )

            target_policies = (
                target_policies.to(
                    device,
                    non_blocking=True,
                )
            )

            target_wdl = (
                target_wdl.to(
                    device,
                    non_blocking=True,
                )
            )

            scalar_values = (
                scalar_values.to(
                    device,
                    non_blocking=True,
                )
            )

            (
                policy_logits,
                wdl_logits,
            ) = model(
                observations
            )

            (
                total_loss,
                policy_loss,
                wdl_loss,
                policy_kl,
            ) = policy_wdl_loss(
                policy_logits,
                wdl_logits,
                target_policies,
                target_wdl,
            )

            probabilities = torch.softmax(
                wdl_logits,
                dim=-1,
            )

            predicted_values = (
                probabilities[
                    :,
                    WDL_WIN
                ]
                - probabilities[
                    :,
                    WDL_LOSS
                ]
            )

            accuracy = (
                wdl_logits.argmax(
                    dim=-1
                )
                == target_wdl
            ).float().mean()

            value_mae = (
                predicted_values
                - scalar_values
            ).abs().mean()

            total_losses.append(
                total_loss.item()
            )

            policy_losses.append(
                policy_loss.item()
            )

            wdl_losses.append(
                wdl_loss.item()
            )

            policy_kls.append(
                policy_kl.item()
            )

            wdl_accuracies.append(
                accuracy.item()
            )

            value_maes.append(
                value_mae.item()
            )

            loss_probabilities.append(
                probabilities[
                    :,
                    WDL_LOSS
                ].mean().item()
            )

            draw_probabilities.append(
                probabilities[
                    :,
                    WDL_DRAW
                ].mean().item()
            )

            win_probabilities.append(
                probabilities[
                    :,
                    WDL_WIN
                ].mean().item()
            )

            target_loss_fractions.append(
                target_wdl.eq(
                    WDL_LOSS
                ).float().mean().item()
            )

            target_draw_fractions.append(
                target_wdl.eq(
                    WDL_DRAW
                ).float().mean().item()
            )

            target_win_fractions.append(
                target_wdl.eq(
                    WDL_WIN
                ).float().mean().item()
            )

    if was_training:
        model.train()

    return {
        "total_loss":
            float(
                np.mean(
                    total_losses
                )
            ),

        "policy_loss":
            float(
                np.mean(
                    policy_losses
                )
            ),

        "wdl_loss":
            float(
                np.mean(
                    wdl_losses
                )
            ),

        "policy_kl":
            float(
                np.mean(
                    policy_kls
                )
            ),

        "wdl_accuracy":
            float(
                np.mean(
                    wdl_accuracies
                )
            ),

        "value_mae":
            float(
                np.mean(
                    value_maes
                )
            ),

        "loss_probability":
            float(
                np.mean(
                    loss_probabilities
                )
            ),

        "draw_probability":
            float(
                np.mean(
                    draw_probabilities
                )
            ),

        "win_probability":
            float(
                np.mean(
                    win_probabilities
                )
            ),

        "target_loss_fraction":
            float(
                np.mean(
                    target_loss_fractions
                )
            ),

        "target_draw_fraction":
            float(
                np.mean(
                    target_draw_fractions
                )
            ),

        "target_win_fraction":
            float(
                np.mean(
                    target_win_fractions
                )
            ),
    }


def print_metrics(
    label,
    metrics,
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
        "Total loss:      ",
        f"{metrics['total_loss']:.4f}",
    )

    print(
        "Policy loss:     ",
        f"{metrics['policy_loss']:.4f}",
    )

    print(
        "Policy KL:       ",
        f"{metrics['policy_kl']:.4f}",
    )

    print(
        "WDL loss:        ",
        f"{metrics['wdl_loss']:.4f}",
    )

    print(
        "WDL accuracy:    ",
        f"{metrics['wdl_accuracy']:.3f}",
    )

    print(
        "Scalar value MAE:",
        f"{metrics['value_mae']:.4f}",
    )

    print(
        "Predicted L/D/W: ",
        (
            f"{metrics['loss_probability']:.3f} / "
            f"{metrics['draw_probability']:.3f} / "
            f"{metrics['win_probability']:.3f}"
        ),
    )

    print(
        "Target L/D/W:    ",
        (
            f"{metrics['target_loss_fraction']:.3f} / "
            f"{metrics['target_draw_fraction']:.3f} / "
            f"{metrics['target_win_fraction']:.3f}"
        ),
    )


# ============================================================
# PHASE A: WDL HEAD ONLY
# ============================================================


def train_wdl_head_epoch(
    model,
    data_loader,
    optimizer,
    device,
):
    model.train()

    losses = []
    accuracies = []

    for (
        observations,
        _,
        target_wdl,
        _,
    ) in data_loader:

        observations = observations.to(
            device,
            non_blocking=True,
        )

        target_wdl = target_wdl.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        # Trunk is frozen. Compute transferred features
        # without storing a backward graph.
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
            ).float().mean()

        losses.append(
            loss.item()
        )

        accuracies.append(
            accuracy.item()
        )

    return {
        "wdl_loss":
            float(
                np.mean(
                    losses
                )
            ),

        "wdl_accuracy":
            float(
                np.mean(
                    accuracies
                )
            ),
    }


# ============================================================
# PHASE B: JOINT POLICY + WDL
# ============================================================


def train_joint_epoch(
    model,
    data_loader,
    optimizer,
    device,
    grad_clip=None,
):
    model.train()

    total_losses = []
    policy_losses = []
    wdl_losses = []
    policy_kls = []
    grad_norms = []

    for (
        observations,
        target_policies,
        target_wdl,
        _,
    ) in data_loader:

        observations = observations.to(
            device,
            non_blocking=True,
        )

        target_policies = (
            target_policies.to(
                device,
                non_blocking=True,
            )
        )

        target_wdl = target_wdl.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        (
            policy_logits,
            wdl_logits,
        ) = model(
            observations
        )

        (
            total_loss,
            policy_loss,
            wdl_loss,
            policy_kl,
        ) = policy_wdl_loss(
            policy_logits,
            wdl_logits,
            target_policies,
            target_wdl,
        )

        total_loss.backward()

        if grad_clip is None:

            grad_norm = (
                torch.nn.utils
                .clip_grad_norm_(
                    model.parameters(),
                    max_norm=float(
                        "inf"
                    ),
                )
            )

        else:

            grad_norm = (
                torch.nn.utils
                .clip_grad_norm_(
                    model.parameters(),
                    max_norm=grad_clip,
                )
            )

        optimizer.step()

        total_losses.append(
            total_loss.item()
        )

        policy_losses.append(
            policy_loss.item()
        )

        wdl_losses.append(
            wdl_loss.item()
        )

        policy_kls.append(
            policy_kl.item()
        )

        grad_norms.append(
            float(
                grad_norm.item()
            )
        )

    return {
        "total_loss":
            float(
                np.mean(
                    total_losses
                )
            ),

        "policy_loss":
            float(
                np.mean(
                    policy_losses
                )
            ),

        "wdl_loss":
            float(
                np.mean(
                    wdl_losses
                )
            ),

        "policy_kl":
            float(
                np.mean(
                    policy_kls
                )
            ),

        "grad_norm":
            float(
                np.mean(
                    grad_norms
                )
            ),
    }


# ============================================================
# MAIN
# ============================================================


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Pretrain Model 3 WDL on the old H12 "
            "(observation, policy, scalar value) replay format."
        )
    )

    parser.add_argument(
        "--replay",
        default=(
            "splendor_v1/training/data/"
            "h12_replay_data_c.pkl"
        ),
    )

    parser.add_argument(
        "--checkpoint",
        default=(
            "checkpoints/gen_3/"
            "gen_3_wdl_transfer_g2h12.pt"
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "checkpoints/gen_3"
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.10,
    )

    parser.add_argument(
        "--head-epochs",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--head-lr",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--joint-epochs",
        type=int,
        default=10,
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
        "--grad-clip",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20260922,
    )

    args = parser.parse_args()

    if not (
        0.0
        < args.val_fraction
        < 1.0
    ):
        raise ValueError(
            "--val-fraction must be "
            "between 0 and 1."
        )

    if args.head_epochs < 0:
        raise ValueError(
            "--head-epochs must be >= 0."
        )

    if args.joint_epochs < 0:
        raise ValueError(
            "--joint-epochs must be >= 0."
        )

    if args.batch_size < 1:
        raise ValueError(
            "--batch-size must be >= 1."
        )

    set_seed(
        args.seed
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
        "LEGACY H12 REPLAY FORMAT"
    )

    print(
        "=" * 70
    )

    print(
        f"Device:          {device}"
    )

    print(
        f"Replay:          {args.replay}"
    )

    print(
        f"Checkpoint:      {args.checkpoint}"
    )

    print(
        f"Batch size:      {args.batch_size}"
    )

    print(
        f"Validation:      "
        f"{args.val_fraction:.0%}"
    )

    print(
        f"WDL-head epochs: {args.head_epochs}"
    )

    print(
        f"Joint epochs:    {args.joint_epochs}"
    )

    print()

    # ========================================================
    # DATA
    # ========================================================

    dataset = OldReplayDataset(
        args.replay
    )

    (
        train_dataset,
        val_dataset,
        split_index,
    ) = build_contiguous_split(
        dataset,
        args.val_fraction,
    )

    print(
        f"Total positions: {len(dataset):,}"
    )

    print(
        f"Split strategy:  contiguous "
        f"{1.0 - args.val_fraction:.0%}/"
        f"{args.val_fraction:.0%}"
    )

    print(
        f"Split index:     {split_index:,}"
    )

    print(
        f"Train range:     "
        f"[0, {split_index:,})"
    )

    print(
        f"Val range:       "
        f"[{split_index:,}, "
        f"{len(dataset):,})"
    )

    print(
        f"Train positions: "
        f"{len(train_dataset):,}"
    )

    print(
        f"Val positions:   "
        f"{len(val_dataset):,}"
    )

    print()

    train_distribution = (
        get_wdl_distribution(
            train_dataset
        )
    )

    val_distribution = (
        get_wdl_distribution(
            val_dataset
        )
    )

    print_distribution(
        "TRAIN",
        train_distribution,
    )

    print()

    print_distribution(
        "VAL",
        val_distribution,
    )

    if (
        train_distribution[
            "draw"
        ]
        == 0
    ):

        print()
        print(
            "WARNING: this legacy replay contains "
            "no DRAW targets."
        )

        print(
            "The WDL head can learn LOSS/WIN from "
            "this dataset, but DRAW behavior cannot "
            "be supervised until replay data includes "
            "actual draws."
        )

    # ========================================================
    # LOADERS
    # ========================================================

    pin_memory = (
        device.type
        == "cuda"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    # ========================================================
    # MODEL
    # ========================================================

    (
        model,
        source_checkpoint,
    ) = load_model3(
        args.checkpoint,
        device,
    )

    print()
    print(
        "Transferred Model 3 loaded."
    )

    baseline_metrics = evaluate(
        model,
        val_loader,
        device,
    )

    print_metrics(
        "BEFORE PRETRAINING",
        baseline_metrics,
    )

    config = {
        "batch_size":
            args.batch_size,

        "val_fraction":
            args.val_fraction,

        "split_strategy":
            "contiguous_positions",

        "split_index":
            split_index,

        "head_epochs":
            args.head_epochs,

        "head_lr":
            args.head_lr,

        "joint_epochs":
            args.joint_epochs,

        "joint_lr":
            args.joint_lr,

        "weight_decay":
            args.weight_decay,

        "grad_clip":
            args.grad_clip,

        "seed":
            args.seed,
    }

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # PHASE A: HEAD ONLY
    # ========================================================

    best_head_val_loss = float(
        "inf"
    )

    best_head_metrics = None

    if args.head_epochs > 0:

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

        freeze_except_wdl_head(
            model
        )

        head_optimizer = (
            torch.optim.AdamW(
                model.wdl_head.parameters(),
                lr=args.head_lr,
                weight_decay=(
                    args.weight_decay
                ),
            )
        )

        for epoch in range(
            1,
            args.head_epochs + 1,
        ):

            train_metrics = (
                train_wdl_head_epoch(
                    model=model,
                    data_loader=train_loader,
                    optimizer=head_optimizer,
                    device=device,
                )
            )

            val_metrics = evaluate(
                model,
                val_loader,
                device,
            )

            print()
            print(
                f"Head epoch "
                f"{epoch}/{args.head_epochs}"
            )

            print(
                "  train WDL loss: ",
                f"{train_metrics['wdl_loss']:.4f}",
            )

            print(
                "  train WDL acc:  ",
                f"{train_metrics['wdl_accuracy']:.3f}",
            )

            print(
                "  val WDL loss:   ",
                f"{val_metrics['wdl_loss']:.4f}",
            )

            print(
                "  val WDL acc:    ",
                f"{val_metrics['wdl_accuracy']:.3f}",
            )

            print(
                "  val draw prob:  ",
                f"{val_metrics['draw_probability']:.3f}",
            )

            if (
                val_metrics[
                    "wdl_loss"
                ]
                < best_head_val_loss
            ):

                best_head_val_loss = (
                    val_metrics[
                        "wdl_loss"
                    ]
                )

                best_head_metrics = (
                    val_metrics
                )

                save_checkpoint(
                    path=(
                        output_dir
                        / (
                            "gen_3_wdl_"
                            "head_warmup_best.pt"
                        )
                    ),
                    model=model,
                    source_checkpoint=(
                        args.checkpoint
                    ),
                    replay_path=(
                        args.replay
                    ),
                    phase=(
                        "wdl_head_warmup"
                    ),
                    epoch=epoch,
                    metrics={
                        "train":
                            train_metrics,

                        "validation":
                            val_metrics,
                    },
                    config=config,
                )

        print_metrics(
            "AFTER WDL HEAD WARMUP",
            evaluate(
                model,
                val_loader,
                device,
            ),
        )

    # ========================================================
    # PHASE B: FULL NETWORK
    # ========================================================

    best_joint_val_loss = float(
        "inf"
    )

    best_joint_metrics = None

    if args.joint_epochs > 0:

        print()
        print(
            "=" * 70
        )

        print(
            "PHASE B: JOINT POLICY + WDL"
        )

        print(
            "=" * 70
        )

        unfreeze_all(
            model
        )

        joint_optimizer = (
            torch.optim.AdamW(
                model.parameters(),
                lr=args.joint_lr,
                weight_decay=(
                    args.weight_decay
                ),
            )
        )

        for epoch in range(
            1,
            args.joint_epochs + 1,
        ):

            train_metrics = (
                train_joint_epoch(
                    model=model,
                    data_loader=train_loader,
                    optimizer=joint_optimizer,
                    device=device,
                    grad_clip=(
                        args.grad_clip
                    ),
                )
            )

            val_metrics = evaluate(
                model,
                val_loader,
                device,
            )

            print()
            print(
                f"Joint epoch "
                f"{epoch}/{args.joint_epochs}"
            )

            print(
                "  train total:    ",
                f"{train_metrics['total_loss']:.4f}",
            )

            print(
                "  train policy:   ",
                f"{train_metrics['policy_loss']:.4f}",
            )

            print(
                "  train WDL:      ",
                f"{train_metrics['wdl_loss']:.4f}",
            )

            print(
                "  train policy KL:",
                f"{train_metrics['policy_kl']:.4f}",
            )

            print(
                "  grad norm:      ",
                f"{train_metrics['grad_norm']:.4f}",
            )

            print(
                "  val total:      ",
                f"{val_metrics['total_loss']:.4f}",
            )

            print(
                "  val policy KL:  ",
                f"{val_metrics['policy_kl']:.4f}",
            )

            print(
                "  val WDL loss:   ",
                f"{val_metrics['wdl_loss']:.4f}",
            )

            print(
                "  val WDL acc:    ",
                f"{val_metrics['wdl_accuracy']:.3f}",
            )

            if (
                val_metrics[
                    "total_loss"
                ]
                < best_joint_val_loss
            ):

                best_joint_val_loss = (
                    val_metrics[
                        "total_loss"
                    ]
                )

                best_joint_metrics = (
                    val_metrics
                )

                save_checkpoint(
                    path=(
                        output_dir
                        / (
                            "gen_3_wdl_"
                            "joint_best.pt"
                        )
                    ),
                    model=model,
                    source_checkpoint=(
                        args.checkpoint
                    ),
                    replay_path=(
                        args.replay
                    ),
                    phase=(
                        "joint_policy_wdl"
                    ),
                    epoch=epoch,
                    metrics={
                        "train":
                            train_metrics,

                        "validation":
                            val_metrics,
                    },
                    config=config,
                )

    # ========================================================
    # FINAL
    # ========================================================

    unfreeze_all(
        model
    )

    final_metrics = evaluate(
        model,
        val_loader,
        device,
    )

    final_path = (
        output_dir
        / "gen_3_wdl_pretrained_final.pt"
    )

    save_checkpoint(
        path=final_path,
        model=model,
        source_checkpoint=(
            args.checkpoint
        ),
        replay_path=(
            args.replay
        ),
        phase=(
            "legacy_h12_pretraining_complete"
        ),
        epoch=(
            args.joint_epochs
            if args.joint_epochs > 0
            else args.head_epochs
        ),
        metrics={
            "baseline":
                baseline_metrics,

            "best_head_validation":
                best_head_metrics,

            "best_joint_validation":
                best_joint_metrics,

            "final_validation":
                final_metrics,
        },
        config=config,
    )

    print_metrics(
        "FINAL VALIDATION",
        final_metrics,
    )

    print()
    print(
        "=" * 70
    )

    print(
        "PRETRAINING COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        "Final checkpoint:"
    )

    print(
        f"  {final_path}"
    )

    if args.head_epochs > 0:

        print(
            "Best head-only checkpoint:"
        )

        print(
            "  "
            + str(
                output_dir
                / (
                    "gen_3_wdl_"
                    "head_warmup_best.pt"
                )
            )
        )

    if args.joint_epochs > 0:

        print(
            "Best joint checkpoint:"
        )

        print(
            "  "
            + str(
                output_dir
                / (
                    "gen_3_wdl_"
                    "joint_best.pt"
                )
            )
        )

    print()


if __name__ == "__main__":
    main()
