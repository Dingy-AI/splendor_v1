from __future__ import annotations

import argparse
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from splendor_v1.network.model_4_legal_scorer import (
    SplendorNetwork,
    WDL_DRAW,
    WDL_LOSS,
    WDL_WIN,
)
from splendor_v1.training_v4.batch_builder_v4_legacy import (
    make_model4_legacy_dataloaders,
)
from splendor_v1.network.losses_4_wdl import (
    policy_wdl_loss,
)


# ============================================================
# REPRODUCIBILITY
# ============================================================


def set_seed(
    seed: int,
) -> None:

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
# CHECKPOINT HELPERS
# ============================================================


def extract_model_state_dict(
    checkpoint,
):
    """
    Accept:
        - raw state_dict
        - {"model_state_dict": ...}
        - {"state_dict": ...}
        - {"model": ...}
    """

    if not isinstance(
        checkpoint,
        dict,
    ):

        raise TypeError(
            "Expected checkpoint to be a dictionary."
        )

    if (
        checkpoint
        and all(
            torch.is_tensor(
                value
            )
            for value
            in checkpoint.values()
        )
    ):

        return checkpoint

    for key in (
        "model_state_dict",
        "state_dict",
        "model",
    ):

        state_dict = checkpoint.get(
            key
        )

        if isinstance(
            state_dict,
            dict,
        ):

            return state_dict

    raise KeyError(
        "Could not find model state_dict in checkpoint."
    )


def load_model4(
    checkpoint_path: str | Path,
    device: torch.device,
):
    checkpoint_path = Path(
        checkpoint_path
    )

    if not checkpoint_path.exists():

        raise FileNotFoundError(
            "Model 4 checkpoint not found: "
            f"{checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    state_dict = extract_model_state_dict(
        checkpoint
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

    model.eval()

    return (
        model,
        checkpoint,
    )


def save_checkpoint(
    path: str | Path,
    model,
    source_checkpoint_path,
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
            4,

        "architecture":
            "attention_wdl_legal_action_scorer",

        "source_checkpoint":
            str(
                source_checkpoint_path
            ),

        "pretraining_replay":
            str(
                replay_path
            ),

        "pretraining_data_format":
            "legacy_h12_dense_policy",

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
        str(
            path
        )
        + ".tmp"
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
# TRAINABLE-PARAMETER CONTROL
# ============================================================


def freeze_except_model4_policy(
    model,
) -> None:
    """
    Phase A:
        freeze every transferred Model 3 component
        train only Model 4's new policy machinery
    """

    for parameter in model.parameters():

        parameter.requires_grad = False

    for parameter in model.action_embedding.parameters():

        parameter.requires_grad = True

    for parameter in model.legal_action_scorer.parameters():

        parameter.requires_grad = True


def unfreeze_all(
    model,
) -> None:

    for parameter in model.parameters():

        parameter.requires_grad = True


def trainable_parameters(
    model,
):

    return [
        parameter
        for parameter
        in model.parameters()
        if parameter.requires_grad
    ]


def count_parameters(
    parameters,
) -> int:

    return sum(
        int(
            parameter.numel()
        )
        for parameter
        in parameters
    )


# ============================================================
# DEVICE BATCH
# ============================================================


def move_batch_to_device(
    batch,
    device,
):
    """
    Move only tensors needed by the Model 4 objective.
    """

    return {
        "observations":
            batch[
                "observations"
            ].to(
                device,
                non_blocking=True,
            ),

        "legal_action_ids":
            batch[
                "legal_action_ids"
            ].to(
                device,
                non_blocking=True,
            ),

        "legal_action_mask":
            batch[
                "legal_action_mask"
            ].to(
                device,
                non_blocking=True,
            ),

        "target_policy":
            batch[
                "target_policy"
            ].to(
                device,
                non_blocking=True,
            ),

        "target_wdl":
            batch[
                "target_wdl"
            ].to(
                device,
                non_blocking=True,
            ),
    }


# ============================================================
# POLICY-ONLY WARMUP LOSS
# ============================================================


def policy_only_loss(
    policy_logits,
    target_policy,
    legal_action_mask,
):
    """
    Same policy objective used by losses_4_wdl.py, without WDL.

    This is used only in Phase A because the transferred trunk and
    WDL head are frozen and should remain exactly unchanged.
    """

    masked_value = torch.finfo(
        policy_logits.dtype
    ).min

    masked_logits = policy_logits.masked_fill(
        ~legal_action_mask,
        masked_value,
    )

    log_probs = F.log_softmax(
        masked_logits,
        dim=-1,
    )

    policy_loss = -(
        target_policy
        * log_probs
    ).sum(
        dim=-1
    ).mean()

    with torch.no_grad():

        target_log_probs = torch.log(
            target_policy.clamp_min(
                1e-8
            )
        )

        policy_kl = (
            target_policy
            * (
                target_log_probs
                - log_probs
            )
        ).sum(
            dim=-1
        ).mean()

        target_entropy = -(
            target_policy
            * target_log_probs
        ).sum(
            dim=-1
        ).mean()

    return (
        policy_loss,
        policy_kl,
        target_entropy,
    )


# ============================================================
# METRIC HELPERS
# ============================================================


def _weighted_average(
    total,
    count,
):
    if count == 0:

        return float(
            "nan"
        )

    return float(
        total
        / count
    )


def _wdl_scalar_from_logits(
    wdl_logits,
):
    probabilities = F.softmax(
        wdl_logits,
        dim=-1,
    )

    scalar = (
        probabilities[
            :,
            WDL_WIN
        ]
        - probabilities[
            :,
            WDL_LOSS
        ]
    )

    return (
        probabilities,
        scalar,
    )


def _target_scalar_from_wdl(
    target_wdl,
):
    scalar = torch.zeros_like(
        target_wdl,
        dtype=torch.float32,
    )

    scalar = torch.where(
        target_wdl
        == WDL_WIN,
        torch.ones_like(
            scalar
        ),
        scalar,
    )

    scalar = torch.where(
        target_wdl
        == WDL_LOSS,
        -torch.ones_like(
            scalar
        ),
        scalar,
    )

    return scalar


# ============================================================
# EVALUATION
# ============================================================


@torch.inference_mode()
def evaluate(
    model,
    data_loader,
    device,
):
    model.eval()

    total_samples = 0

    total_loss_sum = 0.0
    policy_loss_sum = 0.0
    wdl_loss_sum = 0.0
    policy_kl_sum = 0.0

    target_entropy_sum = 0.0

    policy_top1_correct = 0
    wdl_correct = 0

    scalar_abs_error_sum = 0.0

    predicted_ldw_sum = torch.zeros(
        3,
        dtype=torch.float64,
    )

    target_ldw_count = torch.zeros(
        3,
        dtype=torch.float64,
    )

    for raw_batch in data_loader:

        batch = move_batch_to_device(
            raw_batch,
            device,
        )

        (
            policy_logits,
            wdl_logits,
        ) = model(
            batch[
                "observations"
            ],
            batch[
                "legal_action_ids"
            ],
            batch[
                "legal_action_mask"
            ],
        )

        (
            total_loss,
            policy_loss,
            wdl_loss,
            policy_kl,
        ) = policy_wdl_loss(
            policy_logits=policy_logits,
            wdl_logits=wdl_logits,
            target_policy=batch[
                "target_policy"
            ],
            target_wdl=batch[
                "target_wdl"
            ],
            legal_action_mask=batch[
                "legal_action_mask"
            ],
        )

        target_log_probs = torch.log(
            batch[
                "target_policy"
            ].clamp_min(
                1e-8
            )
        )

        target_entropy = -(
            batch[
                "target_policy"
            ]
            * target_log_probs
        ).sum(
            dim=-1
        ).mean()

        batch_size = int(
            batch[
                "observations"
            ].shape[0]
        )

        total_samples += batch_size

        total_loss_sum += (
            total_loss.item()
            * batch_size
        )

        policy_loss_sum += (
            policy_loss.item()
            * batch_size
        )

        wdl_loss_sum += (
            wdl_loss.item()
            * batch_size
        )

        policy_kl_sum += (
            policy_kl.item()
            * batch_size
        )

        target_entropy_sum += (
            target_entropy.item()
            * batch_size
        )

        predicted_action = (
            policy_logits.argmax(
                dim=-1
            )
        )

        target_action = (
            batch[
                "target_policy"
            ].argmax(
                dim=-1
            )
        )

        policy_top1_correct += int(
            (
                predicted_action
                == target_action
            ).sum().item()
        )

        predicted_wdl = (
            wdl_logits.argmax(
                dim=-1
            )
        )

        wdl_correct += int(
            (
                predicted_wdl
                == batch[
                    "target_wdl"
                ]
            ).sum().item()
        )

        (
            wdl_probabilities,
            predicted_scalar,
        ) = _wdl_scalar_from_logits(
            wdl_logits
        )

        target_scalar = (
            _target_scalar_from_wdl(
                batch[
                    "target_wdl"
                ]
            )
        )

        scalar_abs_error_sum += float(
            (
                predicted_scalar
                - target_scalar
            ).abs().sum().item()
        )

        predicted_ldw_sum += (
            wdl_probabilities
            .sum(
                dim=0
            )
            .detach()
            .cpu()
            .double()
        )

        target_ldw_count += (
            F.one_hot(
                batch[
                    "target_wdl"
                ],
                num_classes=3,
            )
            .sum(
                dim=0
            )
            .detach()
            .cpu()
            .double()
        )

    return {
        "total_loss":
            _weighted_average(
                total_loss_sum,
                total_samples,
            ),

        "policy_loss":
            _weighted_average(
                policy_loss_sum,
                total_samples,
            ),

        "wdl_loss":
            _weighted_average(
                wdl_loss_sum,
                total_samples,
            ),

        "policy_kl":
            _weighted_average(
                policy_kl_sum,
                total_samples,
            ),

        "target_policy_entropy":
            _weighted_average(
                target_entropy_sum,
                total_samples,
            ),

        "policy_top1":
            _weighted_average(
                policy_top1_correct,
                total_samples,
            ),

        "wdl_accuracy":
            _weighted_average(
                wdl_correct,
                total_samples,
            ),

        "scalar_mae":
            _weighted_average(
                scalar_abs_error_sum,
                total_samples,
            ),

        "loss_probability":
            float(
                predicted_ldw_sum[
                    WDL_LOSS
                ]
                / total_samples
            ),

        "draw_probability":
            float(
                predicted_ldw_sum[
                    WDL_DRAW
                ]
                / total_samples
            ),

        "win_probability":
            float(
                predicted_ldw_sum[
                    WDL_WIN
                ]
                / total_samples
            ),

        "target_loss_fraction":
            float(
                target_ldw_count[
                    WDL_LOSS
                ]
                / total_samples
            ),

        "target_draw_fraction":
            float(
                target_ldw_count[
                    WDL_DRAW
                ]
                / total_samples
            ),

        "target_win_fraction":
            float(
                target_ldw_count[
                    WDL_WIN
                ]
                / total_samples
            ),

        "samples":
            int(
                total_samples
            ),
    }


# ============================================================
# PHASE A: MODEL 4 POLICY WARMUP
# ============================================================


def train_policy_warmup_epoch(
    model,
    data_loader,
    optimizer,
    device,
    grad_clip=None,
):
    """
    Train ONLY:
        action_embedding
        legal_action_scorer

    The transferred attention trunk and WDL head remain frozen.

    The state features are computed under no_grad() so Phase A does
    not build a backward graph through the attention trunk.
    """

    # Keep the transferred trunk deterministic/frozen during
    # policy warmup. The current architecture uses dropout=0, but
    # eval mode also makes this robust to future nonzero dropout.
    model.eval()

    model.action_embedding.train()
    model.legal_action_scorer.train()

    total_samples = 0

    policy_loss_sum = 0.0
    policy_kl_sum = 0.0
    target_entropy_sum = 0.0

    policy_top1_correct = 0

    grad_norm_sum = 0.0
    grad_batches = 0

    for raw_batch in data_loader:

        batch = move_batch_to_device(
            raw_batch,
            device,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with torch.no_grad():

            state_features = (
                model._forward_features(
                    batch[
                        "observations"
                    ]
                )
            )

        policy_logits = (
            model._score_legal_actions(
                state_features=state_features,
                legal_action_ids=batch[
                    "legal_action_ids"
                ],
                legal_action_mask=batch[
                    "legal_action_mask"
                ],
            )
        )

        (
            policy_loss,
            policy_kl,
            target_entropy,
        ) = policy_only_loss(
            policy_logits=policy_logits,
            target_policy=batch[
                "target_policy"
            ],
            legal_action_mask=batch[
                "legal_action_mask"
            ],
        )

        policy_loss.backward()

        trainable = trainable_parameters(
            model
        )

        if grad_clip is None:

            grad_norm = (
                torch.nn.utils.clip_grad_norm_(
                    trainable,
                    max_norm=float(
                        "inf"
                    ),
                )
            )

        else:

            grad_norm = (
                torch.nn.utils.clip_grad_norm_(
                    trainable,
                    max_norm=grad_clip,
                )
            )

        optimizer.step()

        batch_size = int(
            batch[
                "observations"
            ].shape[0]
        )

        total_samples += batch_size

        policy_loss_sum += (
            policy_loss.item()
            * batch_size
        )

        policy_kl_sum += (
            policy_kl.item()
            * batch_size
        )

        target_entropy_sum += (
            target_entropy.item()
            * batch_size
        )

        policy_top1_correct += int(
            (
                policy_logits.argmax(
                    dim=-1
                )
                == batch[
                    "target_policy"
                ].argmax(
                    dim=-1
                )
            ).sum().item()
        )

        grad_norm_sum += float(
            grad_norm.item()
        )

        grad_batches += 1

    return {
        "policy_loss":
            _weighted_average(
                policy_loss_sum,
                total_samples,
            ),

        "policy_kl":
            _weighted_average(
                policy_kl_sum,
                total_samples,
            ),

        "target_policy_entropy":
            _weighted_average(
                target_entropy_sum,
                total_samples,
            ),

        "policy_top1":
            _weighted_average(
                policy_top1_correct,
                total_samples,
            ),

        "grad_norm":
            _weighted_average(
                grad_norm_sum,
                grad_batches,
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

    total_samples = 0

    total_loss_sum = 0.0
    policy_loss_sum = 0.0
    wdl_loss_sum = 0.0
    policy_kl_sum = 0.0
    target_entropy_sum = 0.0

    policy_top1_correct = 0
    wdl_correct = 0

    grad_norm_sum = 0.0
    grad_batches = 0

    for raw_batch in data_loader:

        batch = move_batch_to_device(
            raw_batch,
            device,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        (
            policy_logits,
            wdl_logits,
        ) = model(
            batch[
                "observations"
            ],
            batch[
                "legal_action_ids"
            ],
            batch[
                "legal_action_mask"
            ],
        )

        (
            total_loss,
            policy_loss,
            wdl_loss,
            policy_kl,
        ) = policy_wdl_loss(
            policy_logits=policy_logits,
            wdl_logits=wdl_logits,
            target_policy=batch[
                "target_policy"
            ],
            target_wdl=batch[
                "target_wdl"
            ],
            legal_action_mask=batch[
                "legal_action_mask"
            ],
        )

        target_log_probs = torch.log(
            batch[
                "target_policy"
            ].clamp_min(
                1e-8
            )
        )

        target_entropy = -(
            batch[
                "target_policy"
            ]
            * target_log_probs
        ).sum(
            dim=-1
        ).mean()

        total_loss.backward()

        trainable = trainable_parameters(
            model
        )

        if grad_clip is None:

            grad_norm = (
                torch.nn.utils.clip_grad_norm_(
                    trainable,
                    max_norm=float(
                        "inf"
                    ),
                )
            )

        else:

            grad_norm = (
                torch.nn.utils.clip_grad_norm_(
                    trainable,
                    max_norm=grad_clip,
                )
            )

        optimizer.step()

        batch_size = int(
            batch[
                "observations"
            ].shape[0]
        )

        total_samples += batch_size

        total_loss_sum += (
            total_loss.item()
            * batch_size
        )

        policy_loss_sum += (
            policy_loss.item()
            * batch_size
        )

        wdl_loss_sum += (
            wdl_loss.item()
            * batch_size
        )

        policy_kl_sum += (
            policy_kl.item()
            * batch_size
        )

        target_entropy_sum += (
            target_entropy.item()
            * batch_size
        )

        policy_top1_correct += int(
            (
                policy_logits.argmax(
                    dim=-1
                )
                == batch[
                    "target_policy"
                ].argmax(
                    dim=-1
                )
            ).sum().item()
        )

        wdl_correct += int(
            (
                wdl_logits.argmax(
                    dim=-1
                )
                == batch[
                    "target_wdl"
                ]
            ).sum().item()
        )

        grad_norm_sum += float(
            grad_norm.item()
        )

        grad_batches += 1

    return {
        "total_loss":
            _weighted_average(
                total_loss_sum,
                total_samples,
            ),

        "policy_loss":
            _weighted_average(
                policy_loss_sum,
                total_samples,
            ),

        "wdl_loss":
            _weighted_average(
                wdl_loss_sum,
                total_samples,
            ),

        "policy_kl":
            _weighted_average(
                policy_kl_sum,
                total_samples,
            ),

        "target_policy_entropy":
            _weighted_average(
                target_entropy_sum,
                total_samples,
            ),

        "policy_top1":
            _weighted_average(
                policy_top1_correct,
                total_samples,
            ),

        "wdl_accuracy":
            _weighted_average(
                wdl_correct,
                total_samples,
            ),

        "grad_norm":
            _weighted_average(
                grad_norm_sum,
                grad_batches,
            ),
    }


# ============================================================
# OUTPUT
# ============================================================


def print_validation_metrics(
    title,
    metrics,
):
    print()
    print(
        title
    )

    print(
        "  total loss:       ",
        f"{metrics['total_loss']:.6f}",
    )

    print(
        "  policy loss:      ",
        f"{metrics['policy_loss']:.6f}",
    )

    print(
        "  target entropy:   ",
        f"{metrics['target_policy_entropy']:.6f}",
    )

    print(
        "  policy KL:        ",
        f"{metrics['policy_kl']:.6f}",
    )

    print(
        "  policy top-1:     ",
        f"{metrics['policy_top1']:.3%}",
    )

    print(
        "  WDL loss:         ",
        f"{metrics['wdl_loss']:.6f}",
    )

    print(
        "  WDL accuracy:     ",
        f"{metrics['wdl_accuracy']:.3%}",
    )

    print(
        "  scalar MAE:       ",
        f"{metrics['scalar_mae']:.6f}",
    )

    print(
        "  predicted L/D/W:  ",
        (
            f"{metrics['loss_probability']:.3f} / "
            f"{metrics['draw_probability']:.3f} / "
            f"{metrics['win_probability']:.3f}"
        ),
    )

    print(
        "  target L/D/W:     ",
        (
            f"{metrics['target_loss_fraction']:.3f} / "
            f"{metrics['target_draw_fraction']:.3f} / "
            f"{metrics['target_win_fraction']:.3f}"
        ),
    )


# ============================================================
# MAIN
# ============================================================


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Pretrain Model 4's new legal-action scorer on "
            "legacy H12 dense-policy replay data, then optionally "
            "jointly fine-tune the full Model 4 network."
        )
    )

    parser.add_argument(
        "--replay",
        default=(
            "splendor_v1/training/data/h12/"
            "all_h12_replay_buffers_combined.pkl"
        ),
        help=(
            "Legacy H12 replay .pkl in "
            "(observation, dense_policy, scalar_value) format."
        ),
    )

    parser.add_argument(
        "--checkpoint",
        default=(
            "checkpoints/gen_4/"
            "gen_4_legal_scorer_transfer_g3.pt"
        ),
        help=(
            "Transferred Model 4 checkpoint."
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "checkpoints/gen_4"
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help=(
            "Legacy batches score all 1,139 actions. "
            "Start smaller than rich-replay batches."
        ),
    )

    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.10,
    )

    parser.add_argument(
        "--policy-warmup-epochs",
        type=int,
        default=5,
        help=(
            "Phase A epochs training only action_embedding "
            "and legal_action_scorer."
        ),
    )

    parser.add_argument(
        "--policy-lr",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--joint-epochs",
        type=int,
        default=3,
        help=(
            "Phase B full-network epochs. Set 0 to skip."
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
        "--grad-clip",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--device",
        default=None,
        help=(
            "Examples: cuda, cpu. Default chooses CUDA "
            "when available."
        ),
    )

    parser.add_argument(
        "--validate-all-replay",
        action="store_true",
        help=(
            "Validate every legacy replay sample while loading. "
            "Useful once when checking a new replay file."
        ),
    )

    args = parser.parse_args()

    # ========================================================
    # ARGUMENT VALIDATION
    # ========================================================

    if args.batch_size < 1:

        raise ValueError(
            "--batch-size must be >= 1."
        )

    if not (
        0.0
        < args.val_fraction
        < 1.0
    ):

        raise ValueError(
            "--val-fraction must be between 0 and 1."
        )

    if args.policy_warmup_epochs < 0:

        raise ValueError(
            "--policy-warmup-epochs must be >= 0."
        )

    if args.joint_epochs < 0:

        raise ValueError(
            "--joint-epochs must be >= 0."
        )

    if args.policy_lr <= 0:

        raise ValueError(
            "--policy-lr must be positive."
        )

    if args.joint_lr <= 0:

        raise ValueError(
            "--joint-lr must be positive."
        )

    if args.weight_decay < 0:

        raise ValueError(
            "--weight-decay must be >= 0."
        )

    if args.grad_clip is not None and args.grad_clip <= 0:

        raise ValueError(
            "--grad-clip must be positive."
        )

    set_seed(
        args.seed
    )

    if args.device is None:

        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    else:

        device = torch.device(
            args.device
        )

    if device.type == "cuda":

        torch.set_float32_matmul_precision(
            "high"
        )

    pin_memory = (
        device.type
        == "cuda"
    )

    # ========================================================
    # HEADER
    # ========================================================

    print()
    print(
        "=" * 72
    )

    print(
        "MODEL 4 LEGACY H12 PRETRAINING"
    )

    print(
        "=" * 72
    )

    print(
        "Device:               ",
        device,
    )

    print(
        "Checkpoint:           ",
        args.checkpoint,
    )

    print(
        "Replay:               ",
        args.replay,
    )

    print(
        "Batch size:           ",
        args.batch_size,
    )

    print(
        "Policy warmup epochs: ",
        args.policy_warmup_epochs,
    )

    print(
        "Policy LR:            ",
        args.policy_lr,
    )

    print(
        "Joint epochs:         ",
        args.joint_epochs,
    )

    print(
        "Joint LR:             ",
        args.joint_lr,
    )

    print(
        "Weight decay:         ",
        args.weight_decay,
    )

    print(
        "Grad clip:            ",
        args.grad_clip,
    )

    print(
        "Uniform 1139-way CE:  ",
        f"{math.log(1139):.6f}",
    )

    # ========================================================
    # DATA
    # ========================================================

    (
        train_loader,
        val_loader,
        split_info,
    ) = make_model4_legacy_dataloaders(
        replay_path=args.replay,
        batch_size=args.batch_size,
        val_fraction=args.val_fraction,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last_train=False,
        validate_all=(
            args.validate_all_replay
        ),
    )

    print()
    print(
        "=" * 72
    )

    print(
        "LEGACY REPLAY SPLIT"
    )

    print(
        "=" * 72
    )

    print(
        "Total positions: ",
        f"{split_info['dataset_size']:,}",
    )

    print(
        "Train positions: ",
        f"{split_info['train_size']:,}",
    )

    print(
        "Val positions:   ",
        f"{split_info['val_size']:,}",
    )

    print(
        "Split index:     ",
        f"{split_info['split_index']:,}",
    )

    print(
        "Strategy:         contiguous positions"
    )

    # ========================================================
    # MODEL
    # ========================================================

    (
        model,
        source_checkpoint,
    ) = load_model4(
        checkpoint_path=args.checkpoint,
        device=device,
    )

    total_parameter_count = count_parameters(
        model.parameters()
    )

    print()
    print(
        "Model 4 parameters:",
        f"{total_parameter_count:,}",
    )

    baseline_metrics = evaluate(
        model=model,
        data_loader=val_loader,
        device=device,
    )

    print_validation_metrics(
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
            split_info[
                "split_index"
            ],

        "policy_warmup_epochs":
            args.policy_warmup_epochs,

        "policy_lr":
            args.policy_lr,

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

        "legacy_policy_candidates":
            1139,
    }

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    warmup_best_path = (
        output_dir
        / "gen_4_policy_warmup_best.pt"
    )

    joint_best_path = (
        output_dir
        / "gen_4_joint_best.pt"
    )

    final_path = (
        output_dir
        / "gen_4_pretrained_final.pt"
    )

    # ========================================================
    # PHASE A: NEW POLICY COMPONENTS ONLY
    # ========================================================

    best_policy_val_loss = float(
        "inf"
    )

    best_warmup_metrics = None

    if args.policy_warmup_epochs > 0:

        print()
        print(
            "=" * 72
        )

        print(
            "PHASE A: MODEL 4 POLICY WARMUP"
        )

        print(
            "=" * 72
        )

        freeze_except_model4_policy(
            model
        )

        phase_a_trainable = (
            trainable_parameters(
                model
            )
        )

        print(
            "Trainable parameters:",
            f"{count_parameters(phase_a_trainable):,}",
        )

        print(
            "Frozen transferred trunk/WDL:",
            "YES",
        )

        policy_optimizer = (
            torch.optim.AdamW(
                phase_a_trainable,
                lr=args.policy_lr,
                weight_decay=(
                    args.weight_decay
                ),
            )
        )

        for epoch in range(
            1,
            args.policy_warmup_epochs + 1,
        ):

            train_metrics = (
                train_policy_warmup_epoch(
                    model=model,
                    data_loader=train_loader,
                    optimizer=policy_optimizer,
                    device=device,
                    grad_clip=args.grad_clip,
                )
            )

            val_metrics = evaluate(
                model=model,
                data_loader=val_loader,
                device=device,
            )

            print()
            print(
                f"Policy warmup epoch "
                f"{epoch}/{args.policy_warmup_epochs}"
            )

            print(
                "  TRAIN | "
                f"policy={train_metrics['policy_loss']:.6f} | "
                f"KL={train_metrics['policy_kl']:.6f} | "
                f"top1={train_metrics['policy_top1']:.3%} | "
                f"grad={train_metrics['grad_norm']:.4f}"
            )

            print(
                "  VAL   | "
                f"policy={val_metrics['policy_loss']:.6f} | "
                f"KL={val_metrics['policy_kl']:.6f} | "
                f"top1={val_metrics['policy_top1']:.3%} | "
                f"WDL={val_metrics['wdl_loss']:.6f}"
            )

            if (
                val_metrics[
                    "policy_loss"
                ]
                < best_policy_val_loss
            ):

                best_policy_val_loss = (
                    val_metrics[
                        "policy_loss"
                    ]
                )

                best_warmup_metrics = (
                    val_metrics
                )

                save_checkpoint(
                    path=warmup_best_path,
                    model=model,
                    source_checkpoint_path=(
                        args.checkpoint
                    ),
                    replay_path=args.replay,
                    phase="policy_warmup_best",
                    epoch=epoch,
                    metrics=val_metrics,
                    config=config,
                )

                print(
                    "  Saved new best policy warmup:",
                    warmup_best_path,
                )

        # Start joint training from the best warmup checkpoint,
        # not automatically from the last warmup epoch.
        (
            model,
            _,
        ) = load_model4(
            checkpoint_path=warmup_best_path,
            device=device,
        )

        print_validation_metrics(
            "BEST POLICY WARMUP",
            best_warmup_metrics,
        )

    # ========================================================
    # PHASE B: JOINT FULL-NETWORK TRAINING
    # ========================================================

    best_joint_val_loss = float(
        "inf"
    )

    best_joint_metrics = None

    if args.joint_epochs > 0:

        print()
        print(
            "=" * 72
        )

        print(
            "PHASE B: JOINT MODEL 4 TRAINING"
        )

        print(
            "=" * 72
        )

        unfreeze_all(
            model
        )

        phase_b_trainable = (
            trainable_parameters(
                model
            )
        )

        print(
            "Trainable parameters:",
            f"{count_parameters(phase_b_trainable):,}",
        )

        joint_optimizer = (
            torch.optim.AdamW(
                phase_b_trainable,
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
                    grad_clip=args.grad_clip,
                )
            )

            val_metrics = evaluate(
                model=model,
                data_loader=val_loader,
                device=device,
            )

            print()
            print(
                f"Joint epoch "
                f"{epoch}/{args.joint_epochs}"
            )

            print(
                "  TRAIN | "
                f"total={train_metrics['total_loss']:.6f} | "
                f"policy={train_metrics['policy_loss']:.6f} | "
                f"WDL={train_metrics['wdl_loss']:.6f} | "
                f"KL={train_metrics['policy_kl']:.6f} | "
                f"top1={train_metrics['policy_top1']:.3%} | "
                f"grad={train_metrics['grad_norm']:.4f}"
            )

            print(
                "  VAL   | "
                f"total={val_metrics['total_loss']:.6f} | "
                f"policy={val_metrics['policy_loss']:.6f} | "
                f"WDL={val_metrics['wdl_loss']:.6f} | "
                f"KL={val_metrics['policy_kl']:.6f} | "
                f"top1={val_metrics['policy_top1']:.3%} | "
                f"WDLacc={val_metrics['wdl_accuracy']:.3%}"
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
                    path=joint_best_path,
                    model=model,
                    source_checkpoint_path=(
                        args.checkpoint
                    ),
                    replay_path=args.replay,
                    phase="joint_best",
                    epoch=epoch,
                    metrics=val_metrics,
                    config=config,
                )

                print(
                    "  Saved new best joint checkpoint:",
                    joint_best_path,
                )

        (
            model,
            _,
        ) = load_model4(
            checkpoint_path=joint_best_path,
            device=device,
        )

        print_validation_metrics(
            "BEST JOINT",
            best_joint_metrics,
        )

    # ========================================================
    # FINAL CHECKPOINT
    # ========================================================

    final_metrics = evaluate(
        model=model,
        data_loader=val_loader,
        device=device,
    )

    final_phase = (
        "joint_best"
        if args.joint_epochs > 0
        else (
            "policy_warmup_best"
            if args.policy_warmup_epochs > 0
            else "source_checkpoint"
        )
    )

    save_checkpoint(
        path=final_path,
        model=model,
        source_checkpoint_path=args.checkpoint,
        replay_path=args.replay,
        phase=final_phase,
        epoch=(
            args.joint_epochs
            if args.joint_epochs > 0
            else args.policy_warmup_epochs
        ),
        metrics=final_metrics,
        config=config,
    )

    print_validation_metrics(
        "FINAL SELECTED MODEL",
        final_metrics,
    )

    print()
    print(
        "=" * 72
    )

    print(
        "MODEL 4 LEGACY PRETRAINING COMPLETE"
    )

    print(
        "=" * 72
    )

    if args.policy_warmup_epochs > 0:

        print(
            "Best policy warmup:",
            warmup_best_path,
        )

    if args.joint_epochs > 0:

        print(
            "Best joint:        ",
            joint_best_path,
        )

    print(
        "Final selected:     ",
        final_path,
    )


if __name__ == "__main__":
    main()
