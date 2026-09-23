from __future__ import annotations

import argparse
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
from splendor_v1.training_v4.batch_builder_v4 import (
    load_replay,
    make_model4_dataloader,
)
from splendor_v1.network.losses_4_wdl import (
    policy_wdl_loss,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def extract_model_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        raise TypeError("Expected checkpoint to be a dictionary.")

    if checkpoint and all(torch.is_tensor(v) for v in checkpoint.values()):
        return checkpoint

    for key in ("model_state_dict", "state_dict", "model"):
        state_dict = checkpoint.get(key)
        if isinstance(state_dict, dict):
            return state_dict

    raise KeyError("Could not find model state_dict in checkpoint.")


def load_model4(checkpoint_path: str | Path, device: torch.device):
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Model 4 checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    model = SplendorNetwork()
    model.load_state_dict(
        extract_model_state_dict(checkpoint),
        strict=True,
    )

    model = model.to(
        device=device,
        dtype=torch.float32,
    )
    model.eval()

    return model, checkpoint


def save_checkpoint(
    path: str | Path,
    model,
    source_checkpoint_path,
    replay_path,
    epoch,
    metrics,
    config,
    phase,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model_state_dict": model.state_dict(),
        "model_generation": 4,
        "architecture": "attention_wdl_legal_action_scorer",
        "source_checkpoint": str(source_checkpoint_path),
        "fine_tune_replay": str(replay_path),
        "fine_tune_data_format": "rich_replay_legal_actions",
        "fine_tune_phase": phase,
        "fine_tune_epoch": int(epoch),
        "metrics": metrics,
        "config": config,
    }

    temp_path = Path(str(path) + ".tmp")
    torch.save(payload, temp_path)
    os.replace(temp_path, path)


def move_batch_to_device(batch, device):
    return {
        "observations": batch["observations"].to(
            device,
            non_blocking=True,
        ),
        "legal_action_ids": batch["legal_action_ids"].to(
            device,
            non_blocking=True,
        ),
        "legal_action_mask": batch["legal_action_mask"].to(
            device,
            non_blocking=True,
        ),
        "target_policy": batch["target_policy"].to(
            device,
            non_blocking=True,
        ),
        "target_wdl": batch["target_wdl"].to(
            device,
            non_blocking=True,
        ),
        "action_counts": batch["action_counts"].to(
            device,
            non_blocking=True,
        ),
    }


def weighted_average(total, count):
    if count == 0:
        return float("nan")
    return float(total / count)


def target_scalar_from_wdl(target_wdl):
    target_value = torch.zeros_like(
        target_wdl,
        dtype=torch.float32,
    )

    target_value = torch.where(
        target_wdl == WDL_WIN,
        torch.ones_like(target_value),
        target_value,
    )

    target_value = torch.where(
        target_wdl == WDL_LOSS,
        -torch.ones_like(target_value),
        target_value,
    )

    return target_value


@torch.inference_mode()
def evaluate(model, data_loader, device):
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

    action_count_sum = 0
    action_count_min = None
    action_count_max = None

    for raw_batch in data_loader:
        batch = move_batch_to_device(
            raw_batch,
            device,
        )

        policy_logits, wdl_logits = model(
            batch["observations"],
            batch["legal_action_ids"],
            batch["legal_action_mask"],
        )

        (
            total_loss,
            policy_loss,
            wdl_loss,
            policy_kl,
        ) = policy_wdl_loss(
            policy_logits=policy_logits,
            wdl_logits=wdl_logits,
            target_policy=batch["target_policy"],
            target_wdl=batch["target_wdl"],
            legal_action_mask=batch["legal_action_mask"],
        )

        target_log_probs = torch.log(
            batch["target_policy"].clamp_min(1e-8)
        )
        target_entropy = -(
            batch["target_policy"]
            * target_log_probs
        ).sum(dim=-1).mean()

        batch_size = int(batch["observations"].shape[0])
        total_samples += batch_size

        total_loss_sum += total_loss.item() * batch_size
        policy_loss_sum += policy_loss.item() * batch_size
        wdl_loss_sum += wdl_loss.item() * batch_size
        policy_kl_sum += policy_kl.item() * batch_size
        target_entropy_sum += target_entropy.item() * batch_size

        policy_top1_correct += int(
            (
                policy_logits.argmax(dim=-1)
                == batch["target_policy"].argmax(dim=-1)
            ).sum().item()
        )

        wdl_correct += int(
            (
                wdl_logits.argmax(dim=-1)
                == batch["target_wdl"]
            ).sum().item()
        )

        wdl_probabilities = F.softmax(
            wdl_logits,
            dim=-1,
        )

        predicted_value = (
            wdl_probabilities[:, WDL_WIN]
            - wdl_probabilities[:, WDL_LOSS]
        )

        target_value = target_scalar_from_wdl(
            batch["target_wdl"]
        )

        scalar_abs_error_sum += float(
            (
                predicted_value
                - target_value
            ).abs().sum().item()
        )

        predicted_ldw_sum += (
            wdl_probabilities
            .sum(dim=0)
            .detach()
            .cpu()
            .double()
        )

        target_ldw_count += (
            F.one_hot(
                batch["target_wdl"],
                num_classes=3,
            )
            .sum(dim=0)
            .detach()
            .cpu()
            .double()
        )

        counts = batch["action_counts"]
        action_count_sum += int(counts.sum().item())

        batch_min = int(counts.min().item())
        batch_max = int(counts.max().item())

        if action_count_min is None or batch_min < action_count_min:
            action_count_min = batch_min

        if action_count_max is None or batch_max > action_count_max:
            action_count_max = batch_max

    return {
        "total_loss": weighted_average(
            total_loss_sum,
            total_samples,
        ),
        "policy_loss": weighted_average(
            policy_loss_sum,
            total_samples,
        ),
        "wdl_loss": weighted_average(
            wdl_loss_sum,
            total_samples,
        ),
        "policy_kl": weighted_average(
            policy_kl_sum,
            total_samples,
        ),
        "target_policy_entropy": weighted_average(
            target_entropy_sum,
            total_samples,
        ),
        "policy_top1": weighted_average(
            policy_top1_correct,
            total_samples,
        ),
        "wdl_accuracy": weighted_average(
            wdl_correct,
            total_samples,
        ),
        "scalar_mae": weighted_average(
            scalar_abs_error_sum,
            total_samples,
        ),
        "loss_probability": float(
            predicted_ldw_sum[WDL_LOSS]
            / total_samples
        ),
        "draw_probability": float(
            predicted_ldw_sum[WDL_DRAW]
            / total_samples
        ),
        "win_probability": float(
            predicted_ldw_sum[WDL_WIN]
            / total_samples
        ),
        "target_loss_fraction": float(
            target_ldw_count[WDL_LOSS]
            / total_samples
        ),
        "target_draw_fraction": float(
            target_ldw_count[WDL_DRAW]
            / total_samples
        ),
        "target_win_fraction": float(
            target_ldw_count[WDL_WIN]
            / total_samples
        ),
        "mean_legal_actions": weighted_average(
            action_count_sum,
            total_samples,
        ),
        "min_legal_actions": int(action_count_min),
        "max_legal_actions": int(action_count_max),
        "samples": int(total_samples),
    }


def train_epoch(
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

    policy_top1_correct = 0
    wdl_correct = 0

    grad_norm_sum = 0.0
    grad_batches = 0
    action_count_sum = 0

    for raw_batch in data_loader:
        batch = move_batch_to_device(
            raw_batch,
            device,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        policy_logits, wdl_logits = model(
            batch["observations"],
            batch["legal_action_ids"],
            batch["legal_action_mask"],
        )

        (
            total_loss,
            policy_loss,
            wdl_loss,
            policy_kl,
        ) = policy_wdl_loss(
            policy_logits=policy_logits,
            wdl_logits=wdl_logits,
            target_policy=batch["target_policy"],
            target_wdl=batch["target_wdl"],
            legal_action_mask=batch["legal_action_mask"],
        )

        total_loss.backward()

        if grad_clip is None:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=float("inf"),
            )
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=grad_clip,
            )

        optimizer.step()

        batch_size = int(batch["observations"].shape[0])
        total_samples += batch_size

        total_loss_sum += total_loss.item() * batch_size
        policy_loss_sum += policy_loss.item() * batch_size
        wdl_loss_sum += wdl_loss.item() * batch_size
        policy_kl_sum += policy_kl.item() * batch_size

        policy_top1_correct += int(
            (
                policy_logits.argmax(dim=-1)
                == batch["target_policy"].argmax(dim=-1)
            ).sum().item()
        )

        wdl_correct += int(
            (
                wdl_logits.argmax(dim=-1)
                == batch["target_wdl"]
            ).sum().item()
        )

        grad_norm_sum += float(grad_norm.item())
        grad_batches += 1

        action_count_sum += int(
            batch["action_counts"].sum().item()
        )

    return {
        "total_loss": weighted_average(
            total_loss_sum,
            total_samples,
        ),
        "policy_loss": weighted_average(
            policy_loss_sum,
            total_samples,
        ),
        "wdl_loss": weighted_average(
            wdl_loss_sum,
            total_samples,
        ),
        "policy_kl": weighted_average(
            policy_kl_sum,
            total_samples,
        ),
        "policy_top1": weighted_average(
            policy_top1_correct,
            total_samples,
        ),
        "wdl_accuracy": weighted_average(
            wdl_correct,
            total_samples,
        ),
        "grad_norm": weighted_average(
            grad_norm_sum,
            grad_batches,
        ),
        "mean_legal_actions": weighted_average(
            action_count_sum,
            total_samples,
        ),
    }


def print_validation_metrics(title, metrics):
    print()
    print(title)

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
    print(
        "  legal actions:    ",
        (
            f"mean={metrics['mean_legal_actions']:.2f}, "
            f"min={metrics['min_legal_actions']}, "
            f"max={metrics['max_legal_actions']}"
        ),
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune pretrained Model 4 on rich replay data "
            "with true legal-action candidate sets and MCTS visit targets."
        )
    )

    parser.add_argument(
        "--checkpoint",
        default=(
            "checkpoints/gen_4/"
            "gen_4_joint_best.pt"
        ),
        help="Best legacy-pretrained Model 4 checkpoint.",
    )

    parser.add_argument(
        "--replay",
        required=True,
        help=(
            "Rich replay .pkl containing game metadata, "
            "policy/legal action IDs, and visit counts."
        ),
    )

    parser.add_argument(
        "--output-dir",
        default="checkpoints/gen_4",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--learning-rate",
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
    )

    args = parser.parse_args()

    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1.")

    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1.")

    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive.")

    if args.weight_decay < 0:
        raise ValueError("--weight-decay must be >= 0.")

    if args.grad_clip is not None and args.grad_clip <= 0:
        raise ValueError("--grad-clip must be positive.")

    set_seed(args.seed)

    if args.device is None:
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
    else:
        device = torch.device(args.device)

    if device.type == "cuda":
        torch.set_float32_matmul_precision(
            "high"
        )

    pin_memory = (
        device.type == "cuda"
    )

    print()
    print("=" * 72)
    print("MODEL 4 RICH-REPLAY FINE-TUNING")
    print("=" * 72)

    print("Device:         ", device)
    print("Checkpoint:     ", args.checkpoint)
    print("Replay:         ", args.replay)
    print("Epochs:         ", args.epochs)
    print("Batch size:     ", args.batch_size)
    print("Learning rate:  ", args.learning_rate)
    print("Weight decay:   ", args.weight_decay)
    print("Grad clip:      ", args.grad_clip)

    replay = load_replay(
        args.replay
    )

    train_loader = make_model4_dataloader(
        replay=replay,
        split="train",
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    val_loader = make_model4_dataloader(
        replay=replay,
        split="val",
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    train_positions = len(
        train_loader.dataset
    )
    val_positions = len(
        val_loader.dataset
    )
    total_positions = (
        train_positions
        + val_positions
    )

    games = replay["games"]

    if isinstance(games, dict):
        game_values = list(
            games.values()
        )
    else:
        game_values = list(
            games
        )

    train_games = sum(
        1
        for game in game_values
        if game.get("split") == "train"
    )

    val_games = sum(
        1
        for game in game_values
        if game.get("split") == "val"
    )

    print()
    print("=" * 72)
    print("RICH REPLAY SPLIT")
    print("=" * 72)
    print("Total positions: ", f"{total_positions:,}")
    print("Train positions: ", f"{train_positions:,}")
    print("Val positions:   ", f"{val_positions:,}")
    print("Train games:     ", f"{train_games:,}")
    print("Val games:       ", f"{val_games:,}")
    print("Split source:     replay game metadata")

    model, _ = load_model4(
        checkpoint_path=args.checkpoint,
        device=device,
    )

    total_parameters = sum(
        int(parameter.numel())
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        int(parameter.numel())
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print()
    print(
        "Model 4 parameters:    ",
        f"{total_parameters:,}",
    )
    print(
        "Trainable parameters:  ",
        f"{trainable_parameters:,}",
    )

    baseline_metrics = evaluate(
        model=model,
        data_loader=val_loader,
        device=device,
    )

    print_validation_metrics(
        "BEFORE RICH FINE-TUNING",
        baseline_metrics,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    config = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "seed": args.seed,
        "split_strategy": "replay_game_metadata",
        "train_positions": train_positions,
        "val_positions": val_positions,
        "train_games": train_games,
        "val_games": val_games,
    }

    output_dir = Path(
        args.output_dir
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_path = (
        output_dir
        / "gen_4_rich_finetune_best.pt"
    )

    last_path = (
        output_dir
        / "gen_4_rich_finetune_last.pt"
    )

    best_val_total_loss = float(
        "inf"
    )
    best_epoch = None
    best_metrics = None

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        train_metrics = train_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            device=device,
            grad_clip=args.grad_clip,
        )

        val_metrics = evaluate(
            model=model,
            data_loader=val_loader,
            device=device,
        )

        print()
        print("=" * 72)
        print(
            f"RICH FINE-TUNE EPOCH "
            f"{epoch}/{args.epochs}"
        )
        print("=" * 72)

        print(
            "TRAIN | "
            f"total={train_metrics['total_loss']:.6f} | "
            f"policy={train_metrics['policy_loss']:.6f} | "
            f"WDL={train_metrics['wdl_loss']:.6f} | "
            f"KL={train_metrics['policy_kl']:.6f} | "
            f"top1={train_metrics['policy_top1']:.3%} | "
            f"WDLacc={train_metrics['wdl_accuracy']:.3%} | "
            f"grad={train_metrics['grad_norm']:.4f}"
        )

        print(
            "VAL   | "
            f"total={val_metrics['total_loss']:.6f} | "
            f"policy={val_metrics['policy_loss']:.6f} | "
            f"WDL={val_metrics['wdl_loss']:.6f} | "
            f"KL={val_metrics['policy_kl']:.6f} | "
            f"top1={val_metrics['policy_top1']:.3%} | "
            f"WDLacc={val_metrics['wdl_accuracy']:.3%}"
        )

        if (
            val_metrics["total_loss"]
            < best_val_total_loss
        ):
            best_val_total_loss = (
                val_metrics["total_loss"]
            )
            best_epoch = epoch
            best_metrics = val_metrics

            save_checkpoint(
                path=best_path,
                model=model,
                source_checkpoint_path=args.checkpoint,
                replay_path=args.replay,
                epoch=epoch,
                metrics=val_metrics,
                config=config,
                phase="rich_finetune_best",
            )

            print(
                "Saved new best rich checkpoint:",
                best_path,
            )

    last_metrics = evaluate(
        model=model,
        data_loader=val_loader,
        device=device,
    )

    save_checkpoint(
        path=last_path,
        model=model,
        source_checkpoint_path=args.checkpoint,
        replay_path=args.replay,
        epoch=args.epochs,
        metrics=last_metrics,
        config=config,
        phase="rich_finetune_last_epoch",
    )

    best_model, _ = load_model4(
        checkpoint_path=best_path,
        device=device,
    )

    selected_metrics = evaluate(
        model=best_model,
        data_loader=val_loader,
        device=device,
    )

    print_validation_metrics(
        "BEST RICH FINE-TUNED MODEL",
        selected_metrics,
    )

    print()
    print("=" * 72)
    print("MODEL 4 RICH-REPLAY FINE-TUNING COMPLETE")
    print("=" * 72)
    print("Source checkpoint:", args.checkpoint)
    print("Best epoch:      ", best_epoch)
    print("Best checkpoint: ", best_path)
    print("Last epoch:      ", last_path)


if __name__ == "__main__":
    main()
