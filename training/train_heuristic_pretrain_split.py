import argparse
import pickle
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from splendor_v1.network.model_2_attention import SplendorNetwork


class ReplaySampleDataset(Dataset):
    def __init__(self, samples, name="dataset"):
        self.samples = samples
        self.name = name

        if len(self.samples) == 0:
            raise ValueError(f"{name} is empty.")

        obs, policy, value = self.samples[0]

        print(f"{name}")
        print(f"  samples:       {len(self.samples):,}")
        print(f"  observation:   {np.asarray(obs).shape}")
        print(f"  policy:        {np.asarray(policy).shape}")
        print(f"  example value: {value}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        observation, policy, value = self.samples[idx]

        observation = torch.as_tensor(
            observation,
            dtype=torch.float32,
        )

        policy = torch.as_tensor(
            policy,
            dtype=torch.float32,
        )

        value = torch.tensor(
            value,
            dtype=torch.float32,
        )

        return (
            observation,
            policy,
            value,
        )


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _validate_range(start, end, buffer_len, replay_path):
    if not isinstance(start, int) or not isinstance(end, int):
        raise TypeError(
            f"{replay_path}: game record start_index/end_index must be ints."
        )

    if start < 0 or end < start or end > buffer_len:
        raise ValueError(
            f"{replay_path}: invalid game record range "
            f"[{start}, {end}) for buffer length {buffer_len}."
        )


def split_replay_file(replay_path: str):
    """
    Load one replay pickle and return:
        train_samples, val_samples, summary

    Supported split formats, in priority order:

    1. game_records
       Each record must contain:
           start_index
           end_index
           split == "train" or "val"

       This is the preferred format because whole games stay together.

    2. Explicit train_indices / val_indices

    3. Top-level split
       split == "train" -> entire buffer goes to training
       split == "val"   -> entire buffer goes to validation

       This supports migrated legacy files that are intentionally marked
       training-only with:
           split = "train"
           is_legacy = True

    The loader intentionally does NOT invent a random position-level split.
    If a file has no recognized split metadata, it raises an error.
    """
    replay_path = Path(replay_path)

    if not replay_path.exists():
        raise FileNotFoundError(
            f"Replay buffer not found: {replay_path}"
        )

    with replay_path.open("rb") as f:
        data = pickle.load(f)

    if not isinstance(data, dict):
        raise TypeError(
            f"{replay_path}: expected replay .pkl to contain a dict."
        )

    if "buffer" not in data:
        raise KeyError(
            f"{replay_path}: replay dict does not contain a 'buffer' key."
        )

    buffer = data["buffer"]

    if not isinstance(buffer, list):
        raise TypeError(
            f"{replay_path}: expected data['buffer'] to be a list."
        )

    if len(buffer) == 0:
        raise ValueError(
            f"{replay_path}: replay buffer is empty."
        )

    train_samples = []
    val_samples = []

    summary = {
        "path": str(replay_path),
        "total": len(buffer),
        "train": 0,
        "val": 0,
        "strategy": None,
        "is_legacy": bool(data.get("is_legacy", False)),
        "train_games": None,
        "val_games": None,
    }

    # ============================================================
    # 1. PREFERRED: WHOLE-GAME SPLIT METADATA
    # ============================================================
    game_records = data.get("game_records")

    if game_records:
        train_games = 0
        val_games = 0

        for record in game_records:
            if not isinstance(record, dict):
                raise TypeError(
                    f"{replay_path}: each game_record must be a dict."
                )

            if "start_index" not in record or "end_index" not in record:
                raise KeyError(
                    f"{replay_path}: game_record missing "
                    "'start_index' or 'end_index'."
                )

            split = record.get("split")

            # Optional compatibility with files that record game IDs
            # separately rather than embedding a split in each record.
            if split is None:
                game_id = record.get("game_id")
                train_ids = set(data.get("train_game_ids", []))
                val_ids = set(data.get("val_game_ids", []))

                if game_id in train_ids:
                    split = "train"
                elif game_id in val_ids:
                    split = "val"

            if split not in {"train", "val"}:
                raise ValueError(
                    f"{replay_path}: game_record has invalid/missing split: "
                    f"{split!r}. Expected 'train' or 'val'."
                )

            start = record["start_index"]
            end = record["end_index"]

            _validate_range(
                start,
                end,
                len(buffer),
                replay_path,
            )

            if split == "train":
                train_samples.extend(buffer[start:end])
                train_games += 1
            else:
                val_samples.extend(buffer[start:end])
                val_games += 1

        summary["strategy"] = "game_records"
        summary["train_games"] = train_games
        summary["val_games"] = val_games

    # ============================================================
    # 2. EXPLICIT SAMPLE INDICES
    # ============================================================
    elif (
        "train_indices" in data
        or "val_indices" in data
    ):
        train_indices = data.get("train_indices", [])
        val_indices = data.get("val_indices", [])

        for idx in train_indices:
            if idx < 0 or idx >= len(buffer):
                raise IndexError(
                    f"{replay_path}: train index out of range: {idx}"
                )
            train_samples.append(buffer[idx])

        for idx in val_indices:
            if idx < 0 or idx >= len(buffer):
                raise IndexError(
                    f"{replay_path}: val index out of range: {idx}"
                )
            val_samples.append(buffer[idx])

        summary["strategy"] = "explicit_indices"

    # ============================================================
    # 3. TOP-LEVEL FILE SPLIT
    # ============================================================
    elif data.get("split") in {"train", "val"}:
        split = data["split"]

        if split == "train":
            train_samples.extend(buffer)
        else:
            val_samples.extend(buffer)

        summary["strategy"] = f"file_level_{split}"

    else:
        raise ValueError(
            f"{replay_path}: no recognized split metadata found.\n"
            "Expected one of:\n"
            "  - game_records with train/val split tags\n"
            "  - train_indices / val_indices\n"
            "  - top-level split='train' or split='val'\n"
            "Refusing to perform a random position-level split because "
            "that can leak positions from the same game across train/validation."
        )

    summary["train"] = len(train_samples)
    summary["val"] = len(val_samples)

    return (
        train_samples,
        val_samples,
        summary,
    )


def load_and_combine_replays(replay_paths):
    all_train_samples = []
    all_val_samples = []
    summaries = []

    for replay_path in replay_paths:
        (
            train_samples,
            val_samples,
            summary,
        ) = split_replay_file(replay_path)

        all_train_samples.extend(train_samples)
        all_val_samples.extend(val_samples)
        summaries.append(summary)

    if len(all_train_samples) == 0:
        raise ValueError(
            "Combined training split is empty."
        )

    if len(all_val_samples) == 0:
        raise ValueError(
            "Combined validation split is empty. "
            "At least one replay file must contribute validation games."
        )

    return (
        all_train_samples,
        all_val_samples,
        summaries,
    )


def compute_losses(
    policy_logits,
    predicted_value,
    target_policy,
    target_value,
):
    log_probs = F.log_softmax(
        policy_logits,
        dim=1,
    )

    policy_loss = -(
        target_policy * log_probs
    ).sum(
        dim=1
    ).mean()

    predicted_value = (
        predicted_value.squeeze(-1)
    )

    value_loss = F.mse_loss(
        predicted_value,
        target_value,
    )

    total_loss = (
        policy_loss
        + value_loss
    )

    return (
        total_loss,
        policy_loss,
        value_loss,
    )


@torch.no_grad()
def batch_metrics(
    policy_logits,
    predicted_value,
    target_policy,
    target_value,
):
    predicted_action = (
        policy_logits.argmax(
            dim=1
        )
    )

    target_action = (
        target_policy.argmax(
            dim=1
        )
    )

    policy_top1 = (
        predicted_action
        == target_action
    ).float().mean().item()

    predicted_value = (
        predicted_value.squeeze(-1)
    )

    predicted_sign = torch.where(
        predicted_value >= 0,
        torch.ones_like(
            predicted_value
        ),
        -torch.ones_like(
            predicted_value
        ),
    )

    target_sign = torch.where(
        target_value >= 0,
        torch.ones_like(
            target_value
        ),
        -torch.ones_like(
            target_value
        ),
    )

    value_sign = (
        predicted_sign
        == target_sign
    ).float().mean().item()

    return (
        policy_top1,
        value_sign,
    )


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    grad_clip,
):
    model.train()

    total_examples = 0
    total_loss_sum = 0.0
    policy_loss_sum = 0.0
    value_loss_sum = 0.0
    policy_top1_sum = 0.0
    value_sign_sum = 0.0

    for (
        observations,
        target_policies,
        target_values,
    ) in loader:

        observations = observations.to(
            device,
            non_blocking=True,
        )

        target_policies = target_policies.to(
            device,
            non_blocking=True,
        )

        target_values = target_values.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        (
            policy_logits,
            predicted_values,
        ) = model(
            observations
        )

        (
            loss,
            policy_loss,
            value_loss,
        ) = compute_losses(
            policy_logits,
            predicted_values,
            target_policies,
            target_values,
        )

        loss.backward()

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                grad_clip,
            )

        optimizer.step()

        batch_size = (
            observations.shape[0]
        )

        (
            policy_top1,
            value_sign,
        ) = batch_metrics(
            policy_logits,
            predicted_values,
            target_policies,
            target_values,
        )

        total_examples += batch_size

        total_loss_sum += (
            loss.item()
            * batch_size
        )

        policy_loss_sum += (
            policy_loss.item()
            * batch_size
        )

        value_loss_sum += (
            value_loss.item()
            * batch_size
        )

        policy_top1_sum += (
            policy_top1
            * batch_size
        )

        value_sign_sum += (
            value_sign
            * batch_size
        )

    return {
        "loss":
            total_loss_sum
            / total_examples,

        "policy_loss":
            policy_loss_sum
            / total_examples,

        "value_loss":
            value_loss_sum
            / total_examples,

        "policy_top1":
            policy_top1_sum
            / total_examples,

        "value_sign":
            value_sign_sum
            / total_examples,
    }


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
):
    model.eval()

    total_examples = 0
    total_loss_sum = 0.0
    policy_loss_sum = 0.0
    value_loss_sum = 0.0
    policy_top1_sum = 0.0
    value_sign_sum = 0.0

    for (
        observations,
        target_policies,
        target_values,
    ) in loader:

        observations = observations.to(
            device,
            non_blocking=True,
        )

        target_policies = target_policies.to(
            device,
            non_blocking=True,
        )

        target_values = target_values.to(
            device,
            non_blocking=True,
        )

        (
            policy_logits,
            predicted_values,
        ) = model(
            observations
        )

        (
            loss,
            policy_loss,
            value_loss,
        ) = compute_losses(
            policy_logits,
            predicted_values,
            target_policies,
            target_values,
        )

        batch_size = (
            observations.shape[0]
        )

        (
            policy_top1,
            value_sign,
        ) = batch_metrics(
            policy_logits,
            predicted_values,
            target_policies,
            target_values,
        )

        total_examples += (
            batch_size
        )

        total_loss_sum += (
            loss.item()
            * batch_size
        )

        policy_loss_sum += (
            policy_loss.item()
            * batch_size
        )

        value_loss_sum += (
            value_loss.item()
            * batch_size
        )

        policy_top1_sum += (
            policy_top1
            * batch_size
        )

        value_sign_sum += (
            value_sign
            * batch_size
        )

    return {
        "loss":
            total_loss_sum
            / total_examples,

        "policy_loss":
            policy_loss_sum
            / total_examples,

        "value_loss":
            value_loss_sum
            / total_examples,

        "policy_top1":
            policy_top1_sum
            / total_examples,

        "value_sign":
            value_sign_sum
            / total_examples,
    }


def save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    train_metrics,
    val_metrics,
    args,
    replay_summaries,
):
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "epoch":
                epoch,

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "train_metrics":
                train_metrics,

            "val_metrics":
                val_metrics,

            "args":
                vars(args),

            "split_strategy":
                "replay_metadata",

            "replay_summaries":
                replay_summaries,
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Blake heuristic pretraining using replay-provided "
            "train/validation split metadata."
        )
    )

    parser.add_argument(
        "--replay",
        type=str,
        nargs="+",
        required=True,
        help=(
            "One or more replay .pkl files. "
            "Training and validation samples are extracted from "
            "the split metadata in each file and then concatenated."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=(
            "checkpoints/"
            "h16_pretrain"
        ),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=40,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )

    args = parser.parse_args()

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
        "BLAKE HEURISTIC PRETRAINING"
    )
    print(
        "REPLAY-METADATA TRAIN / VALIDATION SPLIT"
    )
    print(
        "=" * 70
    )

    print(
        f"Device:         "
        f"{device}"
    )

    print(
        f"Epochs:         "
        f"{args.epochs}"
    )

    print(
        f"Batch size:     "
        f"{args.batch_size}"
    )

    print(
        f"Learning rate:  "
        f"{args.learning_rate}"
    )

    print(
        f"Weight decay:   "
        f"{args.weight_decay}"
    )

    print()

    (
        train_samples,
        val_samples,
        replay_summaries,
    ) = load_and_combine_replays(
        args.replay
    )

    print(
        "=" * 70
    )
    print(
        "REPLAY SPLITS"
    )
    print(
        "=" * 70
    )

    for summary in replay_summaries:
        print(
            f"Replay:         "
            f"{summary['path']}"
        )
        print(
            f"  strategy:     "
            f"{summary['strategy']}"
        )
        print(
            f"  legacy:       "
            f"{summary['is_legacy']}"
        )
        print(
            f"  total:        "
            f"{summary['total']:,}"
        )
        print(
            f"  train:        "
            f"{summary['train']:,}"
        )
        print(
            f"  validation:   "
            f"{summary['val']:,}"
        )

        if summary["train_games"] is not None:
            print(
                f"  train games:  "
                f"{summary['train_games']:,}"
            )
            print(
                f"  val games:    "
                f"{summary['val_games']:,}"
            )

        print()

    total_combined = (
        len(train_samples)
        + len(val_samples)
    )

    print(
        f"Combined train samples: "
        f"{len(train_samples):,}"
    )

    print(
        f"Combined val samples:   "
        f"{len(val_samples):,}"
    )

    print(
        f"Combined total samples: "
        f"{total_combined:,}"
    )

    print(
        f"Effective val fraction: "
        f"{len(val_samples) / total_combined:.2%}"
    )

    print()

    train_dataset = ReplaySampleDataset(
        train_samples,
        name="TRAIN DATASET",
    )

    print()

    val_dataset = ReplaySampleDataset(
        val_samples,
        name="VALIDATION DATASET",
    )

    print()

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

    model = (
        SplendorNetwork()
        .to(
            device=device,
            dtype=torch.float32,
        )
    )

    optimizer = (
        torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=(
                args.weight_decay
            ),
        )
    )

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"Fresh Blake parameters: "
        f"{parameter_count:,}"
    )

    baseline_metrics = evaluate(
        model,
        val_loader,
        device,
    )

    print()
    print(
        "Before training"
    )

    print(
        f"  val loss:        "
        f"{baseline_metrics['loss']:.4f}"
    )

    print(
        f"  val policy loss: "
        f"{baseline_metrics['policy_loss']:.4f}"
    )

    print(
        f"  val value loss:  "
        f"{baseline_metrics['value_loss']:.4f}"
    )

    print(
        f"  policy top-1:    "
        f"{baseline_metrics['policy_top1']:.2%}"
    )

    print(
        f"  value sign acc:  "
        f"{baseline_metrics['value_sign']:.2%}"
    )

    print()

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_val_loss = float(
        "inf"
    )

    best_epoch = None

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        train_metrics = (
            train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                device=device,
                grad_clip=args.grad_clip,
            )
        )

        val_metrics = (
            evaluate(
                model=model,
                loader=val_loader,
                device=device,
            )
        )

        print(
            "=" * 70
        )

        print(
            f"EPOCH "
            f"{epoch}/"
            f"{args.epochs}"
        )

        print(
            "=" * 70
        )

        print(
            "TRAIN"
            f" | loss "
            f"{train_metrics['loss']:.4f}"
            f" | policy "
            f"{train_metrics['policy_loss']:.4f}"
            f" | value "
            f"{train_metrics['value_loss']:.4f}"
            f" | top1 "
            f"{train_metrics['policy_top1']:.2%}"
            f" | value sign "
            f"{train_metrics['value_sign']:.2%}"
        )

        print(
            "VAL  "
            f" | loss "
            f"{val_metrics['loss']:.4f}"
            f" | policy "
            f"{val_metrics['policy_loss']:.4f}"
            f" | value "
            f"{val_metrics['value_loss']:.4f}"
            f" | top1 "
            f"{val_metrics['policy_top1']:.2%}"
            f" | value sign "
            f"{val_metrics['value_sign']:.2%}"
        )

        epoch_path = (
            output_dir
            / (
                "heuristic_pretrain_"
                f"epoch_{epoch:02d}.pt"
            )
        )

        save_checkpoint(
            path=epoch_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            train_metrics=(
                train_metrics
            ),
            val_metrics=(
                val_metrics
            ),
            args=args,
            replay_summaries=(
                replay_summaries
            ),
        )

        if (
            val_metrics["loss"]
            < best_val_loss
        ):
            best_val_loss = (
                val_metrics["loss"]
            )

            best_epoch = epoch

            best_path = (
                output_dir
                / (
                    "heuristic_pretrain_"
                    "best.pt"
                )
            )

            save_checkpoint(
                path=best_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                train_metrics=(
                    train_metrics
                ),
                val_metrics=(
                    val_metrics
                ),
                args=args,
                replay_summaries=(
                    replay_summaries
                ),
            )

            print(
                f"Saved new best checkpoint: "
                f"{best_path}"
            )

        print()

    final_weights_path = (
        output_dir
        / (
            "heuristic_pretrained_"
            "weights.pt"
        )
    )

    torch.save(
        model.state_dict(),
        final_weights_path,
    )

    print(
        "=" * 70
    )

    print(
        "TRAINING COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"Final weights: "
        f"{final_weights_path}"
    )

    print(
        f"Best checkpoint: "
        f"{output_dir / 'heuristic_pretrain_best.pt'}"
    )

    print(
        f"Best epoch:      "
        f"{best_epoch}"
    )

    print(
        f"Best val loss:   "
        f"{best_val_loss:.4f}"
    )

    print(
        f"Train samples:   "
        f"{len(train_samples):,}"
    )

    print(
        f"Val samples:     "
        f"{len(val_samples):,}"
    )


if __name__ == "__main__":
    main()
