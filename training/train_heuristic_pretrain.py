import argparse
import pickle
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

from splendor_v1.network.model import SplendorNetwork


class HeuristicReplayDataset(Dataset):
    def __init__(self, replay_path: str):
        replay_path = Path(replay_path)

        if not replay_path.exists():
            raise FileNotFoundError(f"Replay buffer not found: {replay_path}")

        with replay_path.open("rb") as f:
            data = pickle.load(f)

        if not isinstance(data, dict):
            raise TypeError("Expected replay .pkl to contain a dict.")

        if "buffer" not in data:
            raise KeyError("Replay dict does not contain a 'buffer' key.")

        self.samples = data["buffer"]

        if len(self.samples) == 0:
            raise ValueError("Replay buffer is empty.")

        obs, policy, value = self.samples[0]

        print("Loaded replay buffer")
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

        return observation, policy, value


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    ).sum(dim=1).mean()

    predicted_value = predicted_value.squeeze(-1)

    value_loss = F.mse_loss(
        predicted_value,
        target_value,
    )

    total_loss = policy_loss + value_loss

    return total_loss, policy_loss, value_loss


@torch.no_grad()
def batch_metrics(
    policy_logits,
    predicted_value,
    target_policy,
    target_value,
):
    predicted_action = policy_logits.argmax(dim=1)
    target_action = target_policy.argmax(dim=1)

    policy_top1 = (
        predicted_action == target_action
    ).float().mean().item()

    predicted_value = predicted_value.squeeze(-1)

    predicted_sign = torch.where(
        predicted_value >= 0,
        torch.ones_like(predicted_value),
        -torch.ones_like(predicted_value),
    )

    target_sign = torch.where(
        target_value >= 0,
        torch.ones_like(target_value),
        -torch.ones_like(target_value),
    )

    value_sign = (
        predicted_sign == target_sign
    ).float().mean().item()

    return policy_top1, value_sign


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

    for observations, target_policies, target_values in loader:
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

        optimizer.zero_grad(set_to_none=True)

        policy_logits, predicted_values = model(observations)

        loss, policy_loss, value_loss = compute_losses(
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

        batch_size = observations.shape[0]

        policy_top1, value_sign = batch_metrics(
            policy_logits,
            predicted_values,
            target_policies,
            target_values,
        )

        total_examples += batch_size
        total_loss_sum += loss.item() * batch_size
        policy_loss_sum += policy_loss.item() * batch_size
        value_loss_sum += value_loss.item() * batch_size
        policy_top1_sum += policy_top1 * batch_size
        value_sign_sum += value_sign * batch_size

    return {
        "loss": total_loss_sum / total_examples,
        "policy_loss": policy_loss_sum / total_examples,
        "value_loss": value_loss_sum / total_examples,
        "policy_top1": policy_top1_sum / total_examples,
        "value_sign": value_sign_sum / total_examples,
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

    for observations, target_policies, target_values in loader:
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

        policy_logits, predicted_values = model(observations)

        loss, policy_loss, value_loss = compute_losses(
            policy_logits,
            predicted_values,
            target_policies,
            target_values,
        )

        batch_size = observations.shape[0]

        policy_top1, value_sign = batch_metrics(
            policy_logits,
            predicted_values,
            target_policies,
            target_values,
        )

        total_examples += batch_size
        total_loss_sum += loss.item() * batch_size
        policy_loss_sum += policy_loss.item() * batch_size
        value_loss_sum += value_loss.item() * batch_size
        policy_top1_sum += policy_top1 * batch_size
        value_sign_sum += value_sign * batch_size

    return {
        "loss": total_loss_sum / total_examples,
        "policy_loss": policy_loss_sum / total_examples,
        "value_loss": value_loss_sum / total_examples,
        "policy_top1": policy_top1_sum / total_examples,
        "value_sign": value_sign_sum / total_examples,
    }


def save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    train_metrics,
    val_metrics,
    args,
):
    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "args": vars(args),
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Pretrain a fresh SplendorNetwork from "
            "heuristic replay-buffer data."
        )
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--replay",
        type=str,
        default="heuristic_replay_buffer.pkl",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints/heuristic_pretrain",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
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
        "--val-fraction",
        type=float,
        default=0.10,
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

    set_seed(args.seed)

    if not (0.0 < args.val_fraction < 1.0):
        raise ValueError(
            "--val-fraction must be between 0 and 1."
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print("=" * 70)
    print("HEURISTIC PRETRAINING")
    print("=" * 70)
    print(f"Device:         {device}")
    print(f"Replay:         {args.replay}")
    print(f"Epochs:         {args.epochs}")
    print(f"Batch size:     {args.batch_size}")
    print(f"Learning rate:  {args.learning_rate}")
    print(f"Weight decay:   {args.weight_decay}")
    print(f"Validation:     {args.val_fraction:.0%}")
    print()

    dataset = HeuristicReplayDataset(
        args.replay
    )

    total_size = len(dataset)

    val_size = max(
        1,
        int(total_size * args.val_fraction),
    )

    train_size = total_size - val_size

    split_generator = torch.Generator().manual_seed(
        args.seed
    )

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=split_generator,
    )

    print()
    print(f"Train samples: {train_size:,}")
    print(f"Val samples:   {val_size:,}")
    print()

    pin_memory = device.type == "cuda"

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

    model = SplendorNetwork().to(
        device=device,
        dtype=torch.float32,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    start_epoch = 1

    if args.resume is not None:
        checkpoint = torch.load(
            args.resume,
            map_location=device,
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        start_epoch = checkpoint["epoch"] + 1

        print(
            f"Resuming from epoch "
            f"{checkpoint['epoch']}"
        )




    parameter_count = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"Fresh model parameters: "
        f"{parameter_count:,}"
    )

    baseline_metrics = evaluate(
        model,
        val_loader,
        device,
    )

    print()
    print("Before training")
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

    output_dir = Path(args.output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_val_loss = float("inf")

    for epoch in range(
        start_epoch,
        args.epochs + 1,
    ):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            grad_clip=args.grad_clip,
        )

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            device=device,
        )

        print("=" * 70)
        print(
            f"EPOCH {epoch}/{args.epochs}"
        )
        print("=" * 70)

        print(
            "TRAIN"
            f" | loss {train_metrics['loss']:.4f}"
            f" | policy {train_metrics['policy_loss']:.4f}"
            f" | value {train_metrics['value_loss']:.4f}"
            f" | top1 {train_metrics['policy_top1']:.2%}"
            f" | value sign {train_metrics['value_sign']:.2%}"
        )

        print(
            "VAL  "
            f" | loss {val_metrics['loss']:.4f}"
            f" | policy {val_metrics['policy_loss']:.4f}"
            f" | value {val_metrics['value_loss']:.4f}"
            f" | top1 {val_metrics['policy_top1']:.2%}"
            f" | value sign {val_metrics['value_sign']:.2%}"
        )

        epoch_path = (
            output_dir
            / f"heuristic_pretrain_epoch_{epoch:02d}.pt"
        )

        save_checkpoint(
            path=epoch_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            args=args,
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]

            best_path = (
                output_dir
                / "heuristic_pretrain_best.pt"
            )

            save_checkpoint(
                path=best_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                args=args,
            )

            print(
                f"Saved new best checkpoint: "
                f"{best_path}"
            )

        print()

    final_weights_path = (
        output_dir
        / "heuristic_pretrained_weights.pt"
    )

    torch.save(
        model.state_dict(),
        final_weights_path,
    )

    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(
        f"Final weights: {final_weights_path}"
    )
    print(
        f"Best checkpoint: "
        f"{output_dir / 'heuristic_pretrain_best.pt'}"
    )


if __name__ == "__main__":
    main()
