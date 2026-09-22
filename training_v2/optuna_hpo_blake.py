"""Optuna search for Blake's initial supervised pretraining.

This script deliberately reuses the replay loader and epoch functions from
``train_heuristic_pretrain_split.py`` so HPO and normal training cannot drift
to different split or loss implementations.

Run from the repository root, for example:

    python optuna_hpo_blake.py \
        --replay data/heuristic_replay_buffer.pkl \
        --split-mode legacy-contiguous \
        --trials 30 \
        --epochs 12

``--trials`` is the target number of finished trials in the study, not the
number of additional trials to launch. Every trial starts from fresh weights
with the same seed. The Optuna SQLite database is persistent, so the same
command can resume an interrupted study without silently adding another full
batch of trials.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import pickle
from pathlib import Path
from time import perf_counter
from typing import Any

import optuna
import torch
from torch.utils.data import DataLoader

from splendor_v1.network.model_2_attention import SplendorNetwork
from splendor_v1.training.train_heuristic_pretrain_split import (
    ReplaySampleDataset,
    evaluate,
    load_and_combine_replays,
    set_seed,
    train_one_epoch,
)


SEARCH_SPACE_VERSION = 3

# Evaluate the currently used training defaults before Optuna samples new
# configurations. This gives every study a direct apples-to-apples baseline.
BASELINE_PARAMS = {
    "learning_rate": 3e-4,
    "weight_decay": 1e-4,
    "batch_size": 32,
    "dropout": 0.0,
    "grad_clip": 1.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optuna HPO for Blake supervised pretraining."
    )
    parser.add_argument(
        "--replay",
        nargs="+",
        required=True,
        help=(
            "Replay .pkl file(s). Metadata mode expects saved train/val "
            "split metadata; legacy-contiguous mode recreates H12's split."
        ),
    )
    parser.add_argument(
        "--split-mode",
        choices=("metadata", "legacy-contiguous"),
        default="metadata",
        help=(
            "Use replay-provided split metadata, or recreate the legacy "
            "contiguous split used for H12."
        ),
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.10,
        help=(
            "Validation fraction for --split-mode legacy-contiguous. "
            "Ignored in metadata mode."
        ),
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=30,
        help=(
            "Target number of finished trials in the persistent study. "
            "Rerunning the command resumes only the missing trials."
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=12,
        help="Maximum epochs per HPO trial (use fewer than final training).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--study-name",
        default="blake_supervised_v1",
        help="Stable name used to resume the study.",
    )
    parser.add_argument(
        "--storage",
        default="sqlite:///blake_supervised_hpo.db",
        help="Optuna storage URL.",
    )
    parser.add_argument(
        "--results-dir",
        default="hpo/blake_supervised_v1",
    )
    parser.add_argument(
        "--timeout-hours",
        type=float,
        default=None,
        help=(
            "Optional wall-clock limit for this invocation. The SQLite "
            "study can be resumed later with the same command."
        ),
    )
    parser.add_argument(
        "--objective-value-weight",
        type=float,
        default=1.0,
        help=(
            "Fixed weight used only to rank trials: "
            "val_policy_loss + weight * val_value_loss. "
            "Keep this fixed for an entire study."
        ),
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    return parser.parse_args()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON without leaving a half-written result after interruption."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def replay_identities(
    replay_paths: list[str],
    replay_summaries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Create a cheap guard against resuming a study with different data."""
    identities = []
    for replay_path, summary in zip(replay_paths, replay_summaries):
        path = Path(replay_path).resolve()
        stat = path.stat()
        identities.append(
            {
                "path": str(path),
                "size_bytes": stat.st_size,
                "modified_time_ns": stat.st_mtime_ns,
                "train_samples": summary["train"],
                "validation_samples": summary["val"],
                "split_strategy": summary["strategy"],
            }
        )
    return identities


def load_legacy_contiguous_replays(
    replay_paths: list[str],
    val_fraction: float,
) -> tuple[list[Any], list[Any], list[dict[str, Any]]]:
    """Recreate the contiguous 90/10 position split used by H12.

    The original H12 trainer kept positions in their stored order, used the
    first ``1 - val_fraction`` portion for training, and used the final
    ``val_fraction`` portion for validation. No random permutation was used.
    """
    all_train_samples: list[Any] = []
    all_val_samples: list[Any] = []
    summaries: list[dict[str, Any]] = []

    for replay_path in replay_paths:
        path = Path(replay_path)
        if not path.exists():
            raise FileNotFoundError(f"Replay buffer not found: {path}")

        with path.open("rb") as replay_file:
            data = pickle.load(replay_file)

        if not isinstance(data, dict):
            raise TypeError(f"{path}: expected replay .pkl to contain a dict.")
        if "buffer" not in data:
            raise KeyError(f"{path}: replay dict has no 'buffer' key.")

        buffer = data["buffer"]
        if not isinstance(buffer, (list, tuple)):
            buffer = list(buffer)
        if not buffer:
            raise ValueError(f"{path}: replay buffer is empty.")

        validation_size = max(1, int(len(buffer) * val_fraction))
        split_index = len(buffer) - validation_size
        if split_index <= 0:
            raise ValueError(
                f"{path}: validation split leaves no training samples."
            )

        all_train_samples.extend(buffer[:split_index])
        all_val_samples.extend(buffer[split_index:])
        summaries.append(
            {
                "path": str(path),
                "total": len(buffer),
                "train": split_index,
                "val": validation_size,
                "strategy": "legacy_contiguous_positions",
                "is_legacy": True,
                "split_index": split_index,
                "val_fraction": val_fraction,
                "train_games": None,
                "val_games": None,
            }
        )

    return all_train_samples, all_val_samples, summaries


def experiment_config(
    args: argparse.Namespace,
    replay_summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "search_space_version": SEARCH_SPACE_VERSION,
        "model": "splendor_v1.network.model_2_attention.SplendorNetwork",
        "trainer": "train_heuristic_pretrain_split",
        "split_mode": args.split_mode,
        "legacy_val_fraction": (
            args.val_fraction
            if args.split_mode == "legacy-contiguous"
            else None
        ),
        "epochs": args.epochs,
        "seed": args.seed,
        "objective": (
            "best val_policy_loss + objective_value_weight * "
            "val_value_loss"
        ),
        "objective_value_weight": args.objective_value_weight,
        "search_space": {
            "learning_rate": [3e-5, 2e-3, "log"],
            "weight_decay": [1e-6, 1e-2, "log"],
            "batch_size": [32, 64, 128, 256],
            "dropout": [0.0, 0.20],
            "grad_clip": [0.25, 2.0, "log"],
        },
        "replays": replay_identities(args.replay, replay_summaries),
    }


def validate_or_record_config(
    study: optuna.Study,
    config: dict[str, Any],
) -> None:
    """Prevent incompatible runs from being mixed in one SQLite study."""
    stored = study.user_attrs.get("experiment_config")
    if stored is None:
        if study.trials:
            raise RuntimeError(
                "This study already contains trials but has no experiment "
                "configuration guard. Use a new --study-name and --storage "
                "rather than mixing experiments."
            )
        study.set_user_attr("experiment_config", config)
        return

    if stored != config:
        raise RuntimeError(
            "The requested dataset, epoch budget, seed, objective, or search "
            "space differs from the stored study configuration. Use a new "
            "--study-name and --storage for this experiment."
        )


def serialize_trial(trial: optuna.trial.FrozenTrial) -> dict[str, Any]:
    return {
        "number": trial.number,
        "state": trial.state.name,
        "value": trial.value,
        "params": trial.params,
        "user_attrs": trial.user_attrs,
        "duration_seconds": (
            trial.duration.total_seconds()
            if trial.duration is not None
            else None
        ),
    }


def save_study_snapshot(
    study: optuna.Study,
    results_dir: Path,
    config: dict[str, Any],
    replay_summaries: list[dict[str, Any]],
) -> None:
    """Persist human-readable results after every finished trial."""
    complete = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    ranked = sorted(
        complete,
        key=lambda trial: float(trial.value),
    )

    state_counts: dict[str, int] = {}
    for trial in study.trials:
        state_counts[trial.state.name] = (
            state_counts.get(trial.state.name, 0) + 1
        )

    snapshot: dict[str, Any] = {
        "study_name": study.study_name,
        "direction": study.direction.name,
        "state_counts": state_counts,
        "experiment_config": config,
        "replay_summaries": replay_summaries,
        "top_trials": [serialize_trial(trial) for trial in ranked[:10]],
        "all_trials": [serialize_trial(trial) for trial in study.trials],
    }

    if ranked:
        best = ranked[0]
        snapshot["best_trial"] = serialize_trial(best)
        snapshot["best_epoch"] = best.user_attrs.get("best_epoch")
        snapshot["best_validation_metrics"] = {
            key.removeprefix("best_val_"): value
            for key, value in best.user_attrs.items()
            if key.startswith("best_val_")
        }

    write_json_atomic(results_dir / "study_snapshot.json", snapshot)

    if ranked:
        write_json_atomic(
            results_dir / "best_trial.json",
            {
                "study_name": study.study_name,
                "best_trial": ranked[0].number,
                "best_score": ranked[0].value,
                "best_params": ranked[0].params,
                "best_epoch": ranked[0].user_attrs.get("best_epoch"),
                "best_validation_metrics": snapshot[
                    "best_validation_metrics"
                ],
                "experiment_config": config,
                "replays": replay_summaries,
                "note": (
                    "Confirm the top 3 configurations with full-length "
                    "training over at least 3 independent seeds before "
                    "selecting the final Model 2 settings."
                ),
            },
        )


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is unavailable.")
    return torch.device(name)


def make_loader(
    dataset: ReplaySampleDataset,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=generator,
        persistent_workers=num_workers > 0,
    )


def main() -> None:
    args = parse_args()
    if args.trials < 1:
        raise ValueError("--trials must be at least 1.")
    if args.epochs < 3:
        raise ValueError("--epochs must be at least 3 for meaningful pruning.")
    if args.objective_value_weight < 0:
        raise ValueError("--objective-value-weight cannot be negative.")
    if args.timeout_hours is not None and args.timeout_hours <= 0:
        raise ValueError("--timeout-hours must be positive when supplied.")
    if args.split_mode == "legacy-contiguous" and not (
        0.0 < args.val_fraction < 1.0
    ):
        raise ValueError(
            "--val-fraction must be between 0 and 1 for "
            "legacy-contiguous mode."
        )

    device = resolve_device(args.device)
    pin_memory = device.type == "cuda"
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load once. All trials see precisely the same samples and split.
    if args.split_mode == "legacy-contiguous":
        (
            train_samples,
            val_samples,
            replay_summaries,
        ) = load_legacy_contiguous_replays(
            args.replay,
            args.val_fraction,
        )
    else:
        (
            train_samples,
            val_samples,
            replay_summaries,
        ) = load_and_combine_replays(args.replay)

    train_dataset = ReplaySampleDataset(train_samples, name="HPO TRAIN")
    val_dataset = ReplaySampleDataset(val_samples, name="HPO VALIDATION")

    print(f"Device: {device}")
    print(f"Split mode: {args.split_mode}")
    for summary in replay_summaries:
        split_description = summary["strategy"]
        if "split_index" in summary:
            split_description += f" at index {summary['split_index']:,}"
        print(
            f"Replay split: {summary['path']} | {split_description} | "
            f"train={summary['train']:,} | val={summary['val']:,}"
        )
    print(f"Train samples: {len(train_dataset):,}")
    print(f"Validation samples: {len(val_dataset):,}")

    def objective(trial: optuna.Trial) -> float:
        # First-stage HPO: optimizer/training settings only. Blake's width,
        # depth, head count, and loss definition remain fixed.
        learning_rate = trial.suggest_float(
            "learning_rate", 3e-5, 2e-3, log=True
        )
        weight_decay = trial.suggest_float(
            "weight_decay", 1e-6, 1e-2, log=True
        )
        batch_size = trial.suggest_categorical(
            "batch_size", [32, 64, 128, 256]
        )
        dropout = trial.suggest_float("dropout", 0.0, 0.20)
        grad_clip = trial.suggest_float(
            "grad_clip", 0.25, 2.0, log=True
        )

        # Identical seed per trial makes comparisons less noisy. The winning
        # region should later be confirmed across multiple seeds.
        set_seed(args.seed)

        train_loader = make_loader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            seed=args.seed,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        )
        val_loader = make_loader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            seed=args.seed,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        )

        model = None
        optimizer = None
        best_score = float("inf")
        best_epoch = 0
        best_metrics: dict[str, float] | None = None
        started_at = perf_counter()

        try:
            model = SplendorNetwork(dropout=dropout).to(
                device=device,
                dtype=torch.float32,
            )
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
            )

            for epoch in range(1, args.epochs + 1):
                train_metrics = train_one_epoch(
                    model=model,
                    loader=train_loader,
                    optimizer=optimizer,
                    device=device,
                    grad_clip=grad_clip,
                )
                val_metrics = evaluate(model, val_loader, device)

                # This ranking formula is fixed across trials. Do not use a
                # tunable loss weight in the objective itself, because those
                # trial scores would no longer be comparable.
                score = (
                    val_metrics["policy_loss"]
                    + args.objective_value_weight
                    * val_metrics["value_loss"]
                )

                if not math.isfinite(score):
                    trial.set_user_attr("failure", "non-finite objective")
                    raise optuna.TrialPruned(
                        f"non-finite objective at epoch {epoch}: {score}"
                    )

                if score < best_score:
                    best_score = score
                    best_epoch = epoch
                    best_metrics = dict(val_metrics)
                    trial.set_user_attr("best_epoch", best_epoch)
                    for key, value in best_metrics.items():
                        trial.set_user_attr(
                            f"best_val_{key}",
                            float(value),
                        )

                trial.report(score, step=epoch)
                print(
                    f"trial={trial.number:03d} epoch={epoch:02d} "
                    f"score={score:.5f} "
                    f"policy={val_metrics['policy_loss']:.5f} "
                    f"value={val_metrics['value_loss']:.5f} "
                    f"top1={val_metrics['policy_top1']:.2%}"
                )

                if trial.should_prune():
                    raise optuna.TrialPruned(
                        f"pruned at epoch {epoch}; score={score:.5f}"
                    )

        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            trial.set_user_attr("failure", "out of memory")
            raise optuna.TrialPruned("out of memory") from exc
        finally:
            trial.set_user_attr(
                "elapsed_seconds",
                perf_counter() - started_at,
            )
            if optimizer is not None:
                del optimizer
            if model is not None:
                del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        return best_score

    sampler = optuna.samplers.TPESampler(
        seed=args.seed,
        n_startup_trials=min(8, args.trials),
        multivariate=True,
    )
    # A single median population is a better fit for this modest study than
    # Hyperband's multiple brackets. With TPE, every Hyperband bracket needs
    # its own startup observations, which wastes most of a 30-trial budget.
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=min(8, args.trials),
        n_warmup_steps=3,
        interval_steps=1,
    )
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )

    config = experiment_config(args, replay_summaries)
    validate_or_record_config(study, config)

    if not study.trials:
        study.enqueue_trial(
            BASELINE_PARAMS,
            user_attrs={"configuration": "current_training_defaults"},
        )

    finished_before = sum(
        trial.state.is_finished()
        for trial in study.trials
    )
    remaining_trials = max(0, args.trials - finished_before)
    timeout_seconds = (
        args.timeout_hours * 60 * 60
        if args.timeout_hours is not None
        else None
    )

    print(f"Study: {study.study_name}")
    print(f"Finished trials: {finished_before}/{args.trials}")
    print(f"Trials to run now: {remaining_trials}")

    def save_after_trial(
        current_study: optuna.Study,
        _finished_trial: optuna.trial.FrozenTrial,
    ) -> None:
        save_study_snapshot(
            current_study,
            results_dir,
            config,
            replay_summaries,
        )

    try:
        if remaining_trials > 0:
            study.optimize(
                objective,
                n_trials=remaining_trials,
                timeout=timeout_seconds,
                n_jobs=1,
                gc_after_trial=True,
                callbacks=[save_after_trial],
            )
    finally:
        save_study_snapshot(
            study,
            results_dir,
            config,
            replay_summaries,
        )

    complete_trials = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    if not complete_trials:
        raise RuntimeError(
            "The study has no completed trials. Inspect the trial failures "
            "in study_snapshot.json before resuming."
        )

    best = study.best_trial
    print("\nBest trial")
    print(f"  number: {best.number}")
    print(f"  score:  {best.value:.6f}")
    print(f"  epoch:  {best.user_attrs.get('best_epoch')}")
    print(f"  params: {json.dumps(best.params, sort_keys=True)}")
    print(f"\nSaved: {results_dir / 'best_trial.json'}")
    print(f"Saved: {results_dir / 'study_snapshot.json'}")


if __name__ == "__main__":
    main()
