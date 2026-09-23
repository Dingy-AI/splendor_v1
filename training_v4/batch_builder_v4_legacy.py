from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from splendor_v1.env.core.action_constants import ACTION_SPACE_SIZE
from splendor_v1.network.model_4_legal_scorer import (
    WDL_DRAW,
    WDL_LOSS,
    WDL_WIN,
)


OBSERVATION_SIZE = 258


# ============================================================
# Legacy replay dataset
# ============================================================


class Model4LegacyReplayDataset(
    Dataset
):
    """
    Dataset wrapper for the OLD H12 replay format.

    Expected file format:

        {
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

    Important limitation of old data:
        The dense [1139] policy does NOT preserve a separate legal
        action list, so a target probability of 0 cannot distinguish:

            illegal action
            vs.
            legal action with zero MCTS visits

    Therefore Model 4 legacy training deliberately presents ALL
    canonical action IDs 0..1138 as candidates. This preserves the
    old dense-policy training problem as faithfully as possible.
    """

    def __init__(
        self,
        replay_path: str | Path,
        validate_all: bool = False,
    ):
        super().__init__()

        self.replay_path = Path(
            replay_path
        )

        if not self.replay_path.exists():

            raise FileNotFoundError(
                "Legacy replay file not found: "
                f"{self.replay_path}"
            )

        with self.replay_path.open(
            "rb"
        ) as file:

            replay = pickle.load(
                file
            )

        if not isinstance(
            replay,
            dict,
        ):

            raise TypeError(
                "Expected legacy replay .pkl to contain "
                "a dictionary."
            )

        if "buffer" not in replay:

            raise KeyError(
                "Legacy replay dictionary does not "
                "contain 'buffer'."
            )

        samples = replay[
            "buffer"
        ]

        if not isinstance(
            samples,
            (
                list,
                tuple,
            ),
        ):

            raise TypeError(
                "Legacy replay['buffer'] must be "
                "a list or tuple."
            )

        if len(
            samples
        ) == 0:

            raise ValueError(
                "Legacy replay buffer is empty."
            )

        self.replay = replay
        self.samples = samples

        # Always validate the first sample immediately so obvious
        # format mismatches fail at construction time.
        self._validate_sample(
            self.samples[0],
            sample_index=0,
        )

        if validate_all:

            for sample_index, sample in enumerate(
                self.samples
            ):

                self._validate_sample(
                    sample,
                    sample_index=sample_index,
                )

    @staticmethod
    def scalar_to_wdl(
        value,
    ) -> int:
        """
        Legacy scalar value -> Model 4 WDL class.

            -1 -> LOSS
             0 -> DRAW
            +1 -> WIN
        """

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
            "Legacy scalar value must be -1, 0, or +1. "
            f"Got {value}."
        )

    @staticmethod
    def _validate_sample(
        sample,
        sample_index: int | None = None,
    ) -> None:

        prefix = (
            ""
            if sample_index is None
            else f"Legacy sample {sample_index}: "
        )

        if (
            not isinstance(
                sample,
                (
                    tuple,
                    list,
                ),
            )
            or len(
                sample
            ) != 3
        ):

            raise TypeError(
                prefix
                + "expected a 3-tuple "
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
            OBSERVATION_SIZE,
        ):

            raise ValueError(
                prefix
                + "expected observation shape "
                f"({OBSERVATION_SIZE},), got "
                f"{observation.shape}."
            )

        if policy.shape != (
            ACTION_SPACE_SIZE,
        ):

            raise ValueError(
                prefix
                + "expected policy shape "
                f"({ACTION_SPACE_SIZE},), got "
                f"{policy.shape}."
            )

        if not np.all(
            np.isfinite(
                observation
            )
        ):

            raise ValueError(
                prefix
                + "observation contains NaN or infinity."
            )

        if not np.all(
            np.isfinite(
                policy
            )
        ):

            raise ValueError(
                prefix
                + "policy contains NaN or infinity."
            )

        if np.any(
            policy < 0
        ):

            raise ValueError(
                prefix
                + "policy contains negative probability."
            )

        policy_sum = float(
            policy.sum()
        )

        if (
            not np.isfinite(
                policy_sum
            )
            or policy_sum <= 0.0
        ):

            raise ValueError(
                prefix
                + "policy must have a positive finite sum."
            )

        # Also validates the scalar target.
        Model4LegacyReplayDataset.scalar_to_wdl(
            value
        )

    def __len__(
        self,
    ) -> int:

        return len(
            self.samples
        )

    def __getitem__(
        self,
        index: int,
    ):

        sample = self.samples[
            index
        ]

        (
            observation,
            policy,
            value,
        ) = sample

        return {
            "observation":
                observation,

            "policy":
                policy,

            "scalar_value":
                float(
                    value
                ),

            "target_wdl":
                self.scalar_to_wdl(
                    value
                ),

            "sample_index":
                int(
                    index
                ),
        }


# ============================================================
# Contiguous train / validation split
# ============================================================


def build_legacy_contiguous_split(
    dataset: Dataset,
    val_fraction: float = 0.10,
):
    """
    Preserve the same split strategy used for the old H12
    pretraining work.

    The legacy replay contains no game IDs, but positions were
    appended sequentially. Use a contiguous split instead of a
    random position split:

        [0, split_index)   -> train
        [split_index, end) -> validation

    This reduces trajectory leakage compared with randomly mixing
    neighboring positions across train and validation.
    """

    if not (
        0.0
        < val_fraction
        < 1.0
    ):

        raise ValueError(
            "val_fraction must be between 0 and 1."
        )

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

    split_index = train_size

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
# Legacy Model 4 collation
# ============================================================


def collate_model4_legacy_batch(
    items,
):
    """
    Convert old dense-policy H12 samples into Model 4 inputs.

    Because the old replay does not contain legal-action IDs,
    every canonical action is presented as a candidate:

        legal_action_ids:
            [0, 1, 2, ..., 1138]

        legal_action_mask:
            all True

    Output:

        observations
            [B, 258]

        legal_action_ids
            [B, 1139]

        legal_action_mask
            [B, 1139]

        target_policy
            [B, 1139]

        target_wdl
            [B]

        scalar_value
            [B]

        sample_indices
            [B]

    There is no padding in a legacy batch because every row has the
    same 1,139 candidate action IDs.
    """

    if not items:

        raise ValueError(
            "Cannot collate an empty legacy Model 4 batch."
        )

    batch_size = len(
        items
    )

    observations_np = np.stack(
        [
            np.asarray(
                item[
                    "observation"
                ],
                dtype=np.float32,
            )
            for item
            in items
        ],
        axis=0,
    )

    policies_np = np.stack(
        [
            np.asarray(
                item[
                    "policy"
                ],
                dtype=np.float32,
            )
            for item
            in items
        ],
        axis=0,
    )

    if observations_np.shape != (
        batch_size,
        OBSERVATION_SIZE,
    ):

        raise ValueError(
            "Unexpected legacy observation batch shape: "
            f"{observations_np.shape}"
        )

    if policies_np.shape != (
        batch_size,
        ACTION_SPACE_SIZE,
    ):

        raise ValueError(
            "Unexpected legacy policy batch shape: "
            f"{policies_np.shape}"
        )

    if not np.all(
        np.isfinite(
            policies_np
        )
    ):

        raise ValueError(
            "Legacy policy batch contains NaN or infinity."
        )

    if np.any(
        policies_np < 0
    ):

        raise ValueError(
            "Legacy policy batch contains negative values."
        )

    observations = torch.from_numpy(
        observations_np
    )

    target_policy = torch.from_numpy(
        policies_np
    )

    # Renormalize each row. Old H12 targets are already effectively
    # normalized, but this removes tiny float32 accumulation drift.
    policy_sums = target_policy.sum(
        dim=1,
        keepdim=True,
    )

    if torch.any(
        policy_sums <= 0
    ):

        raise ValueError(
            "Every legacy policy row must have "
            "a positive sum."
        )

    target_policy = (
        target_policy
        / policy_sums
    )

    # Every old sample scores the entire canonical vocabulary.
    canonical_action_ids = torch.arange(
        ACTION_SPACE_SIZE,
        dtype=torch.long,
    )

    legal_action_ids = (
        canonical_action_ids
        .unsqueeze(
            0
        )
        .repeat(
            batch_size,
            1,
        )
    )

    legal_action_mask = torch.ones(
        (
            batch_size,
            ACTION_SPACE_SIZE,
        ),
        dtype=torch.bool,
    )

    target_wdl = torch.tensor(
        [
            int(
                item[
                    "target_wdl"
                ]
            )
            for item
            in items
        ],
        dtype=torch.long,
    )

    scalar_value = torch.tensor(
        [
            float(
                item[
                    "scalar_value"
                ]
            )
            for item
            in items
        ],
        dtype=torch.float32,
    )

    sample_indices = torch.tensor(
        [
            int(
                item[
                    "sample_index"
                ]
            )
            for item
            in items
        ],
        dtype=torch.long,
    )

    # --------------------------------------------------------
    # Final invariants
    # --------------------------------------------------------

    if not torch.allclose(
        target_policy.sum(
            dim=1
        ),
        torch.ones(
            batch_size,
            dtype=torch.float32,
        ),
        atol=1e-6,
    ):

        raise RuntimeError(
            "Normalized legacy policy rows do not sum to 1."
        )

    if not torch.all(
        legal_action_mask
    ):

        raise RuntimeError(
            "Legacy action mask must be entirely True."
        )

    if not torch.equal(
        legal_action_ids[
            0
        ],
        canonical_action_ids,
    ):

        raise RuntimeError(
            "Legacy canonical action-ID row is malformed."
        )

    return {
        "observations":
            observations,

        "legal_action_ids":
            legal_action_ids,

        "legal_action_mask":
            legal_action_mask,

        "target_policy":
            target_policy,

        "target_wdl":
            target_wdl,

        "scalar_value":
            scalar_value,

        "sample_indices":
            sample_indices,
    }


# ============================================================
# DataLoader helpers
# ============================================================


def make_model4_legacy_dataloaders(
    replay_path: str | Path,
    batch_size: int,
    val_fraction: float = 0.10,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last_train: bool = False,
    validate_all: bool = False,
):
    """
    Create separate train and validation DataLoaders for old H12 data.

    Training:
        contiguous first portion
        shuffle=True within that portion

    Validation:
        contiguous final portion
        shuffle=False

    Note:
        Legacy batches score all 1,139 actions, so their practical
        batch size may need to be smaller than the rich-data batch
        size used by Model 4.
    """

    if batch_size <= 0:

        raise ValueError(
            "batch_size must be positive."
        )

    dataset = Model4LegacyReplayDataset(
        replay_path=replay_path,
        validate_all=validate_all,
    )

    (
        train_dataset,
        val_dataset,
        split_index,
    ) = build_legacy_contiguous_split(
        dataset=dataset,
        val_fraction=val_fraction,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last_train,
        collate_fn=collate_model4_legacy_batch,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        collate_fn=collate_model4_legacy_batch,
    )

    return (
        train_loader,
        val_loader,
        {
            "dataset_size":
                len(
                    dataset
                ),

            "train_size":
                len(
                    train_dataset
                ),

            "val_size":
                len(
                    val_dataset
                ),

            "split_index":
                split_index,

            "val_fraction":
                float(
                    val_fraction
                ),
        },
    )


# ============================================================
# Diagnostics
# ============================================================


def summarize_model4_legacy_batch(
    batch,
) -> None:

    print(
        "observations:",
        tuple(
            batch[
                "observations"
            ].shape
        ),
    )

    print(
        "legal_action_ids:",
        tuple(
            batch[
                "legal_action_ids"
            ].shape
        ),
    )

    print(
        "legal_action_mask:",
        tuple(
            batch[
                "legal_action_mask"
            ].shape
        ),
    )

    print(
        "target_policy:",
        tuple(
            batch[
                "target_policy"
            ].shape
        ),
    )

    print(
        "target_wdl:",
        tuple(
            batch[
                "target_wdl"
            ].shape
        ),
    )

    print(
        "policy row sums:",
        batch[
            "target_policy"
        ].sum(
            dim=1
        )[:8],
    )

    unique_wdl, counts = torch.unique(
        batch[
            "target_wdl"
        ],
        return_counts=True,
    )

    print(
        "WDL counts:",
        {
            int(
                label
            ):
                int(
                    count
                )
            for (
                label,
                count,
            )
            in zip(
                unique_wdl,
                counts,
            )
        },
    )
