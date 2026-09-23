from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from splendor_v1.env.core.action_constants import ACTION_SPACE_SIZE
from splendor_v1.network.model_4_legal_scorer import (
    WDL_DRAW,
    WDL_LOSS,
    WDL_WIN,
)


# ============================================================
# Replay loading
# ============================================================


def load_replay(
    replay_path: str | Path,
) -> dict:
    """
    Load the rich Model 2+ replay-buffer dictionary used by Model 4.

    Expected top-level fields include:
        buffer
        games
    """

    replay_path = Path(
        replay_path
    )

    with replay_path.open(
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
            "Expected replay file to contain a dictionary."
        )

    if "buffer" not in replay:
        raise KeyError(
            "Replay is missing top-level 'buffer'."
        )

    if "games" not in replay:
        raise KeyError(
            "Replay is missing top-level 'games'."
        )

    return replay


# ============================================================
# Game metadata
# ============================================================


def build_game_lookup(
    replay: dict,
) -> dict[int, dict]:
    """
    Normalize replay['games'] into:
        game_id -> game_metadata

    Current rich replay files store games as a dictionary, but this
    also accepts a list/tuple for forward/backward compatibility.
    """

    games = replay[
        "games"
    ]

    if isinstance(
        games,
        dict,
    ):

        game_values = list(
            games.values()
        )

    elif isinstance(
        games,
        (
            list,
            tuple,
        ),
    ):

        game_values = list(
            games
        )

    else:

        raise TypeError(
            "replay['games'] must be a dict, list, or tuple."
        )

    lookup = {}

    for game in game_values:

        if not isinstance(
            game,
            dict,
        ):

            raise TypeError(
                "Every game metadata entry must be a dictionary."
            )

        if "game_id" not in game:

            raise KeyError(
                "Game metadata is missing 'game_id'."
            )

        game_id = int(
            game[
                "game_id"
            ]
        )

        if game_id in lookup:

            raise ValueError(
                f"Duplicate game_id in replay metadata: {game_id}"
            )

        lookup[
            game_id
        ] = game

    return lookup


def wdl_target_from_winners(
    current_player: int,
    winner_ids,
) -> int:
    """
    Convert final game outcome to Model 4's WDL target from the
    CURRENT PLAYER perspective of a replay sample.

    Model 4 semantics:
        LOSS = 0
        DRAW = 1
        WIN  = 2

    A single winner:
        current player is winner -> WIN
        otherwise                -> LOSS

    Zero or multiple winners:
        DRAW

    The multiple-winner rule matches the project's zero-valued
    shared/tied outcome semantics.
    """

    if winner_ids is None:

        winner_ids = []

    winner_ids = [
        int(
            winner_id
        )
        for winner_id
        in winner_ids
    ]

    current_player = int(
        current_player
    )

    if len(
        winner_ids
    ) != 1:

        return WDL_DRAW

    if (
        current_player
        == winner_ids[0]
    ):

        return WDL_WIN

    return WDL_LOSS


# ============================================================
# Dataset
# ============================================================


class Model4ReplayDataset(
    Dataset
):
    """
    Lightweight view over the rich replay buffer.

    The dataset does NOT copy observations or policy arrays. It keeps
    indices into replay['buffer'] and attaches the final-game WDL
    target when an item is requested.

    split:
        None
            use every retained sample

        "train"
            only games whose metadata split == "train"

        "val"
            only games whose metadata split == "val"
    """

    def __init__(
        self,
        replay: dict,
        split: str | None = None,
    ):
        super().__init__()

        self.replay = replay

        self.buffer = replay[
            "buffer"
        ]

        self.game_lookup = (
            build_game_lookup(
                replay
            )
        )

        self.split = split

        if split not in (
            None,
            "train",
            "val",
        ):

            raise ValueError(
                "split must be one of: None, 'train', 'val'."
            )

        self.sample_indices = []

        for sample_index, sample in enumerate(
            self.buffer
        ):

            if not isinstance(
                sample,
                dict,
            ):

                raise TypeError(
                    "Rich Model 4 replay samples must be dictionaries."
                )

            if "game_id" not in sample:

                raise KeyError(
                    "Replay sample is missing 'game_id'."
                )

            game_id = int(
                sample[
                    "game_id"
                ]
            )

            if game_id not in self.game_lookup:

                raise KeyError(
                    "Replay sample references unknown "
                    f"game_id={game_id}."
                )

            game = self.game_lookup[
                game_id
            ]

            if (
                split is None
                or game.get(
                    "split"
                ) == split
            ):

                self.sample_indices.append(
                    sample_index
                )

    def __len__(
        self,
    ) -> int:

        return len(
            self.sample_indices
        )

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, Any]:

        sample_index = (
            self.sample_indices[
                index
            ]
        )

        sample = self.buffer[
            sample_index
        ]

        game_id = int(
            sample[
                "game_id"
            ]
        )

        game = self.game_lookup[
            game_id
        ]

        if "winner_ids" not in game:

            raise KeyError(
                "Game metadata is missing 'winner_ids'."
            )

        if "current_player" not in sample:

            raise KeyError(
                "Replay sample is missing 'current_player'."
            )

        return {
            "sample":
                sample,

            "sample_index":
                sample_index,

            "game":
                game,

            "target_wdl":
                wdl_target_from_winners(
                    current_player=sample[
                        "current_player"
                    ],
                    winner_ids=game[
                        "winner_ids"
                    ],
                ),
        }


# ============================================================
# Batch collation
# ============================================================


def collate_model4_batch(
    items: list[dict[str, Any]],
) -> dict[str, torch.Tensor]:
    """
    Build one dynamically padded Model 4 training batch.

    For each replay sample:

        policy_action_ids
            exact canonical action IDs represented by the MCTS root

        visit_counts
            MCTS visit target in the SAME ordering

    Batch tensors:

        observations
            [B, 258] float32

        legal_action_ids
            [B, N] int64

        legal_action_mask
            [B, N] bool

        target_policy
            [B, N] float32

        target_wdl
            [B] int64

        visit_counts
            [B, N] float32

        action_counts
            [B] int64

    N is the maximum legal-action count IN THIS BATCH.

    There is no global 64-action capacity. Padded slots use
    ACTION_SPACE_SIZE, which Model 4 reserves as its padding ID.
    """

    if not items:

        raise ValueError(
            "Cannot collate an empty Model 4 batch."
        )

    batch_size = len(
        items
    )

    observations_np = []

    policy_action_ids_np = []

    visit_counts_np = []

    target_wdl = []

    sample_indices = []

    game_ids = []

    current_players = []

    action_counts = []

    # --------------------------------------------------------
    # Validate each replay sample before allocating batch tensors
    # --------------------------------------------------------

    for item_index, item in enumerate(
        items
    ):

        if "sample" not in item:

            raise KeyError(
                f"Batch item {item_index} is missing 'sample'."
            )

        sample = item[
            "sample"
        ]

        for required_key in (
            "observation",
            "policy_action_ids",
            "visit_counts",
            "current_player",
            "game_id",
        ):

            if required_key not in sample:

                raise KeyError(
                    f"Replay sample is missing '{required_key}'."
                )

        observation = np.asarray(
            sample[
                "observation"
            ],
            dtype=np.float32,
        )

        if observation.shape != (
            258,
        ):

            raise ValueError(
                "Expected observation shape (258,), "
                f"got {observation.shape}."
            )

        policy_action_ids = np.asarray(
            sample[
                "policy_action_ids"
            ],
            dtype=np.int64,
        )

        visit_counts = np.asarray(
            sample[
                "visit_counts"
            ],
            dtype=np.float32,
        )

        if policy_action_ids.ndim != 1:

            raise ValueError(
                "policy_action_ids must be a 1D array."
            )

        if visit_counts.ndim != 1:

            raise ValueError(
                "visit_counts must be a 1D array."
            )

        if (
            len(
                policy_action_ids
            )
            == 0
        ):

            raise ValueError(
                "Every replay position must contain "
                "at least one policy action."
            )

        if (
            len(
                policy_action_ids
            )
            != len(
                visit_counts
            )
        ):

            raise ValueError(
                "policy_action_ids and visit_counts "
                "must have identical lengths."
            )

        # The rich replay intentionally stores both legality and
        # search-policy ordering. For Model 4 we require them to
        # remain exactly aligned.
        if "legal_action_ids" in sample:

            legal_action_ids = np.asarray(
                sample[
                    "legal_action_ids"
                ],
                dtype=np.int64,
            )

            if not np.array_equal(
                legal_action_ids,
                policy_action_ids,
            ):

                raise ValueError(
                    "legal_action_ids and policy_action_ids "
                    "are not in identical order. Model 4 "
                    "requires exact action/visit alignment."
                )

        if np.any(
            policy_action_ids < 0
        ):

            raise ValueError(
                "policy_action_ids contains a negative ID."
            )

        if np.any(
            policy_action_ids
            >= ACTION_SPACE_SIZE
        ):

            raise ValueError(
                "policy_action_ids contains an ID outside "
                f"0..{ACTION_SPACE_SIZE - 1}."
            )

        if (
            np.unique(
                policy_action_ids
            ).size
            != policy_action_ids.size
        ):

            raise ValueError(
                "policy_action_ids contains duplicate "
                "canonical action IDs."
            )

        if not np.all(
            np.isfinite(
                visit_counts
            )
        ):

            raise ValueError(
                "visit_counts contains NaN or infinity."
            )

        if np.any(
            visit_counts < 0
        ):

            raise ValueError(
                "visit_counts contains a negative value."
            )

        visit_sum = float(
            visit_counts.sum()
        )

        if (
            not np.isfinite(
                visit_sum
            )
            or visit_sum <= 0.0
        ):

            raise ValueError(
                "visit_counts must sum to a positive "
                "finite value."
            )

        observations_np.append(
            observation
        )

        policy_action_ids_np.append(
            policy_action_ids
        )

        visit_counts_np.append(
            visit_counts
        )

        target_wdl.append(
            int(
                item[
                    "target_wdl"
                ]
            )
        )

        sample_indices.append(
            int(
                item[
                    "sample_index"
                ]
            )
        )

        game_ids.append(
            int(
                sample[
                    "game_id"
                ]
            )
        )

        current_players.append(
            int(
                sample[
                    "current_player"
                ]
            )
        )

        action_counts.append(
            int(
                len(
                    policy_action_ids
                )
            )
        )

    # --------------------------------------------------------
    # Dynamic batch width
    # --------------------------------------------------------

    max_actions = max(
        action_counts
    )

    observations = torch.from_numpy(
        np.stack(
            observations_np,
            axis=0,
        )
    )

    legal_action_ids = torch.full(
        (
            batch_size,
            max_actions,
        ),
        fill_value=ACTION_SPACE_SIZE,
        dtype=torch.long,
    )

    legal_action_mask = torch.zeros(
        (
            batch_size,
            max_actions,
        ),
        dtype=torch.bool,
    )

    visit_counts = torch.zeros(
        (
            batch_size,
            max_actions,
        ),
        dtype=torch.float32,
    )

    target_policy = torch.zeros(
        (
            batch_size,
            max_actions,
        ),
        dtype=torch.float32,
    )

    # --------------------------------------------------------
    # Populate real action slots
    # --------------------------------------------------------

    for row in range(
        batch_size
    ):

        count = action_counts[
            row
        ]

        ids_tensor = torch.from_numpy(
            policy_action_ids_np[
                row
            ]
        )

        visits_tensor = torch.from_numpy(
            visit_counts_np[
                row
            ]
        )

        legal_action_ids[
            row,
            :count,
        ] = ids_tensor

        legal_action_mask[
            row,
            :count,
        ] = True

        visit_counts[
            row,
            :count,
        ] = visits_tensor

        target_policy[
            row,
            :count,
        ] = (
            visits_tensor
            / visits_tensor.sum()
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
            "Normalized policy targets do not sum to 1."
        )

    if torch.any(
        target_policy[
            ~legal_action_mask
        ]
        != 0
    ):

        raise RuntimeError(
            "Padded target-policy slots must be exactly zero."
        )

    if torch.any(
        legal_action_ids[
            ~legal_action_mask
        ]
        != ACTION_SPACE_SIZE
    ):

        raise RuntimeError(
            "Padded action-ID slots do not contain the "
            "Model 4 padding ID."
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
            torch.tensor(
                target_wdl,
                dtype=torch.long,
            ),

        # Useful for diagnostics. The training loss can use
        # target_policy directly.
        "visit_counts":
            visit_counts,

        "action_counts":
            torch.tensor(
                action_counts,
                dtype=torch.long,
            ),

        "sample_indices":
            torch.tensor(
                sample_indices,
                dtype=torch.long,
            ),

        "game_ids":
            torch.tensor(
                game_ids,
                dtype=torch.long,
            ),

        "current_players":
            torch.tensor(
                current_players,
                dtype=torch.long,
            ),
    }


# ============================================================
# DataLoader construction
# ============================================================


def make_model4_dataloader(
    replay: dict,
    split: str | None,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
) -> DataLoader:
    """
    Convenience constructor used by Model 4 pretraining.

    Dynamic padding occurs inside collate_model4_batch independently
    for every batch.
    """

    if batch_size <= 0:

        raise ValueError(
            "batch_size must be positive."
        )

    dataset = Model4ReplayDataset(
        replay=replay,
        split=split,
    )

    if len(
        dataset
    ) == 0:

        raise ValueError(
            f"No replay samples found for split={split!r}."
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_model4_batch,
    )


# ============================================================
# Diagnostics
# ============================================================


def summarize_model4_batch(
    batch: dict[str, torch.Tensor],
) -> None:
    """
    Small human-readable batch summary for smoke testing.
    """

    action_counts = batch[
        "action_counts"
    ]

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
        "actions per state:",
        "min=",
        int(
            action_counts.min()
        ),
        "median=",
        float(
            action_counts.float().median()
        ),
        "max=",
        int(
            action_counts.max()
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
