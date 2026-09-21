import pickle
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================

data_dir = Path(
    "splendor_v1/training/data/h16/split"
)

output_path = (
    data_dir
    / "h16_split_all.pkl"
)


# ============================================================
# HELPERS
# ============================================================

def get_local_split_indices(data, path):
    """
    Return local train/validation indices for one replay file.

    Supported metadata:
      1. game_records with split='train' / 'val'
      2. train_indices / val_indices
      3. top-level split='train' / 'val'

    No random fallback is allowed.
    """
    buffer = data["buffer"]
    n = len(buffer)

    # --------------------------------------------------------
    # 1. Whole-game metadata
    # --------------------------------------------------------
    game_records = data.get("game_records")

    if game_records:
        train_indices = []
        val_indices = []

        train_game_ids = set(
            data.get("train_game_ids", [])
        )
        val_game_ids = set(
            data.get("val_game_ids", [])
        )

        for record in game_records:
            start = record["start_index"]
            end = record["end_index"]

            if (
                not isinstance(start, int)
                or not isinstance(end, int)
                or start < 0
                or end < start
                or end > n
            ):
                raise ValueError(
                    f"{path.name}: invalid game range "
                    f"[{start}, {end}) for buffer length {n}"
                )

            split = record.get("split")

            # Compatibility if split is represented only
            # through train_game_ids / val_game_ids.
            if split is None:
                game_id = record.get("game_id")

                if game_id in train_game_ids:
                    split = "train"
                elif game_id in val_game_ids:
                    split = "val"

            if split == "train":
                train_indices.extend(
                    range(start, end)
                )

            elif split == "val":
                val_indices.extend(
                    range(start, end)
                )

            else:
                raise ValueError(
                    f"{path.name}: game record has no valid "
                    f"train/val split: {record}"
                )

        return (
            train_indices,
            val_indices,
            "game_records",
        )

    # --------------------------------------------------------
    # 2. Explicit sample indices
    # --------------------------------------------------------
    if (
        "train_indices" in data
        or "val_indices" in data
    ):
        train_indices = list(
            data.get("train_indices", [])
        )
        val_indices = list(
            data.get("val_indices", [])
        )

        for idx in (
            train_indices
            + val_indices
        ):
            if (
                not isinstance(idx, int)
                or idx < 0
                or idx >= n
            ):
                raise ValueError(
                    f"{path.name}: split index out of range: "
                    f"{idx}"
                )

        return (
            train_indices,
            val_indices,
            "explicit_indices",
        )

    # --------------------------------------------------------
    # 3. Entire file belongs to one split
    # --------------------------------------------------------
    split = data.get("split")

    if split == "train":
        return (
            list(range(n)),
            [],
            "file_level_train",
        )

    if split == "val":
        return (
            [],
            list(range(n)),
            "file_level_val",
        )

    raise ValueError(
        f"{path.name}: no recognized split metadata.\n"
        "Expected one of:\n"
        "  - game_records with train/val tags\n"
        "  - train_indices / val_indices\n"
        "  - top-level split='train' or split='val'"
    )


# ============================================================
# FIND REPLAY BUFFER FILES
# ============================================================

pkl_files = sorted(
    path
    for path in data_dir.glob("*.pkl")
    if path != output_path
)


print("=" * 70)
print("SPLIT-AWARE REPLAY BUFFER COMBINER")
print("=" * 70)

print(
    f"Found {len(pkl_files)} .pkl files:"
)

for path in pkl_files:
    print(
        f"  {path.name}"
    )

print()


# ============================================================
# LOAD + COMBINE
# ============================================================

combined_buffer = []

combined_train_indices = []
combined_val_indices = []

source_files = []
source_ranges = []

files_loaded = 0
files_skipped = 0


for path in pkl_files:

    print(
        f"Loading: {path.name}"
    )

    try:
        with path.open("rb") as f:
            data = pickle.load(f)

    except Exception as e:
        print(
            f"  SKIPPED - could not load: {e}"
        )
        files_skipped += 1
        continue


    # --------------------------------------------------------
    # Verify replay shape
    # --------------------------------------------------------

    if not isinstance(data, dict):
        print(
            "  SKIPPED - not a dictionary"
        )
        files_skipped += 1
        continue

    if "buffer" not in data:
        print(
            "  SKIPPED - no 'buffer' key"
        )
        files_skipped += 1
        continue

    buffer = data["buffer"]

    if not isinstance(
        buffer,
        (list, tuple),
    ):
        print(
            "  SKIPPED - 'buffer' is not a list/tuple"
        )
        files_skipped += 1
        continue


    # --------------------------------------------------------
    # Read this file's split before concatenation
    # --------------------------------------------------------

    try:
        (
            local_train_indices,
            local_val_indices,
            split_strategy,
        ) = get_local_split_indices(
            data,
            path,
        )

    except Exception as e:
        print(
            f"  SKIPPED - split metadata error: {e}"
        )
        files_skipped += 1
        continue


    # --------------------------------------------------------
    # Convert local indices to combined/global indices
    # --------------------------------------------------------

    offset = len(
        combined_buffer
    )

    global_train_indices = [
        offset + idx
        for idx in local_train_indices
    ]

    global_val_indices = [
        offset + idx
        for idx in local_val_indices
    ]


    # --------------------------------------------------------
    # Append samples
    # --------------------------------------------------------

    start_index = offset

    combined_buffer.extend(
        buffer
    )

    end_index = len(
        combined_buffer
    )

    combined_train_indices.extend(
        global_train_indices
    )

    combined_val_indices.extend(
        global_val_indices
    )


    # --------------------------------------------------------
    # Provenance
    # --------------------------------------------------------

    source_files.append(
        path.name
    )

    source_ranges.append(
        {
            "source_file":
                path.name,

            "start_index":
                start_index,

            "end_index":
                end_index,

            "num_samples":
                len(buffer),

            "num_train":
                len(local_train_indices),

            "num_val":
                len(local_val_indices),

            "split_strategy":
                split_strategy,

            "is_legacy":
                bool(
                    data.get(
                        "is_legacy",
                        False,
                    )
                ),
        }
    )

    print(
        f"  samples:    {len(buffer):,}"
    )

    print(
        f"  train:      "
        f"{len(local_train_indices):,}"
    )

    print(
        f"  validation: "
        f"{len(local_val_indices):,}"
    )

    print(
        f"  strategy:   "
        f"{split_strategy}"
    )

    print(
        f"  global range: "
        f"[{start_index:,}, {end_index:,})"
    )

    print()

    files_loaded += 1


# ============================================================
# VALIDATE COMBINED SPLIT
# ============================================================

if len(combined_buffer) == 0:
    raise RuntimeError(
        "No replay samples were found."
    )

if len(combined_train_indices) == 0:
    raise RuntimeError(
        "Combined training split is empty."
    )

if len(combined_val_indices) == 0:
    raise RuntimeError(
        "Combined validation split is empty."
    )


train_set = set(
    combined_train_indices
)

val_set = set(
    combined_val_indices
)


overlap = (
    train_set
    & val_set
)

if overlap:
    raise RuntimeError(
        f"Train/validation overlap detected: "
        f"{len(overlap):,} samples."
    )


covered = (
    train_set
    | val_set
)

if len(covered) != len(
    combined_buffer
):
    missing = (
        len(combined_buffer)
        - len(covered)
    )

    raise RuntimeError(
        f"{missing:,} combined samples were not assigned "
        "to train or validation."
    )


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 70)
print("COMBINATION SUMMARY")
print("=" * 70)

print(
    f"Files loaded:       "
    f"{files_loaded}"
)

print(
    f"Files skipped:      "
    f"{files_skipped}"
)

print(
    f"Total samples:      "
    f"{len(combined_buffer):,}"
)

print(
    f"Training samples:   "
    f"{len(combined_train_indices):,}"
)

print(
    f"Validation samples: "
    f"{len(combined_val_indices):,}"
)

print(
    f"Validation fraction:"
    f" "
    f"{len(combined_val_indices) / len(combined_buffer):.2%}"
)


# ============================================================
# CREATE COMBINED REPLAY BUFFER
# ============================================================

combined_data = {
    "format_version": 2,

    "capacity":
        len(combined_buffer),

    "buffer":
        combined_buffer,

    "position":
        0,

    # These are GLOBAL indices into combined_buffer.
    "train_indices":
        combined_train_indices,

    "val_indices":
        combined_val_indices,

    "split_strategy":
        "combined_preserved_source_splits",

    "source_files":
        source_files,

    "source_ranges":
        source_ranges,

    "num_source_files":
        files_loaded,

    "num_train_positions":
        len(combined_train_indices),

    "num_val_positions":
        len(combined_val_indices),
}


# ============================================================
# SAVE
# ============================================================

with output_path.open("wb") as f:
    pickle.dump(
        combined_data,
        f,
        protocol=pickle.HIGHEST_PROTOCOL,
    )


print()
print(
    "Saved combined replay buffer to:"
)

print(
    f"  {output_path}"
)


# ============================================================
# VERIFY SAVED FILE
# ============================================================

with output_path.open("rb") as f:
    test_data = pickle.load(f)


print()
print("=" * 70)
print("VERIFY")
print("=" * 70)

print(
    f"Combined length: "
    f"{len(test_data['buffer']):,}"
)

print(
    f"Train indices:   "
    f"{len(test_data['train_indices']):,}"
)

print(
    f"Val indices:     "
    f"{len(test_data['val_indices']):,}"
)

print(
    f"Overlap:         "
    f"{len(set(test_data['train_indices']) & set(test_data['val_indices'])):,}"
)


obs, policy, value = (
    test_data["buffer"][0]
)


print(
    "Observation:",
    obs.shape
)

print(
    "Policy:",
    policy.shape
)

print(
    "Value:",
    value
)
