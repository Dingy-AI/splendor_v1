import pickle
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================

data_dir = Path(
    "splendor_v1/training/data"
)

output_path = (
    data_dir
    / "all_replay_buffers_combined.pkl"
)


# ============================================================
# FIND REPLAY BUFFER FILES
# ============================================================

pkl_files = sorted(
    path
    for path in data_dir.glob("*.pkl")

    # Do not accidentally combine the output
    # back into itself on a later run.
    if path != output_path
)


print("=" * 70)
print("REPLAY BUFFER COMBINER")
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
    # Verify this looks like one of our replay-buffer files
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


    if not isinstance(buffer, (list, tuple)):

        print(
            "  SKIPPED - 'buffer' is not a list/tuple"
        )

        files_skipped += 1
        continue


    print(
        f"  samples: {len(buffer):,}"
    )

    combined_buffer.extend(
        buffer
    )

    files_loaded += 1


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 70)
print("COMBINATION SUMMARY")
print("=" * 70)

print(
    f"Files loaded:  {files_loaded}"
)

print(
    f"Files skipped: {files_skipped}"
)

print(
    f"Total samples: {len(combined_buffer):,}"
)


if len(combined_buffer) == 0:

    raise RuntimeError(
        "No replay samples were found."
    )


# ============================================================
# CREATE COMBINED REPLAY BUFFER
# ============================================================

combined_data = {
    "capacity": len(combined_buffer),

    "buffer": combined_buffer,

    # Because capacity == current buffer size,
    # the next circular-buffer insertion would
    # begin replacing from index 0.
    "position": 0,

    # Optional metadata
    "source_files": [
        path.name
        for path in pkl_files
    ],

    "num_source_files": files_loaded,
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
    f"Saved combined replay buffer to:"
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