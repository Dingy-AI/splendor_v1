import argparse
import pickle
from pathlib import Path


def convert_legacy_replay(input_path: str, output_path: str) -> None:
    input_path = Path(input_path)
    output_path = Path(output_path)

    with input_path.open("rb") as f:
        data = pickle.load(f)

    if not isinstance(data, dict):
        raise TypeError(
            f"Expected top-level pickle object to be a dict, got {type(data).__name__}."
        )

    if "buffer" not in data:
        raise KeyError("Pickle does not contain a 'buffer' key.")

    buffer = data["buffer"]

    if not isinstance(buffer, list):
        raise TypeError(
            f"Expected data['buffer'] to be a list, got {type(buffer).__name__}."
        )

    # Preserve the legacy replay samples exactly as-is.
    # Since the old file has no reliable game boundaries, all legacy
    # positions are explicitly marked as training-only.
    data["format_version"] = max(int(data.get("format_version", 1)), 2)
    data["split"] = "train"
    data["split_strategy"] = "all_train"
    data["train_indices"] = list(range(len(buffer)))
    data["val_indices"] = []

    # Provenance / summary metadata.
    data["is_legacy"] = True
    data["source"] = data.get("source", "h16")
    data["num_train_positions"] = len(buffer)
    data["num_val_positions"] = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Loaded:    {input_path}")
    print(f"Saved:     {output_path}")
    print(f"Positions: {len(buffer):,}")
    print("Split:     100% train / 0% validation")
    print("Original buffer entries were not modified.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a legacy replay-buffer pickle into a training-only "
            "split format without altering the original replay samples."
        )
    )
    parser.add_argument("input", help="Path to the legacy .pkl file.")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help=(
            "Output .pkl path. Defaults to '<input_stem>_train_only.pkl' "
            "next to the input file."
        ),
    )

    args = parser.parse_args()
    input_path = Path(args.input)

    if args.output is None:
        output_path = input_path.with_name(
            f"{input_path.stem}_train_only{input_path.suffix}"
        )
    else:
        output_path = Path(args.output)

    if input_path.resolve() == output_path.resolve():
        raise ValueError(
            "Refusing to overwrite the input file. Choose a different output path."
        )

    convert_legacy_replay(str(input_path), str(output_path))


if __name__ == "__main__":
    main()
