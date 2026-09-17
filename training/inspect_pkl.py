#!/usr/bin/env python3
"""
inspect_pickle.py

Inspect the structure of a trusted .pkl replay-data file without dumping the
entire dataset.

Useful for answering questions such as:
- What top-level object is stored?
- What does one training sample contain?
- Is there any game_id / episode_id / trajectory_id information available?
- Are samples grouped by game, or are they just a flat sequence?

WARNING:
    pickle.load() can execute arbitrary Python code.
    Only run this on pickle files you trust.

Usage:
    python inspect_pickle.py path/to/replay.pkl

Optional:
    python inspect_pickle.py path/to/replay.pkl --samples 5 --depth 4
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import pickle
import sys
from pathlib import Path
from typing import Any

try:
    import numpy as np
except ImportError:
    np = None

try:
    import torch
except ImportError:
    torch = None


GAME_KEYWORDS = (
    "game",
    "game_id",
    "gameid",
    "episode",
    "episode_id",
    "episodeid",
    "trajectory",
    "trajectory_id",
    "trajectoryid",
    "match",
    "match_id",
    "matchid",
    "simulation",
    "simulation_id",
    "turn",
    "turn_number",
    "ply",
    "step",
)


def short_repr(value: Any, max_chars: int = 180) -> str:
    try:
        text = repr(value)
    except Exception as exc:
        text = f"<repr failed: {exc}>"

    if len(text) > max_chars:
        return text[: max_chars - 3] + "..."
    return text


def is_numpy_array(obj: Any) -> bool:
    return np is not None and isinstance(obj, np.ndarray)


def is_torch_tensor(obj: Any) -> bool:
    return torch is not None and isinstance(obj, torch.Tensor)


def summarize_array(obj: Any) -> str:
    if is_numpy_array(obj):
        summary = f"numpy.ndarray shape={obj.shape} dtype={obj.dtype}"
        if obj.size:
            try:
                summary += (
                    f" min={obj.min():.5g}"
                    f" max={obj.max():.5g}"
                    f" mean={obj.mean():.5g}"
                )
            except Exception:
                pass
        return summary

    if is_torch_tensor(obj):
        summary = (
            f"torch.Tensor shape={tuple(obj.shape)} "
            f"dtype={obj.dtype} device={obj.device}"
        )
        if obj.numel():
            try:
                x = obj.detach().float()
                summary += (
                    f" min={x.min().item():.5g}"
                    f" max={x.max().item():.5g}"
                    f" mean={x.mean().item():.5g}"
                )
            except Exception:
                pass
        return summary

    raise TypeError("Object is not an array/tensor")


def find_game_like_names(names):
    result = []
    for name in names:
        lower = str(name).lower()
        if any(keyword in lower for keyword in GAME_KEYWORDS):
            result.append(str(name))
    return result


def describe(
    obj: Any,
    *,
    name: str = "root",
    depth: int = 0,
    max_depth: int = 4,
    max_items: int = 5,
    visited: set[int] | None = None,
):
    if visited is None:
        visited = set()

    indent = "  " * depth
    obj_type = type(obj)
    type_name = f"{obj_type.__module__}.{obj_type.__name__}"

    if is_numpy_array(obj) or is_torch_tensor(obj):
        print(f"{indent}{name}: {summarize_array(obj)}")
        return

    if obj is None or isinstance(obj, (bool, int, float, str, bytes)):
        print(f"{indent}{name}: {type_name} = {short_repr(obj)}")
        return

    obj_id = id(obj)
    if obj_id in visited:
        print(f"{indent}{name}: {type_name} <already visited>")
        return
    visited.add(obj_id)

    try:
        length = len(obj)
    except Exception:
        length = None

    if length is None:
        print(f"{indent}{name}: {type_name}")
    else:
        print(f"{indent}{name}: {type_name} len={length}")

    if depth >= max_depth:
        return

    if isinstance(obj, dict):
        keys = list(obj.keys())
        print(f"{indent}  keys: {short_repr(keys[:20], 500)}")

        game_like = find_game_like_names(keys)
        if game_like:
            print(
                f"{indent}  >>> GAME/TRAJECTORY-LIKE KEYS FOUND: "
                f"{game_like}"
            )

        for key in keys[:max_items]:
            describe(
                obj[key],
                name=f"[{short_repr(key, 80)}]",
                depth=depth + 1,
                max_depth=max_depth,
                max_items=max_items,
                visited=visited,
            )

        if len(keys) > max_items:
            print(f"{indent}  ... {len(keys) - max_items} more keys")
        return

    if isinstance(obj, (list, tuple, collections.deque)):
        for i, child in enumerate(list(obj)[:max_items]):
            describe(
                child,
                name=f"[{i}]",
                depth=depth + 1,
                max_depth=max_depth,
                max_items=max_items,
                visited=visited,
            )

        if len(obj) > max_items:
            print(f"{indent}  ... {len(obj) - max_items} more items")
        return

    if dataclasses.is_dataclass(obj):
        fields = dataclasses.fields(obj)
        names = [field.name for field in fields]
        print(f"{indent}  dataclass fields: {names}")

        game_like = find_game_like_names(names)
        if game_like:
            print(
                f"{indent}  >>> GAME/TRAJECTORY-LIKE FIELDS FOUND: "
                f"{game_like}"
            )

        for field in fields[:max_items]:
            describe(
                getattr(obj, field.name),
                name=f".{field.name}",
                depth=depth + 1,
                max_depth=max_depth,
                max_items=max_items,
                visited=visited,
            )
        return

    if hasattr(obj, "__dict__"):
        attrs = vars(obj)
        names = list(attrs.keys())
        print(f"{indent}  attributes: {short_repr(names[:30], 500)}")

        game_like = find_game_like_names(names)
        if game_like:
            print(
                f"{indent}  >>> GAME/TRAJECTORY-LIKE ATTRIBUTES FOUND: "
                f"{game_like}"
            )

        for attr_name in names[:max_items]:
            describe(
                attrs[attr_name],
                name=f".{attr_name}",
                depth=depth + 1,
                max_depth=max_depth,
                max_items=max_items,
                visited=visited,
            )

        if len(names) > max_items:
            print(f"{indent}  ... {len(names) - max_items} more attributes")
        return

    print(f"{indent}  repr: {short_repr(obj)}")


def get_probable_sample_container(root: Any):
    if isinstance(root, (list, tuple, collections.deque)):
        return "root", root

    preferred = (
        "buffer",
        "replay_buffer",
        "replay",
        "memory",
        "data",
        "samples",
        "experiences",
        "transitions",
    )

    if isinstance(root, dict):
        for key in preferred:
            value = root.get(key)
            if isinstance(value, (list, tuple, collections.deque)):
                return f"root[{key!r}]", value

    if hasattr(root, "__dict__"):
        attrs = vars(root)
        for name in preferred:
            value = attrs.get(name)
            if isinstance(value, (list, tuple, collections.deque)):
                return f"root.{name}", value

    return None, None


def inspect_sample_schema(container, count: int):
    print()
    print("=" * 78)
    print("PROBABLE TRAINING SAMPLE CONTAINER")
    print("=" * 78)
    print(f"container type: {type(container).__name__}")
    print(f"sample count:   {len(container)}")

    if not container:
        print("Container is empty.")
        return

    for i, sample in enumerate(list(container)[:count]):
        print()
        print("-" * 78)
        print(f"SAMPLE {i}")
        print("-" * 78)

        if isinstance(sample, dict):
            print("dict keys:", list(sample.keys()))
            game_like = find_game_like_names(sample.keys())
            if game_like:
                print(">>> GAME/TRAJECTORY-LIKE KEYS:", game_like)

            for key, value in sample.items():
                if is_numpy_array(value) or is_torch_tensor(value):
                    print(f"  {key}: {summarize_array(value)}")
                else:
                    print(
                        f"  {key}: {type(value).__name__} "
                        f"{short_repr(value)}"
                    )

        elif isinstance(sample, (tuple, list)):
            print(f"{type(sample).__name__} length={len(sample)}")
            for j, value in enumerate(sample):
                if is_numpy_array(value) or is_torch_tensor(value):
                    print(f"  [{j}]: {summarize_array(value)}")
                else:
                    print(
                        f"  [{j}]: {type(value).__name__} "
                        f"{short_repr(value)}"
                    )

        elif dataclasses.is_dataclass(sample):
            print(
                "dataclass fields:",
                [f.name for f in dataclasses.fields(sample)],
            )

        elif hasattr(sample, "__dict__"):
            print("object attributes:", list(vars(sample).keys()))

        else:
            print(
                f"type={type(sample).__name__}: "
                f"{short_repr(sample)}"
            )


def inspect_possible_game_boundaries(container):
    print()
    print("=" * 78)
    print("GAME-BOUNDARY / GROUPING CHECK")
    print("=" * 78)

    if not container:
        print("No samples available.")
        return

    sample = container[0]

    if isinstance(sample, dict):
        names = list(sample.keys())
        game_like = find_game_like_names(names)

        if game_like:
            print("Explicit game/trajectory-like fields were found:")
            for name in game_like:
                print(f"  - {name}")
            print()
            print(
                "This likely gives you a way to split train/validation "
                "by game or trajectory."
            )
        else:
            print(
                "No obvious game_id / episode_id / trajectory_id field "
                "exists in the first sample."
            )
            print(
                "If every sample is stored independently in this flat "
                "container, exact game boundaries may not be recoverable "
                "from this pickle alone."
            )

    elif isinstance(sample, (tuple, list)):
        print(
            f"Samples are positional {type(sample).__name__} objects "
            f"with {len(sample)} fields."
        )
        print(
            "If the tuple is only something like "
            "(observation, policy, value), there is no explicit game ID."
        )
        print(
            "Compare the field order above with the code that writes "
            "replay samples."
        )

    elif hasattr(sample, "__dict__"):
        names = list(vars(sample).keys())
        game_like = find_game_like_names(names)

        if game_like:
            print("Explicit game/trajectory-like object attributes found:")
            for name in game_like:
                print(f"  - {name}")
        else:
            print("No obvious game/trajectory attribute was found.")

    else:
        print("Could not infer explicit game grouping from the sample type.")


def main():
    parser = argparse.ArgumentParser(
        description="Inspect a trusted pickle replay-data file."
    )
    parser.add_argument(
        "pickle_path",
        type=Path,
        help="Path to the .pkl file",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=3,
        help="Number of probable replay samples to summarize (default: 3)",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=4,
        help="Maximum recursive structure depth (default: 4)",
    )
    parser.add_argument(
        "--items",
        type=int,
        default=5,
        help="Maximum children shown per container (default: 5)",
    )

    args = parser.parse_args()
    path = args.pickle_path

    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        raise SystemExit(1)

    print("=" * 78)
    print("PICKLE INSPECTOR")
    print("=" * 78)
    print(f"File: {path}")
    print(f"Size: {path.stat().st_size / (1024 * 1024):.2f} MiB")
    print()
    print(
        "WARNING: pickle.load() can execute code. "
        "Only inspect files you trust."
    )
    print()

    with path.open("rb") as file:
        root = pickle.load(file)

    print("=" * 78)
    print("TOP-LEVEL STRUCTURE")
    print("=" * 78)

    describe(
        root,
        max_depth=max(1, args.depth),
        max_items=max(1, args.items),
    )

    container_name, container = get_probable_sample_container(root)

    if container is None:
        print()
        print("=" * 78)
        print("TRAINING SAMPLE SEARCH")
        print("=" * 78)
        print(
            "Could not automatically identify a probable flat sample container."
        )
        return

    print()
    print(
        f"Automatically identified probable sample container: "
        f"{container_name}"
    )

    inspect_sample_schema(
        container,
        count=max(1, args.samples),
    )
    inspect_possible_game_boundaries(container)


if __name__ == "__main__":
    main()
