"""Standalone Model 4 batch/precision sweep with FP32 prediction comparisons.

Uses real native game observations/legal sets. Checkpoint/replay are read-only.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import numpy as np
import torch

from splendor_v1.rust_engine_v2 import reset
from splendor_v1.rust_engine_v2.inference import (
    InferenceOptions, PackedModel4Evaluator, load_model,
)


def positions(count, seed_start):
    rng = np.random.default_rng(seed_start)
    requests = []
    for index in range(count):
        state = reset(seed_start + index)
        for _ in range(int(rng.integers(0, 65))):
            actions = state.legal_action_ids()
            if state.game_over or not actions: break
            state.step(int(rng.choice(actions)))
        actions = state.legal_action_ids()
        if state.game_over or not actions:
            state = reset(seed_start + index)
            actions = state.legal_action_ids()
        requests.append((np.asarray(state.observation(), dtype=np.float32), np.asarray(actions, dtype=np.int64)))
    return requests


def pack(requests):
    width = max(len(ids) for _, ids in requests)
    actions = np.zeros((len(requests), width), dtype=np.int64)
    mask = np.zeros_like(actions, dtype=np.bool_)
    for row, (_, ids) in enumerate(requests):
        actions[row, :len(ids)] = ids; mask[row, :len(ids)] = True
    return np.stack([obs for obs, _ in requests]), actions, mask


def comparison(reference, actual, mask):
    error = np.abs(reference - actual)
    return dict(max_policy_absolute_error=float(error[:, :-1][mask].max()),
        max_value_absolute_error=float(error[:, -1].max()),
        policy_argmax_agreement=float(np.mean(reference[:, :-1].argmax(axis=1)
                                             == actual[:, :-1].argmax(axis=1))),
        finite=bool(np.isfinite(actual).all()),
        max_policy_sum_error=float(np.abs(actual[:, :-1].sum(axis=1) - 1).max()))


def main(argv=None):
    from splendor_v1.training_v6 import run_training_v6 as v6
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path(v6.INFERENCE_SNAPSHOT_PATH))
    parser.add_argument("--device", default=str(v6.device))
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[16, 32, 64, 128, 256])
    parser.add_argument("--precisions", choices=("fp32", "fp16", "bf16"), nargs="+", default=["fp32", "fp16"])
    parser.add_argument("--compile", dest="compile_mode", choices=("off", "default", "reduce-overhead"), default="off")
    parser.add_argument("--bucket-shapes", action="store_true")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--seed-start", type=int, default=10000)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    if min(args.batch_sizes + [args.repeats, args.cpu_threads]) <= 0 or args.warmup < 0 or args.seed_start < 0:
        parser.error("Invalid batch sizes, repeat count, or seed")
    torch.set_num_threads(args.cpu_threads)
    print(f"Loading Model 4 on {args.device}; generating {max(args.batch_sizes)} real positions", flush=True)
    model = load_model(args.checkpoint, args.device)
    requests = positions(max(args.batch_sizes), args.seed_start)
    baseline = PackedModel4Evaluator(model)
    results = []
    for batch_size in args.batch_sizes:
        arrays = pack(requests[:batch_size])
        reference = baseline.evaluate_packed(*arrays)
        for precision in args.precisions:
            options = InferenceOptions(precision=precision, compile_mode=args.compile_mode,
                                       profile=True, bucket_shapes=args.bucket_shapes)
            record = dict(batch_size=batch_size, options=asdict(options), candidates=arrays[1].shape[1])
            try:
                evaluator = PackedModel4Evaluator(model, options)
                for _ in range(args.warmup): evaluator.evaluate_packed(*arrays)
                evaluator.reset_profile()
                started = time.perf_counter()
                for _ in range(args.repeats): actual = evaluator.evaluate_packed(*arrays)
                elapsed = time.perf_counter() - started
                record.update(status="ok", wall_seconds=elapsed,
                    pipeline_positions_per_second=batch_size * args.repeats / elapsed,
                    comparison_to_fp32=comparison(reference, actual, arrays[2]),
                    profile=evaluator.profile())
                print(f"batch={batch_size:3d} precision={precision}: "
                      f"{record['pipeline_positions_per_second']:,.0f} positions/s | "
                      f"agreement={record['comparison_to_fp32']['policy_argmax_agreement']:.1%}", flush=True)
            except (ValueError, RuntimeError) as exc:
                record.update(status="unavailable", error=str(exc))
                print(f"batch={batch_size} precision={precision}: unavailable: {exc}", flush=True)
            results.append(record)
    report = dict(device=args.device, checkpoint=str(args.checkpoint),
        checkpoint_sha256=model.checkpoint_sha256,
        torch_version=torch.__version__, gpu_name=torch.cuda.get_device_name(torch.device(args.device))
            if torch.device(args.device).type == "cuda" else None,
        warmup=args.warmup, repeats=args.repeats, seed_start=args.seed_start, results=results,
        note="Standalone inference; not completed games/hour. CUDA events add profiling overhead.")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not any(row["status"] == "ok" for row in results):
        raise RuntimeError("No inference configuration completed successfully")
    return report


if __name__ == "__main__": main()
