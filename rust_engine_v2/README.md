# Splendor rust_engine_v2

V2 is a separate, opt-in backend. It keeps Model 4 checkpoints, V6 search
settings, replay targets, training, validation, scheduler, and resume handling.
Your existing `rust_engine` package and launch command remain available.

The measured reference is 381.980 games/hour for V1 at 32 workers, 100 games,
400 simulations, seed 10000, checkpoint SHA256
`ed1d4d704d9faef1c473fec11ee5c84b7bf7741a473de19c25aea02196de69c9`.
The established original baseline is 200 games/hour. V2 GPU speed has **not**
been measured here; use the same hardware/checkpoint/workload for comparison.

## What changed

- A Rust arena owns many independent search trees, gathers pending leaves,
  pads action IDs/masks, and applies whole prediction batches. Python does not
  create a worker thread, Future, or queue request for each game/leaf.
- Observations and legal inputs cross the boundary as contiguous binary
  arrays, rather than Python lists of floats and per-leaf calls.
- NumPy noise/action RNG streams, noise/temperature schedules, adaptive
  budgets, legal-action order, backups, and subtree reuse retain V1 behavior.
- The evaluator checks IDs/masks on the CPU. Its checkpoint-compatible Model 4
  inference subclass avoids the general validator's GPU-to-Python predicates.
  The original model/training source is not edited.
- An eight-shape LRU reuses host and CUDA input buffers. CUDA host buffers are
  pinned, input copies are enqueued asynchronously, and policies plus values
  return in one blocking readback as arrays rather than per-row Python lists.
- FP16/BF16 are optional for the attention trunk. Policy scoring, WDL heads,
  and softmax remain FP32, preserving closely spaced output logits better.
  Weights/checkpoint files stay FP32. Default inference is FP32/eager.
- Optional `torch.compile` and shape buckets can reduce launch overhead.
  Startup messages and a ten-second heartbeat show activity before game one
  finishes. Compile warmup can take time before the first heartbeat.

Python still creates seeded starts and rich replay objects once per **real
decision**. Simulations and cross-game leaf scheduling execute in Rust.
Failed, capped, or nonterminal dead-end games never commit partial replays.
Only two-player base Splendor/neural PUCT is supported, as in V1.

## Install on Windows

Extract the package into the inner `splendor_v1` folder so that
`splendor_v1/rust_engine_v2/Cargo.toml` exists. Keep the phase-3 `rust_engine`
folder: V2 reuses its Python replay helpers. Installing V2 creates a separate
`splendor_rust_v2` extension and does not replace `splendor_rust`.

In your activated virtual environment, from the project parent directory:

```powershell
python -m maturin develop --release --manifest-path splendor_v1/rust_engine_v2/Cargo.toml
python -m pytest splendor_v1/rust_engine_v2 -q
```

Use Python 3.10–3.14, Rust >=1.87 with the Windows MSVC prerequisites,
maturin >=1.9.4,<2, and your existing NumPy/PyTorch/Gymnasium/pytest dependencies.
V2 uses the same pinned PyO3 0.27.2 as the Python 3.14 compatibility update.
No table regeneration is needed to install. Rebuild after changing Rust code,
dependencies, or generated tables; Python settings/model updates need no rebuild.
For wheel installation, use `python -m maturin build --release` with the same
manifest path, then install the resulting wheel into the active environment.

## First comparison: V2 FP32 with 32 games in flight

```powershell
python -m splendor_v1.rust_engine_v2.benchmark_self_play --games 100 --workers 32 --batch-size 32 --device cuda --precision fp32 --report splendor_v1/rust_engine_v2/benchmark_fp32_32.json
```

The V6 file supplies search/adaptive settings, inference checkpoint, and default
seeds. `--workers` means **concurrent native game slots**, not Python threads.
One Rust driver pumps all slots and one PyTorch owner evaluates their batches.
`native_worker_threads: 0` is intentional. Batch sizes smaller than game slots
are supported with fair round-robin scheduling.

There is no batching wait queue: Rust supplies the available leaf batch directly.
V6 process startup/stagger/shutdown and `GPU_BATCH_WAIT_MS` settings do not apply.
The progress timeout checks a scheduler/evaluation iteration rather than the
interval between completed games; many healthy concurrent games may take longer
to produce the first completion. The game decision cap still rejects unfinished
games. A blocked native/CUDA call can only be diagnosed after that call returns.
The simulation cap remains unchanged unless you explicitly pass `--simulations`.

Each benchmark uses a fresh replay buffer; it does not update your training
replay/checkpoint. Optional `--replay-out <path>` saves only the new replay.
`--train-step` verifies one update on a separate normal Model 4 without saving it.
Reports include checkpoint SHA256, search settings, precision, batch sizes,
completed games/hour, neural requests, and pipeline timings.

## Profile inference before tuning

```powershell
python -m splendor_v1.rust_engine_v2.profile_inference --device cuda --batch-sizes 16 32 64 128 256 --precisions fp32 fp16 bf16 --report splendor_v1/rust_engine_v2/inference_gpu_profile.json
```

This reads the checkpoint and generates varied real native observations/legal
sets. It times inference without search/replay generation in the measured loop.
For every configuration it reports maximum policy/value differences, top-policy
agreement, and normalization error against the eager FP32 predictions for those
same positions. These are numerical checks, not a playing-strength guarantee.
Mixed precision can alter MCTS visits/actions and seeded trajectories; do not
expect exact search parity against FP32 in reduced-precision mode.

The profile separates host preparation, transfer enqueue, forward dispatch,
and blocking readback. CUDA events additionally measure input transfer, forward,
postprocessing, and readback on the device stream. Host and device timings overlap;
do not add them together. CUDA profiling adds overhead; standalone results are
**positions/second**, not completed games/hour. Unsupported configurations record
their error; the command fails if none succeed.

For a complete-game diagnostic run, add `--profile` to the self-play benchmark.
Leave it off for the final throughput comparison. Compilation warmup is excluded
from the standalone timed repetitions, but included in self-play wall time.

After reviewing the precision comparison, test FP16 at the same 32-slot workload:

```powershell
python -m splendor_v1.rust_engine_v2.benchmark_self_play --games 100 --workers 32 --batch-size 32 --device cuda --precision fp16 --report splendor_v1/rust_engine_v2/benchmark_fp16_32.json
```

Then test larger batches independently, for example 128 games/slots/batch size.
Using 128 slots with only 100 games creates at most 100 active games. Several
waves of games give a more sustained measurement; ensure comparable seed ranges.
More slots can increase tree/replay memory and may not improve throughput.

Optional compilation experiment:

```powershell
python -m splendor_v1.rust_engine_v2.profile_inference --device cuda --batch-sizes 32 64 128 --precisions fp32 fp16 --compile reduce-overhead --bucket-shapes --report splendor_v1/rust_engine_v2/inference_compiled.json
```

Compiler support depends on your PyTorch/Windows setup. V2 reports a clear error
and recommends `--compile off` if it fails; it does not silently substitute a
different execution mode. Buckets pad batch rows to powers of two and candidate
counts to multiples of 16, trading extra computation/memory for more stable
shapes. They are opt-in and may be slower. Do not enable compilation or padding
by default until your measurements support it.

## Normal training

```powershell
python -m splendor_v1.rust_engine_v2.run_training
```

This reads your existing `training_v6/run_training_v6.py` configuration and uses
its model/replay paths and resume choice. The configured resume files must exist.
The inference snapshot is freshly loaded each training iteration.
To select native concurrency/batching without editing that file:

```powershell
python -m splendor_v1.rust_engine_v2.run_training --games-in-flight 128 --batch-size 128 --precision fp16
```

Only use a precision/batch setting you have validated on your checkpoint/GPU.
With 100 games per iteration, that example runs at most 100 simultaneously.
Training remains the normal FP32 Model 4 pipeline; reduced precision affects
self-play inference only. Inference choices are recorded in game provenance.
Some inherited V6 log labels still say "multiprocess"; V2 logs/replay/summary
identify the native arena accurately. Stop the running process before changing
backends or rebuilding the Windows extension.

The training launcher permits up to 10 shared rules dead-end rejections per
block (`--max-rejected-games 0` stops immediately). Each rejection saves the
actual state and action trajectory in `splendor_v1/training_v6/data/native_failures`
and verifies the whole trajectory against Python before allowing a replacement.
Only completed games enter replay. Replacements get fresh job indices/seeds;
attempt/failure counts and all retry time are included in the block summary.
Rules, action IDs, and winner labels are unchanged. Native mismatches, inference
errors, and step caps still stop the block. Benchmark/API defaults remain strict.
Use `--failure-dir` to change the diagnostic directory, and inspect a saved
V1 or V2 failure without a model:

```powershell
python -m splendor_v1.rust_engine.replay_failure "path/to/failure.json"
```

## Compare two models

```powershell
python -m splendor_v1.rust_engine_v2.evaluate_models --model-a "path/to/new_model.pt" --model-b "path/to/old_model.pt" --games 1000 --workers 256 --batch-size 256 --device cuda --precision fp32 --report splendor_v1/rust_engine_v2/match_report.json
```

This plays 500 paired seeds with swapped seats, greedy moves and no root noise.
Both models share the GPU, with separate inference batches for each search's
owner. It matches the Python V6 evaluator's search ownership and tree reuse,
and leaves model files and training replay untouched. Use the report's
`complete_pair_results` for the balanced match score. Failures stop by default
and save a partial report; they never count as draws.

This Python-only addition needs no Rust rebuild when V2 is already installed.
See [EVALUATION.md](EVALUATION.md) for settings, reports, failure handling and
verification.

## Compare a model with a heuristic agent

```powershell
python -m splendor_v1.rust_engine_v2.evaluate_heuristic --model "path/to/model.pt" --opponent random --games 1000 --workers 256 --batch-size 256 --device cuda --precision fp32 --report splendor_v1/rust_engine_v2/model_vs_random.json
```

Choose `random`, `greedy`, `h3` or `h12`. The model uses Rust MCTS and GPU
batching; its opponent chooses direct moves using the selected policy.
Greedy/H3/H12 retain their existing Python scoring, including H12's rollouts.
Seats and seeds are paired, and failures never count as draws. This update
needs no rebuild. See [HEURISTIC_EVALUATION.md](HEURISTIC_EVALUATION.md).

## Validation

On Linux CPython 3.14.7 with CPU PyTorch, V2 passed 127 Python tests and six Rust
unit tests; three CUDA-specific tests were skipped. These include engine/search differential
tests, complete Python/V2 replay comparisons, 128-slot bulk gathering, bounded
batch fairness, retryable/atomic response rejection, failure/cap/dead-end handling,
and two full V6 training iterations with validation/checkpoint/resume checks.
FP32 predictions match the existing evaluator within tight absolute tolerances.
BF16 CPU checks compare predictions and confirm finite normalized outputs.
`torch.compile` default mode also completed a standalone CPU inference check.

The supplied CPU smoke report used only 24 simulations, completed two games,
committed 123 positions, and trained one step. The CPU profiler is diagnostic
evidence only; neither demonstrates a speedup over your GPU baseline.
Windows/CUDA, FP16 on your GPU, and CUDA compilation are not tested here.
CUDA-specific tests are included and skipped when no CUDA device is available.

```powershell
cargo test --manifest-path splendor_v1/rust_engine_v2/Cargo.toml --no-default-features --locked
```

After card/payment definitions change, regenerate both backends' tables and
rebuild/revalidate each backend that you use.
