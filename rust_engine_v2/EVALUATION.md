# Rust V2 model-versus-model evaluation

Run from the project parent directory in the environment where
`splendor_rust_v2` is installed. Replace the two checkpoint paths with compatible
Model 4 inference or training checkpoints:

```powershell
python -m splendor_v1.rust_engine_v2.evaluate_models --model-a "path/to/new_model.pt" --model-b "path/to/old_model.pt" --games 1000 --workers 256 --batch-size 256 --device cuda --precision fp32 --simulations 400 --report splendor_v1/rust_engine_v2/match_report.json
```

This addition changes Python code only. If your current Rust V2 extension works,
you do **not** need to rebuild it or regenerate tables. Keep both the
`rust_engine` and `rust_engine_v2` source folders: V2 uses shared V1 helpers.
For a first installation, follow the build instructions in `README.md`.

## What the match does

- Loads A and B once and records each checkpoint's SHA256. It does not update
  weights, train either model, or write to the training replay buffer.
- Plays 1,000 total games as 500 pairs. Both games in a pair start from the
  same seeded board/decks; A plays player 0 in one and player 1 in the other.
  Seeds default to 500000 through 500499 for this command. `--games` must be even.
- Uses greedy root visit selection, temperature zero and no Dirichlet noise.
  The real player choosing a move owns that entire search: its network also
  evaluates simulated opponent turns. Reuses the tree only while that same
  real player retains control through discard/noble sub-decisions, then drops
  it when the opponent takes over. This matches the Python **V6** evaluator.
- Uses the same search settings for both models. The default is adaptive,
  capped at 400 new simulations per decision, with minimum 80, checks every
  20, target 20 visits per legal action, 3 stability checks, a 4-simulation
  single-action budget and `c_puct=3`. Add `--fixed-simulations` for exactly
  400 new simulations per decision, including single-action positions.
- Limits each game to 300 real decisions, including forced sub-decisions.

Use fixed checkpoint filenames while training continues; a changing
`inference_latest.pt` can select a different model on your next invocation.
The report identifies the exact bytes loaded in the current invocation.

## Concurrency and memory

`--workers 256` means up to 256 concurrent native game slots. It does not start
256 CPU threads. One process owns both models and a Rust arena containing the
active search trees. The combined gathered batch is capped by `--batch-size`;
requests are separated into an A batch and a B batch before inference and then
returned to their original slots. A combined batch of 256 might therefore make
two model calls of about 128 each, rather than one model call of 256.

The evaluator retains each game's actual action path for failure diagnosis,
plus small result records. Completed games release their native trees, and it
does not build rich training replay samples. Both models, their inference
buffers and active trees still require memory. Evaluation throughput must be
measured separately from your single-model self-play benchmark.

FP32 is the default. Optional `--precision fp16` or `bf16` affects the attention
trunk; policy/value heads and policy softmax remain FP32. Lower precision can
alter close decisions. Use the same precision and settings for comparisons.
`--compile off` is the default; `--profile` adds device timing overhead.

## Results and failures

Startup prints identify both models. Heartbeats default to every 10 seconds.
The JSON report includes wins for A and B, draws, per-seat results, per-game
scores, search statistics, checkpoint hashes, inference profiles and games/hour.
Add `--include-moves` to include every played action ID in successful results.

The primary paired score is
`complete_pair_results.model_a_match_score`: `(A wins + 0.5 * draws) / games`
using only pairs in which **both** seat-swapped games completed successfully.
`completed_pairs` states the number of valid pairs. A score of 0.5 is even;
larger favors A. The top-level score also includes completed games whose paired
partner failed; use the complete-pair score for a balanced comparison. A small
match is a smoke test, not a reliable playing-strength estimate.

By default any failure stops the match, saves a partial report and returns a
nonzero exit status. Shared nonterminal dead-ends and decision caps save a full
state/action diagnostic in `splendor_v1/rust_engine_v2/evaluation_failures`.
They count as failures, never wins or draws. `--keep-going` permits these two
failure types to continue the **original requested** jobs without replacement
seeds. A shared dead-end must replay identically against Python first; native
mismatches, malformed inference and unexpected errors always stop.
Inspect any discarded pairs before interpreting an incomplete match's score.

The progress timeout defaults to 1,800 seconds per scheduler/inference
iteration, adjustable with `--iteration-timeout-seconds`. A blocked native or
CUDA call can only be checked after it returns.

## Verification

```powershell
python -m pytest splendor_v1/rust_engine_v2/test_evaluate_models.py -q
```

Tests compare complete games and every model's inference requests against the
Python V6 evaluator with two distinct deterministic evaluators at different
batch/slot limits. They also check identical-model seat balance, fixed budgets,
shared dead-end trajectory verification, mismatch rejection, partial reports,
and a CPU match between two different Model 4 checkpoint files without changing
either file. CUDA performance requires validation on your own GPU.

The combined V1/V2 Python suites passed 270 tests on Linux CPython 3.14.7;
three CUDA-specific tests were skipped because this validation environment
has no GPU. No Windows/CUDA throughput claim is made by these tests.
