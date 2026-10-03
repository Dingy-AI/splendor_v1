# Evaluate a model against a direct agent

Run from your project directory containing `splendor_v1`, with Rust V2 already
installed. Replace the checkpoint path below:

```powershell
python -m splendor_v1.rust_engine_v2.evaluate_heuristic --model "path/to/model.pt" --opponent random --games 1000 --workers 256 --batch-size 256 --device cuda --precision fp32 --simulations 400 --report splendor_v1/rust_engine_v2/model_vs_random.json
```

Choose an opponent with `--opponent`:

| Option | Behavior |
| --- | --- |
| `random` | Uniform random legal move, using an independent RNG per game. |
| `greedy` | Existing GreedyAgent: buy the highest-point affordable card, otherwise prefer taking gems. |
| `h3` | Existing HeuristicAgent3: scores buys, gem takes and reserves using its rule-based priorities. |
| `h12` | Existing HeuristicAgent12: protects its root candidates, adds strategic candidates, and compares sampled H3 continuation rollouts. |

For an H3 match, change `--opponent random` to `--opponent h3` and use a separate
report filename. For H12, default rollout settings are 8 sampled worlds per
candidate and at most 200 decisions per rollout. These can be changed with
`--heuristic-rollouts` and `--heuristic-max-rollout-steps`; record the same
settings when comparing checkpoints.

## Model search and opponent moves

The model is participant **A** and the direct agent is participant **B**.
Only A runs neural PUCT/MCTS. Its default adaptive search is capped at 400
simulations, with the same V6 settings as `evaluate_models`. Moves use greedy
visit selection and no root noise. Add `--fixed-simulations` to disable early
stopping for the model. Each simulated leaf, including simulated opponent
turns, is evaluated by the model. The opponent's direct policy is not injected
into those search simulations.

B chooses one legal action at each of its real decisions, including discard
and noble sub-decisions, without MCTS or GPU inference. Random play samples
native legal IDs directly. Greedy/H3/H12 score a reconstructed Python state
using their existing agent code; the adapter checks Python/Rust legal-action
agreement before accepting an action. H12's sampled continuation rollouts also
run in Python. Actual game transitions and A's searches execute in Rust.

After a player change, the previous search tree is discarded. Same-player
model sub-decisions may retain the model's tree. The report's B search and
simulation counts are zero: its direct decisions appear as `heuristic_decisions`.

## Pairing, results and failures

1,000 total games means 500 seed pairs. Each seed is used twice, with A playing
player 0 in one game and player 1 in the other. Seeds start at 500000 by
default; `--base-seed` changes this. `--games` must be even. Random/H12 policy
RNGs are seeded from each pair's board seed, independent of slot scheduling.
Changing concurrent slots or batch size does not change their random streams.

The report contains `model_wins`, `heuristic_wins`, `draws`, `failed_games`,
per-seat and per-game results, the exact loaded checkpoint hash, search
settings, opponent configuration, GPU inference counters and throughput.
The original A/B result fields remain available for comparison with
`evaluate_models` reports. `model_match_score` is the balanced score over
**complete valid pairs**: `(model wins + 0.5 * draws) / paired games`.
Always read `completed_pairs` alongside that score.

Default behavior stops at the first failure and saves a partial report.
`--keep-going` can record a decision cap or a dead-end verified against the
full Python trajectory and then continue the remaining original requested
jobs. Failed games never count as draws and never get replacement seeds.
Their paired partners are excluded from the balanced match score. Native
mismatches, invalid opponent actions and inference errors always stop.
Random agents can cause long games or rules dead-ends; inspect failure counts
before interpreting a result. The default real-decision cap is 300, adjustable
with `--max-game-steps`. This is separate from H12's internal rollout cap.

Use `--include-moves` to include action IDs in game results. Failure diagnostics
always include the actual action trajectory and state. Their default directory
is `splendor_v1/rust_engine_v2/heuristic_evaluation_failures`.

## Performance and installation

`--workers` denotes concurrent native game slots, not CPU threads. Only one
model is loaded, and every neural batch belongs to that model. The evaluator
does not train, change checkpoints, create training replay samples, or modify
the training replay buffer. Memory holds the model, active trees, action paths
and small result records.

Greedy/H3/H12 scoring runs at the real-move boundary in the driver process;
H12 can become a CPU bottleneck. Its action time is shown in
`pipeline_profile.heuristic_action_seconds` and is already included in
`game_boundary_seconds`, so do not add those times together. Measure throughput
separately from self-play or two-model matches.

This is a Python-only update. Copy the supplied runtime files into
`splendor_v1/rust_engine_v2/`: `evaluate_heuristic.py`, the updated
`evaluate_models.py`, and `arena.py` (the latter is the same version supplied
with the previous two-model evaluator). Keep both existing Rust engine source
folders and the rest of your project. No Rust rebuild or table regeneration is
needed if V2 is already installed. Test and documentation files are optional
for running matches.

```powershell
python -m pytest splendor_v1/rust_engine_v2/test_evaluate_heuristic.py -q
```

Verification compares complete games and every neural request against Python
V6 searches with each of the four direct opponents. Tests also cover paired
seats, independent RNGs, invalid actions, legal mismatches, first-player caps,
shared dead-end trajectory verification, partial reports and an unchanged
real Model 4 checkpoint in a CPU match. CUDA performance needs measurement
on your hardware.

Combined V1/V2 validation on Linux CPython 3.14.7: **289 tests passed**,
with three CUDA-only tests skipped. The original two-model evaluator remains
covered by its complete-game/request parity and checkpoint integrity tests.
