# Splendor Rust backend — game engine, MCTS, and concurrent self-play

This is an explicit two-player backend for the existing game engine at reference
commit `9761a20`. The established self-play baseline is **200 games/hour**.
This change does not rerun that baseline or claim a new neural self-play speed.

## Implemented

- Rust legal-action generation, including optional gold-payment choices, in the
  same order as the Python engine.
- All seven action types, discards, noble selection, final-round completion,
  and the fewest-purchased-cards tiebreak.
- The same 1,139 action IDs and 258 float32 observation values, including the
  opponent's hidden-reserve mask.
- Fixed-size, copyable native state; only the returned action/observation
  buffers allocate. Cards and payment IDs are generated from Python's tables.
- Import/export using the existing full-state replay schema, preserving card
  identities and deck order. JSON is used at import/export boundaries; moves
  and observations execute in Rust without Python rule callbacks.
- Native neural PUCT matching `mcts_batched/mcts_v5_direct.py` as used by V6:
  selection, expansion, backup, adaptive budgets, first-action tie handling,
  terminal/dead-end values, lazy child states, and subtree reuse.
- A resumable evaluator boundary: native search returns the 258-value leaf
  observation and ordered legal IDs, then accepts policy probabilities and a
  player-to-move WDL scalar. Python runs Model 4 inference and training.
- Root noise and visit-temperature sampling use the existing NumPy algorithms.
  Both current search priors and original network priors are retained.
- Tree reuse discards unreachable siblings, moves retained native nodes without
  copying their states, and flips values when the root player changes.
- Concurrent native games send leaf requests to one shared Model 4 inference
  owner. Worker threads execute native search with the GIL released; requests
  use an in-process queue rather than process IPC.
- Completed games use the existing rich replay schema, WDL/policy targets,
  whole-game train/validation splits, and V6 training/checkpoint/resume handling.

`python -m splendor_v1.rust_engine.run_training` selects native self-play for
the existing V6 training loop. It reads your current V6 settings and paths;
the original `run_training_v6.py` file keeps its Python backend default.
Model 4 architecture and PyTorch training are unchanged. `RustMCTS` also offers
a native tree API rather than the existing Python `Node` interface.

## Windows setup

Install Rust using [rustup](https://rustup.rs/). Use the default Windows MSVC
toolchain (Rust 1.87 or newer) and follow its compiler prerequisite instructions. Reopen PowerShell
after installation and check `rustc --version` and `cargo --version`.

Use your existing project virtual environment with Python 3.10–3.14. From the directory containing
the `splendor_v1` repository (the same directory used for `python -m splendor_v1...`):

```powershell
python -m pip install --upgrade "maturin>=1.9.4,<2"
python -m maturin develop --release --manifest-path splendor_v1/rust_engine/Cargo.toml
python -m pytest splendor_v1/rust_engine/test_parity.py -q
python -m pytest splendor_v1/rust_engine/test_search_parity.py -q
python -m pytest splendor_v1/rust_engine/test_self_play.py -q
cargo test --manifest-path splendor_v1/rust_engine/Cargo.toml --no-default-features
```

The first Cargo build downloads the locked dependencies. `--release` enables
compiler optimization; development builds are unsuitable for speed comparisons.
If you are not using an activated virtual environment, build and install a wheel:

```powershell
python -m maturin build --release --manifest-path splendor_v1/rust_engine/Cargo.toml --out splendor_v1/rust_engine/dist
python -m pip install <path-to-the-built-wheel>
```

These commands use your already-installed Python environment dependencies.
The game engine itself does not import PyTorch or require a checkpoint. The
search comparison suite needs PyTorch to run the original Python MCTS and uses
the supplied `training_v6/data/model4_v6_inference_latest.pt` for neural checks.
The phase-3 package contains the complete `rust_engine` directory, including the
Python 3.14 compatibility update. Extract it into the inner `splendor_v1` folder
and rebuild with the commands above. No table regeneration is needed to install.

### Python 3.14 build error with the original package

If Cargo reports that PyO3 0.23.5 supports only Python through 3.13, replace
`Cargo.toml`, `Cargo.lock`, and `pyproject.toml` with the updated files. The
updated package pins PyO3 0.27.2 and permits Python 3.14. Then repeat the
installation commands above in the same activated virtual environment.
No `PYO3_USE_ABI3_FORWARD_COMPATIBILITY` override is needed. Check the selected
interpreter with `python --version` and `python -c "import sys; print(sys.executable)"`.

## API

```python
from splendor_v1.rust_engine import reset, observation, to_python

state = reset(seed=420)
actions = state.legal_action_ids()       # Same ordered canonical IDs as Python.
inputs = observation(state)             # Fresh NumPy float32 array, shape (258,).
branch = state.clone()                  # Native copy, independent of state.
reward, terminated = branch.step(actions[0])
replay_state = to_python(branch)        # Existing GameState for replay/UI boundaries.
```

`from_python(existing_game_state)` imports an existing position. The native
`RustState(snapshot_json)` constructor also accepts a JSON encoding of
`training_v2.state_serializer.serialize_state(...)`.

Starting-position creation intentionally uses Python's existing NumPy shuffle.
After import, the full ordered decks are native and draws pop from the same
end as Python. No new chance-resampling or hidden-information search scheme
is introduced. This preserves the current implementation, including its
reward being the actor's point gain on that decision.

Imported positions must have two players, the existing three noble slots and
twelve visible slots, known base-game card/noble IDs, matching hidden-reserve
flags, and conserved gem totals. Unknown/unsupported data raises `ValueError`.
Custom expansion cards and 3–4 players are outside this milestone. Illegal
actions and moves after a finished game are rejected without changing state.

## Validation

The differential suite compares the original Python engine and Rust:
full snapshots, ordered action IDs, exact float32 observations, rewards,
termination, cloned successor states, forced decisions, hidden information,
empty decks, and winner tiebreaks. It also checks generated data freshness and
the full existing action-ID round trip.

For a larger correctness run:

```powershell
python -m splendor_v1.rust_engine.check_parity --games 10000 --report splendor_v1/rust_engine/parity_report.json
```

The decision cap is a test-policy limit, not a new game rule. The report lists
finished, capped, and no-legal-action games separately. These are random-policy
games and do not measure Model 4 self-play throughput.

The included 10,000-seed run checked 897,885 decisions and 58,687 alternate
successors. All state, action-order, observation, reward, and termination
comparisons matched. There were 9,770 naturally completed games and 230 games
that reached a nonterminal position with no legal actions in **both** engines;
none reached the decision cap. This existing Python behavior is preserved and
should be addressed separately from the speed rewrite. It is not a measured
failure rate for the learned Model 4 self-play policy.

The original release extension and checks were run on Linux with CPython 3.12.
The Python 3.14 compatibility update was built on Linux with CPython 3.14.7;
all 59 parity tests and 3 Rust unit tests passed with PyO3 0.27.2. The
Windows build instructions are provided, but a Windows build has not been
executed in this environment.

After changing Python card or payment definitions, regenerate and revalidate:

```powershell
python -m splendor_v1.rust_engine.generate_tables
python -m maturin develop --release --manifest-path splendor_v1/rust_engine/Cargo.toml
python -m pytest splendor_v1/rust_engine/test_parity.py -q
```

## Native MCTS API

```python
from splendor_v1.rust_engine import reset
from splendor_v1.rust_engine.mcts import RustMCTS

# Use your existing loaded Model 4 in eval mode, on CPU or CUDA.
model.eval()
search = RustMCTS(reset(seed=42), model=model, seed=42, simulations=400)
best_action_id = search.search(add_root_noise=True)
chosen_action_id = search.select_action(temperature=1.0)
root_statistics = search.summary
reward, terminated = search.advance(chosen_action_id)  # Apply once; retain subtree.
next_state = search.root_state()
```

`advance` is the move application for the search-owned game state; do not apply
the same move twice. If keeping a separate Python game for UI/replay, apply the
matching Python action there once. To import a different external position,
create a new `RustMCTS(from_python(state), ...)` instance.

Supported search settings match V6's neural PUCT: `simulations`, `c_puct`,
`dirichlet_epsilon`, `adaptive_simulations`, `min_simulations`, `check_interval`,
`target_visits_per_action`, `single_action_simulations`, and `stability_checks`.
`dirichlet_alpha` and RNGs live in the Python wrapper. The temperature and noise
schedules remain a caller/game-loop choice. Defaults use the worker's 400-simulation
cap, 80 minimum, checks every 20 simulations, 20 target visits per action, four
simulations for single-action roots, and three stability checks.

Native search supports the production neural PUCT path. UCB, random/heuristic
rollout modes, and the separate teacher mode are not implemented in this backend.
No game-rule or Model 5 architecture changes are included.

For manual or future shared inference:

```python
search.begin(add_root_noise=False)
while (request := search.next_request()) is not None:
    observation, legal_action_ids = request
    priors, value = evaluator.evaluate(observation, legal_action_ids)
    search.respond(priors, value)
```

The request contains a NumPy float32 observation and int64 legal IDs. Responses
must contain probabilities in that same order (not logits or a full 1,139-way
vector) and `P(WIN) - P(LOSS)` for the player to move. Rust selection, terminal
processing, backup, and subtree compaction release the Python GIL. Trees hold
no Python objects. JSON is used for diagnostics, initial state import, and
replay-state export once per real game decision, outside the simulation loop.

`Model4Evaluator.evaluate_batch(requests)` scores multiple pending leaves using
the existing legal scorer, padding to the largest action count in that batch.
`finish_searches(searches, evaluator)` resolves independently started trees
together. The native coordinator uses this evaluator for concurrent self-play.
Batched floating-point results may differ
slightly from single-position inference; the single-position bridge uses the
same forward call as the original evaluator.

## Search verification

The 38 search tests compare every neural request, full trees (including lazy
states), visits, values, priors, chosen actions, and all adaptive metadata with
the V6 Python search. They cover fixed/adaptive budgets, seeded root noise,
repeated searches with value flips and compaction, forced decisions, terminal
winners/ties, dead-end leaves, 71 legal candidates, visit-temperature sampling,
batch scheduling, and the supplied Model 4 checkpoint. Checks ran on Linux with
CPython 3.14.7 and CPU PyTorch; Windows and CUDA must be checked on your machine.
Together with 59 engine tests, 13 self-play/integration tests, and three Rust
unit tests, 113 tests passed.
The included 12-game extended run completed all games and matched across 1,167
decisions, 192,964 simulations, 185,227 neural requests, and 906 player-perspective
flips. There were no decision caps or dead ends in that synthetic-evaluator run.
Search values/metadata use tight absolute floating-point tolerances; observations,
legal IDs, visits, action choices, and stopping decisions are compared exactly.

Run a native search with the supplied checkpoint:

```powershell
python -m splendor_v1.rust_engine.smoke_search --simulations 200
# Optional, with your CUDA-enabled PyTorch installation:
python -m splendor_v1.rust_engine.smoke_search --simulations 200 --device cuda
```

Optional extended development comparison (not a throughput benchmark):

```powershell
python -m splendor_v1.rust_engine.check_search_parity --games 12 --simulations 200 --report splendor_v1/rust_engine/search_parity_report.json
```

This checker uses a deterministic synthetic evaluator and compares complete
trees through many game decisions, alternating fixed and adaptive search.
Finished, capped, and dead-end games are reported separately. You do not need
to rerun the extended checker after every installation or model-weight update.

## Native self-play and V6 training

After installing the release extension, run a short completed-game check:

```powershell
python -m splendor_v1.rust_engine.benchmark_self_play --games 2 --workers 2 --simulations 24 --train-step --report splendor_v1/rust_engine/native_smoke.json
```

This loads `INFERENCE_SNAPSHOT_PATH` from your existing V6 settings and writes
samples into a fresh, separate replay buffer. `--train-step` checks one optimizer
update on a separate model without saving it. It does not change your training
replay or checkpoint. Reduced simulations make this a functionality check,
not a comparison against the 200-games/hour baseline.

For a throughput comparison on your GPU, keep the checkpoint, simulation cap,
adaptive settings, and worker count from your established workload:

```powershell
python -m splendor_v1.rust_engine.benchmark_self_play --games 16 --workers 16 --device cuda --report splendor_v1/rust_engine/native_benchmark.json
```

The simulation cap and other search defaults come from your current V6 file.
Use `--checkpoint <path>` if the baseline used a different snapshot. The report
records completed games/hour, positions, search statistics, inference batch
sizes, device, and checkpoint SHA256. Start with this 16-game measurement;
use more games if completion time varies considerably. Optional `--replay-out
<path>` saves only this benchmark's new replay.

To run normal training with native self-play:

```powershell
python -m splendor_v1.rust_engine.run_training
```

This uses the same resume choice, model/replay paths, games per iteration,
worker count, search settings, GPU batching, optimizer, validation, scheduler,
and checkpoint logic as your V6 configuration. Your configured resume files
must exist, as with the original launcher. No existing training source files
need replacing. Startup prints the native backend; some inherited V6 log
labels still say "multiprocess", while replay metadata and summaries record
the actual Rust/threaded execution.

`NUM_SELF_PLAY_WORKERS` now controls native game threads. One PyTorch model
owns inference for all threads; `GPU_MAX_BATCH_SIZE` and `GPU_BATCH_WAIT_MS`
still control batching. Process spawn/stagger/IPC startup settings are not
used. A new coordinator loads the current inference snapshot each iteration,
so subsequent games use the weights produced by training.

Replay observations, native legal IDs, semantic actions (including gold
payments), visits/priors/values, adaptive-search metadata, rewards, final
termination, winners, and model/job provenance retain the existing format.
Python replay objects are constructed at each real decision, not each search
simulation. Capped or failed games raise an error. Nonterminal dead-end games
are never committed as completed games. A search-only dead-end marker cannot
become an actual game result when a subtree is reused.

The training launcher can replace up to 10 shared rules dead-ends per block.
Before rejection, it saves a JSON diagnostic and replays the complete action
trajectory in Python and Rust, checking every state, legal action set, reward,
and termination flag. Mismatches, inference errors, and step caps still stop
training. Replacement attempts use fresh job indices/seeds through V6's existing
job factory. Completed-game throughput includes time spent on rejected games;
summaries report actual submitted attempts, failures, and diagnostic paths.
This preserves the current rules/action space and excludes unfinished samples;
it does not add a pass move or assign a draw/winner to a dead-end.

Use `--max-rejected-games 0` on the training launcher for strict failure handling.
The coordinator API and benchmark remain strict by default. Diagnostics go to
`splendor_v1/training_v6/data/native_failures`, configurable with `--failure-dir`.
Replay a diagnostic without loading a checkpoint or GPU:

```powershell
python -m splendor_v1.rust_engine.replay_failure "path/to/failure.json"
```

## Self-play verification and remaining measurement

The 13 self-play tests compare complete Python/native trajectories exactly
for three seeded games, including replay arrays and search statistics. They
check shared batches, limits, concurrent scheduling, whole-game splits,
failure shutdown, unfinished-game rejection, replay save/load, and training
targets. An actual Model 4 test runs two V6 iterations with training,
validation, checkpoints/resume data, and verifies that the second iteration
loads the newly trained weights.

The supplied `self_play_smoke_report.json` records a separate Linux CPU run:
two completed games, 123 replay positions, shared inference batches, and one
successful training step. Its 24-simulation budget and CPU hardware differ
from the baseline, so its throughput does not demonstrate a speedup.
Windows/CUDA throughput remains to be measured on your machine. The next
milestone is measuring and tuning native self-play against **200 completed
games/hour** under the same workload. Model 5 changes remain separate.
