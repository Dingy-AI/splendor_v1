# Splendor Rust engine — first milestone

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

The production Python launcher, Model 4 architecture, PyTorch inference,
training settings, and MCTS V6 search behavior are unchanged. This module is
not yet a replacement for the Python MCTS environment interface.

## Windows setup

Install Rust using [rustup](https://rustup.rs/). Use the default Windows MSVC
toolchain and follow its compiler prerequisite instructions. Reopen PowerShell
after installation and check `rustc --version` and `cargo --version`.

Use your existing project virtual environment with Python 3.10–3.13. From the directory containing
the `splendor_v1` repository (the same directory used for `python -m splendor_v1...`):

```powershell
python -m pip install maturin
python -m maturin develop --release --manifest-path splendor_v1/rust_engine/Cargo.toml
python -m pytest splendor_v1/rust_engine/test_parity.py -q
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
The new extension itself does not import PyTorch or require a checkpoint.

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

The release extension and checks were run on Linux with CPython 3.12. The
Windows build instructions are provided, but a Windows build has not been
executed in this environment.

After changing Python card or payment definitions, regenerate and revalidate:

```powershell
python -m splendor_v1.rust_engine.generate_tables
python -m pytest splendor_v1/rust_engine/test_parity.py -q
```

## Next milestone: MCTS and batched self-play

Port `mcts_batched/mcts_v5_direct.py` and `mcts/node.py` search behavior into Rust,
including adaptive simulation budgets, PUCT, backup perspective, lazy state
materialization, root noise, and tree reuse. Use a deterministic evaluator to
compare search behavior before integrating PyTorch. Native MCTS can reuse its
action buffers and apply already-selected legal actions directly.

Then collect pending leaves from many native games, evaluate batches in the
existing Python Model 4 network, and return compatible completed-game replay
samples to the current training pipeline. Benchmark completed self-play
games/hour against **200** using the same checkpoint and search settings.
Model 5 architecture changes remain separate.
