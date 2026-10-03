"""NumPy boundary and per-game RNGs for the Rust-owned search arena."""
import json
import numpy as np

from splendor_v1.rust_engine.mcts import RustMCTS
from splendor_v1.rust_engine_v2.self_play_settings import native_search_settings


def decode_batch(batch):
    token, slots, ready, observations, actions, mask, width = batch
    rows = len(slots)
    return (token, slots, ready,
        np.frombuffer(observations, dtype="<f4").reshape(rows, 258),
        np.frombuffer(actions, dtype="<i8").reshape(rows, width),
        np.frombuffer(mask, dtype=np.bool_).reshape(rows, width))


def response_bytes(output, rows, width):
    array = np.asarray(output)
    if array.shape != (rows, width + 1):
        raise ValueError("Evaluator returned the wrong packed response shape")
    double = array.dtype == np.float64
    return np.asarray(array, dtype="<f8" if double else "<f4", order="C").tobytes(), double


class ArenaSearch(RustMCTS):
    """The replay recorder's root interface, with no separate Python leaf loop."""

    def __init__(self, arena, slot, state, search_config, seed, adaptive_simulations=True):
        self.arena, self.slot = arena, slot
        settings = native_search_settings(search_config, adaptive_simulations)
        self.dirichlet_alpha = settings.pop("dirichlet_alpha")
        self.arena.add(slot, state, json.dumps(settings))
        mcts_seed, action_seed = np.random.SeedSequence(int(seed)).spawn(2)
        self.rng, self.action_rng = np.random.default_rng(mcts_seed), np.random.default_rng(action_seed)
        self.native = self  # Existing rich replay recorder only needs root_action_ids.
        self._summary = None

    @property
    def active(self):
        return self.arena.active(self.slot)

    def root_action_ids(self):
        return self.arena.root_action_ids(self.slot)

    def begin(self, *, add_root_noise=False):
        noise = None
        if add_root_noise:
            n = len(self.root_action_ids())
            noise = self.rng.dirichlet([self.dirichlet_alpha] * n).tolist() if n else []
        self._summary = None
        self.arena.begin(self.slot, noise)

    @property
    def summary(self):
        if self._summary is None:
            self._summary = json.loads(self.arena.summary_json(self.slot))
        return self._summary

    def root_state(self):
        return self.arena.root_state(self.slot)

    def advance(self, action_id):
        self._summary = None
        return self.arena.advance(self.slot, int(action_id))

    def search(self, **kwargs):
        raise RuntimeError("ArenaSearch is resumed through RustArena.gather/respond")

    def next_request(self):
        raise RuntimeError("Use the arena's packed gather/respond boundary")

    def respond(self, *args):
        raise RuntimeError("Use the arena's packed gather/respond boundary")
