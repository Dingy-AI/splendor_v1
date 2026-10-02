"""Single-tree compatibility API; production V2 self-play uses RustArena."""
import json
import numpy as np

from splendor_v1.rust_engine.mcts import RustMCTS as BaseRustMCTS, finish_searches
from splendor_v1.rust_engine_v2.inference import PackedModel4Evaluator as Model4Evaluator


class RustMCTS(BaseRustMCTS):
    def __init__(self, state, *, evaluator=None, model=None, seed=None,
                 dirichlet_alpha=0.3, **search_config):
        from splendor_rust_v2 import RustSearch
        if evaluator is not None and model is not None:
            raise ValueError("Pass either evaluator or model")
        if not np.isfinite(dirichlet_alpha) or dirichlet_alpha <= 0:
            raise ValueError("dirichlet_alpha must be positive and finite")
        self.native = RustSearch(state, json.dumps(search_config))
        self.evaluator = evaluator if model is None else Model4Evaluator(model)
        self.dirichlet_alpha = float(dirichlet_alpha)
        self.rng, self.action_rng = np.random.default_rng(seed), np.random.default_rng(seed)
