"""Native neural PUCT with an explicit observation/action-ID evaluator boundary.

This is an opt-in backend, not a drop-in Python Node/MCTS replacement. The
existing V6 module keeps its Python default; rust_engine.run_training selects the native backend.
"""
import json

import numpy as np


class RustMCTS:
    """Own one native tree; reuse it by calling advance(action_id) after a move.

    Evaluators implement evaluate(observation, legal_action_ids) -> (priors, value).
    Value is P(WIN)-P(LOSS) for the player to move. Priors are probabilities in
    legal-action order, not logits or a full 1,139-entry policy.
    """

    def __init__(self, state, *, evaluator=None, model=None, seed=None,
                 dirichlet_alpha=0.3, **search_config):
        from splendor_rust import RustSearch
        if evaluator is not None and model is not None:
            raise ValueError("Pass either evaluator or model")
        if not np.isfinite(dirichlet_alpha) or dirichlet_alpha <= 0:
            raise ValueError("dirichlet_alpha must be positive and finite")
        self.native = RustSearch(state, json.dumps(search_config))
        self.evaluator = evaluator if model is None else Model4Evaluator(model)
        self.dirichlet_alpha = float(dirichlet_alpha)
        self.rng = np.random.default_rng(seed)
        self.action_rng = np.random.default_rng(seed)

    def begin(self, *, add_root_noise=False):
        if self.native.active:
            raise ValueError("Finish the active search before beginning another")
        noise = None
        if add_root_noise:
            n = len(self.native.root_action_ids())
            # Python's NumPy RNG is deliberately retained for seeded compatibility.
            noise = self.rng.dirichlet([self.dirichlet_alpha] * n).tolist() if n else []
        self.native.begin(noise)

    def next_request(self):
        request = self.native.next_request()
        if request is None:
            return None
        observation, legal_ids = request
        return np.asarray(observation, dtype=np.float32), np.asarray(legal_ids, dtype=np.int64)

    def respond(self, priors, value):
        if hasattr(priors, "detach"):
            priors = priors.detach().cpu().tolist()
        else:
            priors = np.asarray(priors).tolist()
        self.native.respond(priors, float(value))

    def search(self, *, add_root_noise=False):
        """Synchronous convenience loop. Tree selection and backup execute in Rust."""
        if self.evaluator is None:
            raise ValueError("search() needs an evaluator; use begin/next_request/respond for manual inference")
        self.begin(add_root_noise=add_root_noise)
        while (request := self.next_request()) is not None:
            self.respond(*self.evaluator.evaluate(*request))
        return self.summary["best_action_id"]

    @property
    def summary(self):
        """Root counts, priors, values, adaptive-search metadata, and memory counters."""
        return json.loads(self.native.summary_json())

    @property
    def last_search_metadata(self):
        return self.summary["metadata"]

    def advance(self, action_id):
        """Keep the chosen subtree, discard siblings, and adjust value perspective."""
        return self.native.advance(int(action_id))

    def root_state(self):
        return self.native.root_state()

    def select_action(self, temperature=0.0):
        """Same visit-temperature rule as ModelReplayGenerator V5."""
        if self.native.active:
            raise ValueError("Finish search before selecting an action")
        if not np.isfinite(temperature) or temperature < 0:
            raise ValueError("temperature must be nonnegative and finite")
        children = self.summary["children"]
        if not children:
            return None
        visits = np.asarray([c["visits"] for c in children], dtype=np.float64)
        if visits.sum() <= 0:
            return children[0]["action_id"]
        if temperature <= 1e-8:
            index = int(np.argmax(visits))
        else:
            with np.errstate(over="ignore", invalid="ignore"):
                weights = np.power(visits, 1.0 / temperature)
            total = weights.sum()
            index = int(np.argmax(visits)) if not np.isfinite(total) or total <= 0 else int(
                self.action_rng.choice(len(children), p=weights / total))
        return children[index]["action_id"]


class Model4Evaluator:
    """Run the existing Model 4 without reconstructing Python game states.

    evaluate_batch accepts pending leaves from independent native searches.
    It pads only to the current batch's largest legal-action count.
    """

    def __init__(self, model):
        if model.training:
            raise ValueError("Model 4 must be in eval mode; call model.eval() before creating the evaluator")
        self.model = model

    def evaluate(self, observation, legal_action_ids):
        return self.evaluate_batch([(observation, legal_action_ids)])[0]

    def evaluate_batch(self, requests):
        import torch
        if not requests:
            return []
        if self.model.training:
            raise ValueError("Model 4 must remain in eval mode during search")
        weight = self.model.action_embedding.weight
        observations = np.stack([np.asarray(obs, dtype=np.float32) for obs, _ in requests])
        ids = [np.asarray(actions, dtype=np.int64) for _, actions in requests]
        if observations.shape != (len(requests), 258) or any(a.ndim != 1 or len(a) == 0 for a in ids):
            raise ValueError("Expected 258-value observations and nonempty 1D legal IDs")
        obs_tensor = torch.as_tensor(observations, dtype=weight.dtype, device=weight.device)
        with torch.inference_mode():
            if len(requests) == 1:
                # Use the exact original direct-evaluator forward call for single leaves.
                action_tensor = torch.as_tensor(ids[0], dtype=torch.long, device=weight.device)
                logits, wdl_logits = self.model.forward_legal(obs_tensor, action_tensor)
                policy_tensor = torch.softmax(logits[0], dim=0).unsqueeze(0)
            else:
                width = max(map(len, ids))
                padded = np.zeros((len(ids), width), dtype=np.int64)
                mask = np.zeros_like(padded, dtype=np.bool_)
                for row, actions in enumerate(ids):
                    padded[row, :len(actions)] = actions
                    mask[row, :len(actions)] = True
                action_tensor = torch.as_tensor(padded, dtype=torch.long, device=weight.device)
                mask_tensor = torch.as_tensor(mask, dtype=torch.bool, device=weight.device)
                logits, wdl_logits = self.model.forward_legal(obs_tensor, action_tensor, mask_tensor)
                # The model masks padding with -inf; transfer all policies together.
                policy_tensor = torch.softmax(logits, dim=-1)
            wdl = torch.softmax(wdl_logits, dim=-1)
            values = (wdl[:, 2] - wdl[:, 0]).cpu().tolist()
            policy_rows = policy_tensor.cpu().tolist()
            policies = [row[:len(actions)] for row, actions in zip(policy_rows, ids)]
        return list(zip(policies, values))


def finish_searches(searches, evaluator):
    """Resolve started independent searches together through one shared evaluator.

    Call begin() on each search first. This helper is the evaluator boundary for
    future native self-play; it does not create games, replay, or training jobs.
    """
    if len({id(s.native) for s in searches}) != len(searches):
        raise ValueError("Each search in a batch must be distinct")
    while True:
        pending = [(s, r) for s in searches if (r := s.next_request()) is not None]
        if not pending:
            break
        requests = [r for _, r in pending]
        if hasattr(evaluator, "evaluate_batch"):
            responses = evaluator.evaluate_batch(requests)
        else:
            responses = [evaluator.evaluate(*r) for r in requests]
        if len(responses) != len(pending):
            raise ValueError("Evaluator returned the wrong batch size")
        for (search, _), (priors, value) in zip(pending, responses):
            search.respond(priors, value)
    return [s.summary["best_action_id"] for s in searches]
