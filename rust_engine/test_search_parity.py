"""Differential search tests against production V6's mcts_v5_direct.MCTS.

The tests compare every neural request, the entire tree, and search metadata.
They need PyTorch for the original Python reference and Model 4 integration.
"""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from splendor_v1.env.core.enums import ActionType, GemColor, NodeType
from splendor_v1.env.data.data import BASE_TIER_1, BASE_TIER_3
from splendor_v1.env.env import SplendorEnv
from splendor_v1.mcts_batched.mcts_v5_direct import MCTS
from splendor_v1.mcts_batched.direct_neural_evaluator import DirectNeuralEvaluator
from splendor_v1.rust_engine import from_python, reset, to_python
from splendor_v1.rust_engine.mcts import RustMCTS, Model4Evaluator, finish_searches
from splendor_v1.training_v2.state_serializer import serialize_state


class DeterministicEvaluator:
    """Nonconstant values exercise minimax perspective; optional uniform ties."""
    def __init__(self, uniform=False):
        self.uniform = uniform
        self.requests = []

    def evaluate(self, env, state, legal_actions):
        observation = env.observation_encoder.encoder(state)
        ids = [env.action_to_id(a) for a in legal_actions]
        priors, value = self.evaluate_arrays(observation, ids)
        self.requests.append((observation.copy(), ids, priors.tolist(), value))
        return torch.tensor(priors), value

    def evaluate_arrays(self, observation, ids):
        ids = np.asarray(ids, dtype=np.int64)
        weights = np.ones(len(ids), dtype=np.float32) if self.uniform else ((ids * 7 + 3) % 23 + 1).astype(np.float32)
        priors = weights / weights.sum()
        value = float(np.tanh(float(np.dot(np.asarray(observation, dtype=np.float64),
                                          (np.arange(258) % 7 - 3) / 19.0))))
        return priors, value


def make_env(seed=0):
    env = SplendorEnv()
    env.reset(seed=seed)
    return env


def canonical(state):
    return json.loads(json.dumps(serialize_state(state)))


def compare_tree(env, reference, native):
    rows = json.loads(native.native.tree_json())
    stack = [(reference, 0)]
    seen = 0
    while stack:
        py, index = stack.pop()
        row = rows[index]
        seen += 1
        assert row["action_id"] == (None if py.action is None else env.action_to_id(py.action))
        assert row["visits"] == py.visits
        assert row["value"] == pytest.approx(py.value, rel=0, abs=1e-13)
        assert row["prior"] == pytest.approx(py.prior, rel=0, abs=1e-15)
        assert row["network_prior"] == pytest.approx(getattr(py, "network_prior", 0.0), rel=0, abs=1e-15)
        assert row["expanded"] == py.expanded
        assert len(row["children"]) == len(py.children)
        assert (row["state"] is None) == (py.state is None)
        if py.state is not None:
            assert json.loads(row["state"]) == canonical(py.state)
        stack.extend(zip(py.children, row["children"]))
    assert seen == len(rows), "Native arena retained unreachable nodes"


def compare_search(env, reference, native, evaluator, root=None, noise=False):
    evaluator.requests.clear()
    action, root = reference.search(env, env.state, root=root, return_root=True, add_root_noise=noise)
    native.begin(add_root_noise=noise)
    for observation, ids, priors, value in evaluator.requests:
        request = native.next_request()
        assert request is not None, "Native search stopped before Python"
        np.testing.assert_array_equal(request[0], observation)
        assert request[1].tolist() == ids
        native.respond(priors, value)
    assert native.next_request() is None, "Native search requested additional leaves"
    assert native.summary["best_action_id"] == (None if action is None else env.action_to_id(action))
    compare_tree(env, root, native)
    metadata = native.last_search_metadata
    assert metadata.keys() == reference.last_search_metadata.keys()
    for key, value in reference.last_search_metadata.items():
        if isinstance(value, float):
            assert metadata[key] == pytest.approx(value, rel=0, abs=1e-13), key
        else:
            assert metadata[key] == value, key
    return action, root


def make_searches(env, *, seed=100, uniform=False, **config):
    evaluator = DeterministicEvaluator(uniform)
    reference = MCTS(rollout_type="neural", selection_type="puct", evaluator=evaluator, **config)
    reference.rng = np.random.default_rng(seed)
    native = RustMCTS(from_python(env.state), seed=seed, **config)
    return reference, native, evaluator


@pytest.mark.parametrize("seed", [0, 1, 42, 420])
@pytest.mark.parametrize("adaptive,noise,uniform", [(False, False, False), (True, False, False),
                                                   (True, True, False), (False, False, True)])
def test_full_tree_and_request_parity(seed, adaptive, noise, uniform):
    env = make_env(seed)
    config = dict(simulations=100, min_simulations=12, check_interval=5,
                  target_visits_per_action=0.5, stability_checks=3,
                  adaptive_simulations=adaptive)
    reference, native, evaluator = make_searches(env, uniform=uniform, **config)
    compare_search(env, reference, native, evaluator, noise=noise)
    assert native.summary["materialized_states"] <= 1 + native.last_search_metadata["actual_simulations"]
    assert native.summary["node_count"] > native.summary["materialized_states"]


@pytest.mark.parametrize("noise", [False, True])
def test_reuse_across_decisions_and_compaction(noise):
    env = make_env(42)
    config = dict(simulations=80, min_simulations=12, check_interval=5, target_visits_per_action=0.5)
    reference, native, evaluator = make_searches(env, **config)
    root = None
    for turn in range(35):
        action, root = compare_search(env, reference, native, evaluator, root, noise)
        if action is None or env.state.game_over:
            break
        # Include non-greedy chosen children, including a child with a lazy state.
        if turn % 4 == 0:
            action = root.children[-1].action
        child = next(c for c in root.children if c.action == action)
        previous_player = env.state.current_player
        env.step(action)
        native.advance(env.action_to_id(action))
        if child.state is None:
            child.state = env.state.clone()
        if env.state.current_player != previous_player:
            reference.flip_tree_values(child)
        child.parent = None
        root = child
        compare_tree(env, root, native)
        assert canonical(env.state) == json.loads(native.root_state().snapshot_json())
        if env.state.game_over:
            break


def test_same_player_discard_noble_and_single_action_budget():
    env = make_env(4)
    env.state.players[0].gems = {c: v for c, v in zip(GemColor, [4, 4, 3, 0, 0, 0])}
    env.state.bank = {c: (5 if c == GemColor.GOLD else 4) - env.state.players[0].gems[c] for c in GemColor}
    env.state.players[0].bonuses = {c: 4 for c in list(GemColor)[:5]}
    env.state.node_type = NodeType.OVERFLOW_DISCARD
    reference, native, evaluator = make_searches(env, simulations=80, min_simulations=8,
        check_interval=3, target_visits_per_action=0.0)
    root = None
    for _ in range(2):
        action, root = compare_search(env, reference, native, evaluator, root, True)
        child = next(c for c in root.children if c.action == action)
        previous = env.state.current_player
        env.step(action)
        native.advance(env.action_to_id(action))
        if child.state is None: child.state = env.state.clone()
        if env.state.current_player != previous: reference.flip_tree_values(child)
        child.parent = None
        root = child
        compare_tree(env, root, native)

    # A forced one-action noble choice uses just four new simulations.
    env = make_env(0)
    env.state.node_type = NodeType.NOBLE_CLAIM
    noble = env.state.nobles[0]
    env.state.nobles = [noble, None, None]
    env.state.players[0].bonuses = {c: 4 for c in list(GemColor)[:5]}
    reference, native, evaluator = make_searches(env, simulations=80)
    compare_search(env, reference, native, evaluator)
    assert native.last_search_metadata["actual_simulations"] == 4
    assert native.last_search_metadata["stop_reason"] == "single_legal_action"


@pytest.mark.parametrize("points,cards,winners", [([16, 15], [1, 1], [0]),
    ([15, 16], [1, 1], [1]), ([15, 15], [1, 1], [0, 1]), ([15, 15], [2, 1], [1])])
def test_terminal_backup(points, cards, winners):
    env = make_env(0)
    env.state.end_triggered = True
    for i in range(2):
        env.state.players[i].points = points[i]
        env.state.players[i].purchased_cards = BASE_TIER_1[:cards[i]]
    env.state.winners = winners
    reference, native, evaluator = make_searches(env, simulations=17, adaptive_simulations=False)
    compare_search(env, reference, native, evaluator)
    assert not evaluator.requests
    assert native.summary["value"] == (17 if winners == [0] else -17 if winners == [1] else 0)


def test_empty_root():
    env = make_env(0)
    # An inconsistent forced noble decision is supported as a dead-end reference fixture.
    env.state.node_type = NodeType.NOBLE_CLAIM
    reference, native, evaluator = make_searches(env, simulations=12)
    action, root = reference.search(env, env.state, return_root=True)
    native.begin()
    assert native.next_request() is None
    assert action is None and native.summary["best_action_id"] is None
    compare_tree(env, root, native)


def test_dead_end_leaf_backup():
    env = make_env(0)
    env.state.visible_cards = {tier: [None] * 4 for tier in [1, 2, 3]}
    env.state.decks = {tier: [] for tier in [1, 2, 3]}
    for i, gems in enumerate([[2, 2, 2, 2, 2, 0], [2, 2, 2, 2, 2, 0]]):
        env.state.players[i].gems = dict(zip(GemColor, gems))
    env.state.bank = {c: 5 if c == GemColor.GOLD else 0 for c in GemColor}
    env.state.players[1].reserved_cards = [c for c in BASE_TIER_3 if any(
        amount > env.state.players[1].gems[color] for color, amount in c.cost.items())][:3]
    env.state.players[1].reserved_card_hidden = [False] * 3
    env.state.players[0].bonuses = {c: 4 for c in list(GemColor)[:5]}
    env.state.nobles = [env.state.nobles[0], None, None]
    env.state.node_type = NodeType.NOBLE_CLAIM
    reference, native, evaluator = make_searches(env, simulations=12, adaptive_simulations=False)
    _, root = compare_search(env, reference, native, evaluator)
    assert len(evaluator.requests) == 1
    assert root.children[0].state.game_over
    assert root.children[0].state.winners == []
    assert root.children[0].value == 0.0
    action = root.children[0].action
    _, reward, terminated, _, _ = env.step(action)
    assert native.advance(env.action_to_id(action)) == (reward, terminated)
    assert not native.root_state().game_over
    assert not native.root_state().winners


@pytest.mark.parametrize("actor", [0, 1])
def test_final_round_search_waits_for_noble(actor):
    env = make_env(5)
    env.state.current_player = actor
    env.state.end_triggered = True
    env.state.players[actor].points = 15
    env.state.players[actor].bonuses = {c: 4 for c in list(GemColor)[:5]}
    env.state.node_type = NodeType.NOBLE_CLAIM
    reference, native, evaluator = make_searches(env, simulations=40, adaptive_simulations=False)
    compare_search(env, reference, native, evaluator)


def test_more_than_64_legal_candidates():
    env = make_env(1)
    gems = [1, 1, 1, 1, 1, 5]
    env.state.players[0].gems = dict(zip(GemColor, gems))
    env.state.bank = {c: (5 if c == GemColor.GOLD else 4) - env.state.players[0].gems[c] for c in GemColor}
    env.state.players[0].bonuses = {c: 2 for c in list(GemColor)[:5]}
    assert len(env._legal_actions(env.state)) > 64
    reference, native, evaluator = make_searches(env, simulations=400, adaptive_simulations=False)
    compare_search(env, reference, native, evaluator, noise=True)


def test_protocol_validation_and_retry():
    search = RustMCTS(reset(0), simulations=5)
    with pytest.raises(ValueError): search.respond([1.0], 0.0)
    search.begin()
    with pytest.raises(ValueError): search.begin()
    with pytest.raises(ValueError): search.advance(0)
    request = search.next_request()
    before = search.native.tree_json()
    for priors, value in [([1.0], 0.0), ([float("nan")] * len(request[1]), 0.0),
                          ([1.0] * len(request[1]), float("nan")), ([1.0] * len(request[1]), 2.0)]:
        with pytest.raises(ValueError): search.respond(priors, value)
        assert search.native.tree_json() == before
    search.respond(np.full(len(request[1]), 1.0 / len(request[1])), 0.1)
    while (request := search.next_request()) is not None:
        search.respond(np.full(len(request[1]), 1.0 / len(request[1])), 0.0)
    before = search.native.tree_json()
    with pytest.raises(ValueError): search.advance(65535)
    assert search.native.tree_json() == before


@pytest.mark.parametrize("bad", [dict(simulations=0), dict(check_interval=0), dict(c_puct=-1),
                                  dict(dirichlet_epsilon=2), dict(unknown_setting=True)])
def test_bad_config_rejected(bad):
    with pytest.raises(ValueError): RustMCTS(reset(0), **bad)


def test_batched_deterministic_search_matches_sequential():
    class ArraysEvaluator:
        def evaluate(self, obs, ids): return DeterministicEvaluator().evaluate_arrays(obs, ids)
        def evaluate_batch(self, requests): return [self.evaluate(*r) for r in requests]
    evaluator = ArraysEvaluator()
    sequential = [RustMCTS(reset(seed), evaluator=evaluator, simulations=70) for seed in range(4)]
    batched = [RustMCTS(reset(seed), simulations=70) for seed in range(4)]
    expected = [s.search() for s in sequential]
    for s in batched: s.begin()
    assert finish_searches(batched, evaluator) == expected
    for a, b in zip(sequential, batched):
        assert a.native.tree_json() == b.native.tree_json()


def test_temperature_sampling_matches_reference():
    from splendor_v1.training_v5.model_replay_generator_v5_pruning import ModelReplayGenerator
    env = make_env(1)
    reference, native, evaluator = make_searches(env, simulations=50, adaptive_simulations=False)
    _, root = compare_search(env, reference, native, evaluator)
    generator = ModelReplayGenerator.__new__(ModelReplayGenerator)
    generator.action_rng = np.random.default_rng(123)
    native.action_rng = np.random.default_rng(123)
    for temperature in [0.0, 1.0, 0.5, 1e-7] * 5:
        action = generator.select_action_from_root(root, temperature)
        assert native.select_action(temperature) == env.action_to_id(action)


@pytest.fixture(scope="module")
def checkpoint_model():
    from splendor_v1.network.model_4_legal_scorer import SplendorNetwork
    path = Path(__file__).parents[1] / "training_v6/data/model4_v6_inference_latest.pt"
    torch.set_num_threads(1)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = SplendorNetwork()
    weights = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    model.load_state_dict(weights)
    return model.eval()


def test_model4_direct_evaluator_and_search(checkpoint_model):
    model = checkpoint_model
    env = make_env(42)
    evaluator = DirectNeuralEvaluator(model)
    native_evaluator = Model4Evaluator(model)
    actions = env._legal_actions(env.state)
    expected, value = evaluator.evaluate(env, env.state, actions)
    actual, actual_value = native_evaluator.evaluate(env.observation_encoder.encoder(env.state),
        [env.action_to_id(a) for a in actions])
    np.testing.assert_array_equal(actual, expected.tolist())
    assert actual_value == value
    config = dict(simulations=35, adaptive_simulations=False)
    reference = MCTS(rollout_type="neural", selection_type="puct", evaluator=evaluator, **config)
    action, root = reference.search(env, env.state, return_root=True)
    native = RustMCTS(from_python(env.state), model=model, **config)
    assert native.search() == env.action_to_id(action)
    compare_tree(env, root, native)


def test_model4_padding_variable_legal_counts(checkpoint_model):
    evaluator = Model4Evaluator(checkpoint_model)
    requests = []
    for seed, moves in [(0, 0), (1, 7), (42, 13)]:
        state = reset(seed)
        for _ in range(moves): state.step(state.legal_action_ids()[0])
        requests.append((state.observation(), state.legal_action_ids()))
    assert len({len(ids) for _, ids in requests}) > 1
    actual = evaluator.evaluate_batch(requests)
    for request, (priors, value) in zip(requests, actual):
        expected_priors, expected_value = evaluator.evaluate(*request)
        assert len(priors) == len(request[1])
        np.testing.assert_allclose(priors, expected_priors, atol=2e-6, rtol=2e-5)
        assert value == pytest.approx(expected_value, abs=2e-6)
