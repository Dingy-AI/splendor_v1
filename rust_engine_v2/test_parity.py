"""Differential tests against the unchanged Python rules, not a second rules specification.

Run from the directory containing splendor_v1 after installing the native extension:
    python -m pytest splendor_v1/rust_engine_v2/test_parity.py -q
"""
import json
import random
from pathlib import Path

import numpy as np
import pytest

from splendor_v1.env.core import action_constants as ac
from splendor_v1.env.core.enums import ActionType, GemColor, NodeType
from splendor_v1.env.data.data import BASE_TIER_1
from splendor_v1.env.env import SplendorEnv
from splendor_v1.rust_engine_v2 import from_python, observation, reset, to_python
from splendor_v1.rust_engine_v2.generate_tables import render_tables
from splendor_v1.training_v2.state_serializer import serialize_state


def snapshot(native):
    data = json.loads(native.snapshot_json())
    for key in ("visible_card_ids", "deck_card_ids"):
        data[key] = {int(t): ids for t, ids in data[key].items()}
    return data


def assert_parity(env, native):
    assert snapshot(native) == serialize_state(env.state)
    actions = env._legal_actions(env.state)
    assert native.legal_action_ids() == [env.action_to_id(a) for a in actions]
    np.testing.assert_array_equal(observation(native), env.observation_encoder.encoder(env.state))
    return actions


def step_both(env, native, action):
    action_id = env.action_to_id(action)
    _, reward, terminated, truncated, _ = env.step(action)
    assert native.step(action_id) == (reward, terminated)
    assert not truncated
    assert_parity(env, native)


def make_env(seed=42):
    env = SplendorEnv()
    env.reset(seed=seed)
    return env


def give_gems(env, player, gems):
    for color, value in zip(GemColor, gems):
        env.state.players[player].gems[color] = value
        env.state.bank[color] = (5 if color == GemColor.GOLD else 4) - value - env.state.players[1 - player].gems[color]


def test_generated_tables_match_python_source():
    assert (Path(__file__).parent / "src" / "tables.rs").read_text() == render_tables()


def test_all_action_ids_round_trip_through_existing_mapping():
    import splendor_rust_v2
    env = make_env()
    assert splendor_rust_v2.ACTION_SPACE_SIZE == ac.ACTION_SPACE_SIZE == 1139
    assert splendor_rust_v2.OBSERVATION_SIZE == 258
    for action_id in range(ac.ACTION_SPACE_SIZE):
        assert env.action_to_id(env.id_to_action(action_id)) == action_id


@pytest.mark.parametrize("seed", [0, 1, 42, 420, 2**32])
def test_seeded_reset_and_roundtrip(seed):
    env = make_env(seed)
    native = reset(seed)
    assert_parity(env, native)
    rebuilt = to_python(native)
    assert serialize_state(rebuilt) == serialize_state(env.state)
    assert [env.action_to_id(a) for a in env._legal_actions(rebuilt)] == native.legal_action_ids()


@pytest.mark.parametrize("seed", range(32))
def test_seeded_game_every_transition(seed):
    env = make_env(seed)
    native = from_python(env.state)
    rng = random.Random(seed)
    for _ in range(500):
        actions = assert_parity(env, native)
        if env.state.game_over:
            break
        assert actions
        # Every sampled successor is checked; regular branch checks exercise cloning.
        if env.state.turn_number % 17 == 0:
            before = snapshot(native)
            for action in rng.sample(actions, min(3, len(actions))):
                branch = native.clone()
                child = env.state.clone()
                _, reward, terminated, _, _ = env.step(action, state=child)
                assert branch.step(env.action_to_id(action)) == (reward, terminated)
                assert snapshot(branch) == serialize_state(child)
            assert snapshot(native) == before
        step_both(env, native, rng.choice(actions))
    assert env.state.game_over, f"Seed {seed} did not finish within 500 decisions"


@pytest.mark.parametrize("gold", range(6))
def test_optional_gold_enumeration_and_every_buy_successor(gold):
    env = make_env()
    env.state.visible_cards[1][0] = next(c for c in BASE_TIER_1 if max(c.cost.values()) == 1)
    give_gems(env, 0, [1, 1, 1, 1, 1, gold])
    native = from_python(env.state)
    actions = assert_parity(env, native)
    buys = [a for a in actions if a.action_type == ActionType.BUY_VISIBLE]
    assert buys
    for action in buys:
        branch = native.clone()
        child = env.state.clone()
        _, reward, terminal, _, _ = env.step(action, state=child)
        assert branch.step(env.action_to_id(action)) == (reward, terminal)
        assert snapshot(branch) == serialize_state(child)


def test_overflow_then_noble_stays_same_turn_and_claims_once():
    env = make_env()
    give_gems(env, 0, [2, 2, 2, 2, 2, 0])
    env.state.players[0].bonuses = {c: 4 for c in list(GemColor)[:5]}
    native = from_python(env.state)
    triple = next(a for a in env._legal_actions(env.state)
                  if a.action_type == ActionType.TAKE_GEMS and len(a.gem_colors) == 3)
    step_both(env, native, triple)
    assert native.current_player == 0 and native.node_type == NodeType.OVERFLOW_DISCARD.value
    for _ in range(3):
        step_both(env, native, env._legal_actions(env.state)[0])
    assert native.current_player == 0 and native.node_type == NodeType.NOBLE_CLAIM.value
    step_both(env, native, env._legal_actions(env.state)[0])
    assert native.current_player == 1 and native.node_type == NodeType.MAIN_DECISION.value
    assert len(env.state.players[0].nobles) == 1


@pytest.mark.parametrize("actor", [0, 1])
def test_final_round_waits_for_forced_noble_choice(actor):
    env = make_env()
    env.state.current_player = actor
    env.state.players[actor].points = 15
    env.state.players[actor].bonuses = {c: 4 for c in list(GemColor)[:5]}
    env.state.end_triggered = True
    native = from_python(env.state)
    triple = next(a for a in env._legal_actions(env.state)
                  if a.action_type == ActionType.TAKE_GEMS and len(a.gem_colors) == 3)
    step_both(env, native, triple)
    assert not native.game_over
    assert native.current_player == actor
    step_both(env, native, env._legal_actions(env.state)[0])
    assert native.game_over == (actor == 1)


def test_hidden_reserve_mask_and_ownership():
    env = make_env()
    action = next(a for a in env._legal_actions(env.state) if a.action_type == ActionType.RESERVE_TOP_DECK)
    native = from_python(env.state)
    before = observation(native)
    expected_before = env.observation_encoder.encoder(env.state)
    step_both(env, native, action)
    np.testing.assert_array_equal(before, expected_before)
    # Opponent is player 0: 48 features per player, then 6+5 gems/bonuses.
    hidden_slot = observation(native)[59:71]
    np.testing.assert_array_equal(hidden_slot, [0] * 11 + [1])
    # Card identity is retained in the full snapshot for replay reconstruction.
    assert snapshot(native)["players"][0]["reserved_card_ids"]


def test_empty_deck_replacements_and_multiple_hidden_reserves():
    env = make_env()
    env.state.decks[1] = []
    native = from_python(env.state)
    action = next(a for a in env._legal_actions(env.state)
                  if a.action_type == ActionType.RESERVE_VISIBLE and a.tier == 1)
    step_both(env, native, action)
    assert env.state.visible_cards[1][action.slot] is None
    for _ in range(4):
        action = next(a for a in env._legal_actions(env.state) if a.action_type == ActionType.RESERVE_TOP_DECK)
        step_both(env, native, action)


@pytest.mark.parametrize("cards0,cards1,expected", [(1, 2, [0]), (2, 1, [1]), (1, 1, [0, 1])])
def test_winner_tiebreak(cards0, cards1, expected):
    env = make_env()
    env.state.current_player = 1
    env.state.end_triggered = True
    for index, count in enumerate([cards0, cards1]):
        env.state.players[index].points = 15
        env.state.players[index].purchased_cards = BASE_TIER_1[:count]
    native = from_python(env.state)
    action = next(a for a in env._legal_actions(env.state) if a.action_type == ActionType.TAKE_GEMS)
    step_both(env, native, action)
    assert native.winners == expected


@pytest.mark.parametrize("id", [0, 45, ac.DISCARD_START, ac.ACTION_SPACE_SIZE, 65536])
def test_invalid_action_is_rejected_without_mutating(id):
    native = reset(42)
    assert id not in native.legal_action_ids()
    before = native.snapshot_json()
    with pytest.raises(ValueError):
        native.step(id)
    assert native.snapshot_json() == before


def test_invalid_imports_are_rejected():
    from splendor_rust_v2 import RustState
    data = json.loads(reset(42).snapshot_json())
    for field, value in [("current_player", 2), ("node_type", "UNKNOWN"),
                         ("state_schema_version", 2), ("players", []),
                         ("noble_ids", [10, 0, 1])]:
        bad = dict(data, **{field: value})
        with pytest.raises(ValueError):
            RustState(json.dumps(bad))
    bad = json.loads(json.dumps(data))
    bad["deck_card_ids"]["1"][0] = 100
    with pytest.raises(ValueError):
        RustState(json.dumps(bad))
