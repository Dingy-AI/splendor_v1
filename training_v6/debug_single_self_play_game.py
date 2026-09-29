"""
Reproduce one Model 4 / MCTS V5 self-play game with verbose state diagnostics.

Purpose
-------
Re-run a known failing self-play seed using the SAME ModelReplayGenerator
game loop and MCTS configuration as V6 production, but with a direct
single-process neural evaluator.

Default failing case:
    checkpoint: splendor_v1/training_v6/data/model_5440_games.pt
    seed:       1005464

This script does NOT modify the production replay buffer.

Run from repository root:

    python -m splendor_v1.training_v6.debug_single_self_play_game

or, if saved elsewhere:

    python path/to/debug_single_self_play_game.py

Useful overrides:

    python -m splendor_v1.training_v6.debug_single_self_play_game ^
        --checkpoint splendor_v1/training_v6/data/model_5440_games.pt ^
        --seed 1005464 ^
        --device cuda
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pprint
import traceback

import numpy as np
import torch

from splendor_v1.env.env import SplendorEnv

from splendor_v1.network.model_4_legal_scorer import (
    SplendorNetwork,
)

from splendor_v1.training_v2.replay_buffer import (
    ReplayBuffer,
)

from splendor_v1.training_v2.state_serializer import (
    serialize_state,
)

from splendor_v1.training_v5.model_replay_generator_v5_pruning import (
    ModelReplayGenerator,
)

from splendor_v1.mcts_batched.direct_neural_evaluator import (
    DirectNeuralEvaluator,
)

from splendor_v1.mcts_batched.mcts_v5_direct import (
    MCTS,
)


# ============================================================
# DEFAULTS
# ============================================================

DEFAULT_CHECKPOINT = (
    "splendor_v1/training_v6/data/"
    "model_5344_games.pt"
)

DEFAULT_SEED = 1005464

SIMULATIONS = 400
MIN_SIMULATIONS = 80
CHECK_INTERVAL = 20
TARGET_VISITS_PER_ACTION = 20.0
SINGLE_ACTION_SIMULATIONS = 4
STABILITY_CHECKS = 3

C_PUCT = 3.0
DIRICHLET_ALPHA = 0.3
DIRICHLET_EPSILON = 0.25

MAX_GAME_STEPS = 300


# ============================================================
# SMALL DISPLAY HELPERS
# ============================================================

def enum_name(value):
    if hasattr(value, "name"):
        return value.name
    return str(value)


def simple_mapping(mapping):
    if mapping is None:
        return None

    result = {}

    try:
        items = mapping.items()
    except Exception:
        return repr(mapping)

    for key, value in items:
        result[enum_name(key)] = value

    return result


def card_summary(card):
    if card is None:
        return None

    result = {
        "type": type(card).__name__,
    }

    for field in (
        "id",
        "card_id",
        "tier",
        "points",
        "prestige",
        "bonus",
        "color",
    ):
        if hasattr(card, field):
            value = getattr(card, field)
            result[field] = (
                enum_name(value)
                if hasattr(value, "name")
                else value
            )

    if hasattr(card, "cost"):
        result["cost"] = simple_mapping(
            getattr(card, "cost")
        )

    if len(result) == 1:
        result["repr"] = repr(card)

    return result


def noble_summary(noble):
    if noble is None:
        return None

    result = {
        "type": type(noble).__name__,
    }

    for field in (
        "id",
        "noble_id",
        "points",
    ):
        if hasattr(noble, field):
            result[field] = getattr(
                noble,
                field,
            )

    if hasattr(noble, "requirement"):
        result["requirement"] = simple_mapping(
            getattr(
                noble,
                "requirement",
            )
        )

    if len(result) == 1:
        result["repr"] = repr(noble)

    return result


def action_summary(env, action):
    if action is None:
        return None

    result = {}

    try:
        result["action_id"] = int(
            env.action_to_id(
                action
            )
        )
    except Exception as exc:
        result["action_id_error"] = repr(
            exc
        )

    action_type = getattr(
        action,
        "action_type",
        None,
    )

    result["action_type"] = (
        enum_name(action_type)
        if action_type is not None
        else None
    )

    for field in (
        "tier",
        "slot",
        "reserved_index",
        "payment_id",
        "noble_index",
        "gem_colors",
        "gold_payment",
    ):
        if not hasattr(action, field):
            continue

        value = getattr(
            action,
            field,
        )

        if isinstance(
            value,
            (tuple, list),
        ):
            value = [
                enum_name(item)
                for item in value
            ]
        elif hasattr(value, "name"):
            value = value.name

        result[field] = value

    return result


def safe_count(callable_obj):
    try:
        result = callable_obj()
        return len(result)
    except Exception as exc:
        return (
            f"ERROR: {type(exc).__name__}: "
            f"{exc}"
        )


# ============================================================
# STATE DIAGNOSTICS
# ============================================================

def legal_action_breakdown(env, state):
    """
    Show both the actual node-specific legal action count and
    category counts that are useful for diagnosing MAIN_DECISION
    dead ends.
    """

    result = {}

    try:
        actual = env._legal_actions(
            state
        )
        result["actual_node_legal"] = len(
            actual
        )
        result["actual_action_ids"] = [
            int(env.action_to_id(action))
            for action in actual
        ]
    except Exception as exc:
        result[
            "actual_node_legal_error"
        ] = (
            f"{type(exc).__name__}: "
            f"{exc}"
        )

    category_methods = {
        "buy_visible":
            "_legal_buy_visible",
        "buy_reserved":
            "_legal_buy_reserved",
        "reserve_visible":
            "_legal_reserve_visible",
        "reserve_top_deck":
            "_legal_reserve_top_deck",
        "take_gems":
            "_legal_take_gems",
        "discard_gems":
            "_legal_discard_actions",
        "take_noble":
            "_legal_noble_actions",
    }

    for label, method_name in (
        category_methods.items()
    ):
        method = getattr(
            env,
            method_name,
            None,
        )

        if method is None:
            result[label] = (
                f"MISSING METHOD: "
                f"{method_name}"
            )
            continue

        result[label] = safe_count(
            lambda method=method:
                method(state)
        )

    return result


def compact_state_summary(
    env,
    state,
    step_index,
):
    players = []

    for player_index, player in enumerate(
        state.players
    ):
        players.append(
            {
                "player":
                    player_index,
                "points":
                    getattr(
                        player,
                        "points",
                        None,
                    ),
                "gems":
                    simple_mapping(
                        getattr(
                            player,
                            "gems",
                            None,
                        )
                    ),
                "bonuses":
                    simple_mapping(
                        getattr(
                            player,
                            "bonuses",
                            None,
                        )
                    ),
                "reserved":
                    len(
                        getattr(
                            player,
                            "reserved_cards",
                            [],
                        )
                    ),
                "purchased":
                    len(
                        getattr(
                            player,
                            "purchased_cards",
                            [],
                        )
                    ),
                "nobles":
                    len(
                        getattr(
                            player,
                            "nobles",
                            [],
                        )
                    ),
            }
        )

    try:
        legal_count = len(
            env._legal_actions(
                state
            )
        )
    except Exception:
        legal_count = None

    return {
        "step_index":
            int(step_index),
        "turn_number":
            int(
                getattr(
                    state,
                    "turn_number",
                    -1,
                )
            ),
        "current_player":
            int(
                getattr(
                    state,
                    "current_player",
                    -1,
                )
            ),
        "node_type":
            enum_name(
                getattr(
                    state,
                    "node_type",
                    None,
                )
            ),
        "game_over":
            bool(
                getattr(
                    state,
                    "game_over",
                    False,
                )
            ),
        "end_triggered":
            bool(
                getattr(
                    state,
                    "end_triggered",
                    False,
                )
            ),
        "noble_taken":
            bool(
                getattr(
                    state,
                    "noble_taken",
                    False,
                )
            ),
        "legal_count":
            legal_count,
        "bank":
            simple_mapping(
                getattr(
                    state,
                    "bank",
                    None,
                )
            ),
        "players":
            players,
    }


def print_compact_state(
    env,
    state,
    step_index,
):
    summary = compact_state_summary(
        env,
        state,
        step_index,
    )

    print()
    print("-" * 78)
    print(
        "STATE BEFORE DECISION | "
        f"step={summary['step_index']} | "
        f"turn={summary['turn_number']} | "
        f"player={summary['current_player']} | "
        f"node={summary['node_type']} | "
        f"legal={summary['legal_count']} | "
        f"end_triggered="
        f"{summary['end_triggered']} | "
        f"game_over="
        f"{summary['game_over']}"
    )

    print(
        "Bank:",
        summary["bank"],
    )

    for player in summary["players"]:
        print(
            f"P{player['player']}: "
            f"points={player['points']} "
            f"gems={player['gems']} "
            f"bonuses={player['bonuses']} "
            f"reserved={player['reserved']} "
            f"purchased={player['purchased']} "
            f"nobles={player['nobles']}"
        )


def deep_state_dump(
    env,
    state,
    *,
    step_index,
    reason,
):
    print()
    print("=" * 78)
    print("DEEP STATE DUMP")
    print("=" * 78)
    print("Reason:", reason)
    print(
        "step_index:",
        step_index,
    )
    print(
        "turn_number:",
        getattr(
            state,
            "turn_number",
            None,
        ),
    )
    print(
        "current_player:",
        getattr(
            state,
            "current_player",
            None,
        ),
    )
    print(
        "node_type:",
        enum_name(
            getattr(
                state,
                "node_type",
                None,
            )
        ),
    )
    print(
        "game_over:",
        getattr(
            state,
            "game_over",
            None,
        ),
    )
    print(
        "end_triggered:",
        getattr(
            state,
            "end_triggered",
            None,
        ),
    )
    print(
        "noble_taken:",
        getattr(
            state,
            "noble_taken",
            None,
        ),
    )

    try:
        check_terminated = (
            env._check_terminated(
                state
            )
        )
    except Exception as exc:
        check_terminated = (
            f"ERROR: "
            f"{type(exc).__name__}: "
            f"{exc}"
        )

    print(
        "_check_terminated(state):",
        check_terminated,
    )

    check_overflow = getattr(
        env,
        "_check_overflow",
        None,
    )

    if check_overflow is not None:
        try:
            value = check_overflow(
                state
            )
        except Exception as exc:
            value = (
                f"ERROR: "
                f"{type(exc).__name__}: "
                f"{exc}"
            )

        print(
            "_check_overflow(state):",
            value,
        )

    check_nobles = getattr(
        env,
        "_check_nobles",
        None,
    )

    if check_nobles is not None:
        try:
            value = check_nobles(
                state
            )
        except Exception as exc:
            value = (
                f"ERROR: "
                f"{type(exc).__name__}: "
                f"{exc}"
            )

        print(
            "_check_nobles(state):",
            value,
        )

    print()
    print("LEGAL ACTION BREAKDOWN")
    print(
        pprint.pformat(
            legal_action_breakdown(
                env,
                state,
            ),
            width=120,
            sort_dicts=False,
        )
    )

    print()
    print("PLAYERS")

    for player_index, player in enumerate(
        state.players
    ):
        print()
        print(
            f"Player {player_index}"
        )
        print(
            "  points:",
            getattr(
                player,
                "points",
                None,
            ),
        )
        print(
            "  gems:",
            simple_mapping(
                getattr(
                    player,
                    "gems",
                    None,
                )
            ),
        )
        print(
            "  bonuses:",
            simple_mapping(
                getattr(
                    player,
                    "bonuses",
                    None,
                )
            ),
        )

        reserved = getattr(
            player,
            "reserved_cards",
            [],
        )

        purchased = getattr(
            player,
            "purchased_cards",
            [],
        )

        player_nobles = getattr(
            player,
            "nobles",
            [],
        )

        print(
            "  reserved cards:",
            pprint.pformat(
                [
                    card_summary(card)
                    for card in reserved
                ],
                width=120,
                sort_dicts=False,
            ),
        )

        print(
            "  purchased count:",
            len(purchased),
        )

        print(
            "  purchased cards:",
            pprint.pformat(
                [
                    card_summary(card)
                    for card in purchased
                ],
                width=120,
                sort_dicts=False,
            ),
        )

        print(
            "  nobles:",
            pprint.pformat(
                [
                    noble_summary(noble)
                    for noble in player_nobles
                ],
                width=120,
                sort_dicts=False,
            ),
        )

    print()
    print("BANK")
    print(
        pprint.pformat(
            simple_mapping(
                getattr(
                    state,
                    "bank",
                    None,
                )
            ),
            width=120,
            sort_dicts=False,
        )
    )

    print()
    print("VISIBLE CARDS")

    visible_cards = getattr(
        state,
        "visible_cards",
        {},
    )

    try:
        visible_items = (
            visible_cards.items()
        )
    except Exception:
        visible_items = []

    for tier, cards in visible_items:
        print(
            f"Tier {tier}:",
            pprint.pformat(
                [
                    card_summary(card)
                    for card in cards
                ],
                width=120,
                sort_dicts=False,
            ),
        )

    print()
    print("DECK SIZES")

    decks = getattr(
        state,
        "decks",
        {},
    )

    try:
        deck_sizes = {
            tier: len(deck)
            for tier, deck
            in decks.items()
        }
    except Exception:
        deck_sizes = repr(
            decks
        )

    print(
        pprint.pformat(
            deck_sizes,
            width=120,
            sort_dicts=False,
        )
    )

    print()
    print("AVAILABLE NOBLES")

    nobles = getattr(
        state,
        "nobles",
        [],
    )

    print(
        pprint.pformat(
            [
                noble_summary(noble)
                for noble in nobles
            ],
            width=120,
            sort_dicts=False,
        )
    )

    print()
    print("SERIALIZED STATE")
    try:
        serialized = serialize_state(
            state
        )
        print(
            pprint.pformat(
                serialized,
                width=140,
                sort_dicts=False,
            )
        )
    except Exception:
        traceback.print_exc()

    print("=" * 78)
    print("END DEEP STATE DUMP")
    print("=" * 78)


# ============================================================
# DEBUG GENERATOR
# ============================================================

class DebugModelReplayGenerator(
    ModelReplayGenerator
):
    """
    Same production generator, with print-only instrumentation.
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(
            *args,
            **kwargs,
        )

        self.debug_step_index = None
        self.last_selected_action = None

    def build_base_sample(
        self,
        state,
        step_index,
    ):
        self.debug_step_index = int(
            step_index
        )

        print_compact_state(
            self.env,
            state,
            step_index,
        )

        try:
            return super().build_base_sample(
                state=state,
                step_index=step_index,
            )

        except BaseException as exc:
            deep_state_dump(
                self.env,
                state,
                step_index=step_index,
                reason=(
                    "build_base_sample failed: "
                    f"{type(exc).__name__}: "
                    f"{exc}"
                ),
            )
            raise

    def select_action_from_root(
        self,
        root,
        temperature,
        fallback_action=None,
    ):
        action = super().select_action_from_root(
            root=root,
            temperature=temperature,
            fallback_action=(
                fallback_action
            ),
        )

        self.last_selected_action = action

        print(
            "Chosen action:",
            pprint.pformat(
                action_summary(
                    self.env,
                    action,
                ),
                width=120,
                sort_dicts=False,
            ),
        )

        print(
            "Temperature:",
            temperature,
        )

        children = list(
            getattr(
                root,
                "children",
                [],
            )
        )

        ranked = sorted(
            children,
            key=lambda child:
                getattr(
                    child,
                    "visits",
                    0,
                ),
            reverse=True,
        )

        top = []

        for child in ranked[:5]:
            top.append(
                {
                    "visits":
                        int(
                            getattr(
                                child,
                                "visits",
                                0,
                            )
                        ),
                    "prior":
                        float(
                            getattr(
                                child,
                                "prior",
                                0.0,
                            )
                        ),
                    "action":
                        action_summary(
                            self.env,
                            getattr(
                                child,
                                "action",
                                None,
                            ),
                        ),
                }
            )

        print(
            "Top root children:",
            pprint.pformat(
                top,
                width=120,
                sort_dicts=False,
            ),
        )

        metadata = getattr(
            self.mcts,
            "last_search_metadata",
            None,
        )

        if metadata:
            print(
                "Search metadata:",
                pprint.pformat(
                    metadata,
                    width=120,
                    sort_dicts=False,
                ),
            )

        return action

    def step(
        self,
        action,
    ):
        pre_state = self.env.state

        try:
            result = super().step(
                action
            )

        except BaseException as exc:
            deep_state_dump(
                self.env,
                pre_state,
                step_index=(
                    self.debug_step_index
                ),
                reason=(
                    "env.step failed after action "
                    f"{action_summary(self.env, action)}: "
                    f"{type(exc).__name__}: {exc}"
                ),
            )
            raise

        reward, terminated, info = result

        print(
            "Transition result:",
            {
                "reward":
                    reward,
                "terminated":
                    terminated,
                "next_turn":
                    getattr(
                        self.env.state,
                        "turn_number",
                        None,
                    ),
                "next_player":
                    getattr(
                        self.env.state,
                        "current_player",
                        None,
                    ),
                "next_node":
                    enum_name(
                        getattr(
                            self.env.state,
                            "node_type",
                            None,
                        )
                    ),
                "end_triggered":
                    getattr(
                        self.env.state,
                        "end_triggered",
                        None,
                    ),
                "game_over":
                    getattr(
                        self.env.state,
                        "game_over",
                        None,
                    ),
            },
        )

        return result


# ============================================================
# MODEL / MCTS
# ============================================================

def extract_model_state_dict(
    checkpoint,
):
    if (
        isinstance(
            checkpoint,
            dict,
        )
        and "model_state_dict"
        in checkpoint
    ):
        return checkpoint[
            "model_state_dict"
        ]

    if (
        isinstance(
            checkpoint,
            dict,
        )
        and checkpoint
        and all(
            torch.is_tensor(
                value
            )
            for value
            in checkpoint.values()
        )
    ):
        return checkpoint

    raise RuntimeError(
        "Checkpoint must be a raw "
        "state_dict or contain "
        "'model_state_dict'."
    )


def load_model(
    checkpoint_path,
    device,
):
    checkpoint_path = Path(
        checkpoint_path
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            checkpoint_path
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    model = SplendorNetwork()

    model.load_state_dict(
        extract_model_state_dict(
            checkpoint
        ),
        strict=True,
    )

    model.to(
        device
    )

    model.eval()

    return model


def make_mcts(
    model,
):
    evaluator = DirectNeuralEvaluator(
        model=model
    )

    return MCTS(
        simulations=SIMULATIONS,
        rollout_type="neural",
        selection_type="puct",
        evaluator=evaluator,
        c_puct=C_PUCT,
        dirichlet_alpha=(
            DIRICHLET_ALPHA
        ),
        dirichlet_epsilon=(
            DIRICHLET_EPSILON
        ),
        adaptive_simulations=True,
        min_simulations=(
            MIN_SIMULATIONS
        ),
        check_interval=(
            CHECK_INTERVAL
        ),
        target_visits_per_action=(
            TARGET_VISITS_PER_ACTION
        ),
        single_action_simulations=(
            SINGLE_ACTION_SIMULATIONS
        ),
        stability_checks=(
            STABILITY_CHECKS
        ),
    )


def make_generator(
    env,
    mcts,
    replay_buffer,
):
    return DebugModelReplayGenerator(
        env=env,
        mcts=mcts,
        replay_buffer=replay_buffer,
        state_serializer=serialize_state,

        temperature=1.0,

        temperature_fn=lambda turn: (
            1.0
            if turn < 20
            else (
                0.5
                if turn < 40
                else 0.0
            )
        ),

        add_root_noise=True,

        root_noise_fn=lambda turn: (
            turn < 40
        ),

        teacher_mode=False,
        action_space_version=1,
        max_game_steps=(
            MAX_GAME_STEPS
        ),
    )


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Re-run one Splendor Model 4 "
            "self-play seed with verbose "
            "game-state diagnostics."
        )
    )

    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )

    parser.add_argument(
        "--device",
        default=(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        ),
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()

    device = torch.device(
        args.device
    )

    if (
        device.type == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA requested but "
            "torch.cuda.is_available() is False."
        )

    print("=" * 78)
    print("SINGLE SELF-PLAY GAME DEBUGGER")
    print("=" * 78)
    print(
        "Checkpoint:",
        args.checkpoint,
    )
    print(
        "Seed:",
        args.seed,
    )
    print(
        "Device:",
        device,
    )
    print(
        "Simulations:",
        SIMULATIONS,
    )
    print(
        "Adaptive search:",
        True,
    )
    print()

    print("Loading Model 4...")

    model = load_model(
        checkpoint_path=(
            args.checkpoint
        ),
        device=device,
    )

    print("Model loaded.")

    env = SplendorEnv()

    mcts = make_mcts(
        model
    )

    replay_buffer = ReplayBuffer(
        capacity=10_000
    )

    generator = make_generator(
        env=env,
        mcts=mcts,
        replay_buffer=replay_buffer,
    )

    print()
    print(
        "Running deterministic seed...",
        args.seed,
    )

    try:
        result = generator.generate_game(
            seed=int(
                args.seed
            ),
            split="debug",
            model_generation=4,
            model_checkpoint=str(
                args.checkpoint
            ),
            extra_game_metadata={
                "debug_single_game":
                    True,
            },
        )

    except BaseException as exc:
        print()
        print("#" * 78)
        print("GAME FAILED")
        print("#" * 78)
        print(
            "Exception:",
            f"{type(exc).__name__}: "
            f"{exc}",
        )
        print()
        traceback.print_exc()

        # One final dump in case the exception happened somewhere
        # outside build_base_sample()/step().
        try:
            deep_state_dump(
                env,
                env.state,
                step_index=(
                    generator.debug_step_index
                ),
                reason=(
                    "final failure dump"
                ),
            )
        except Exception:
            traceback.print_exc()

        raise

    print()
    print("=" * 78)
    print("GAME COMPLETED SUCCESSFULLY")
    print("=" * 78)
    print(
        pprint.pformat(
            result,
            width=120,
            sort_dicts=False,
        )
    )

    print()
    print(
        "Final winners:",
        getattr(
            env.state,
            "winners",
            None,
        ),
    )

    print(
        "Final scores:",
        [
            getattr(
                player,
                "points",
                None,
            )
            for player
            in env.state.players
        ],
    )


if __name__ == "__main__":
    main()
