"""Native backend; rust_engine.run_training selects it for the existing V6 pipeline."""
import json

import numpy as np

from splendor_v1.env.core.enums import GemColor, NodeType
from splendor_v1.env.core.player import Player
from splendor_v1.env.data.data import BASE_TIER_1, BASE_TIER_2, BASE_TIER_3, NOBLES
from splendor_v1.env.state.base import GameState
from splendor_v1.training_v2.state_serializer import serialize_state


def from_python(state):
    """Import a reference state once; subsequent moves execute entirely in Rust."""
    from splendor_rust import RustState
    return RustState(json.dumps(serialize_state(state)))


def to_python(native):
    """Reconstruct the existing Python state for replay/UI/debug boundaries."""
    snapshot = json.loads(native.snapshot_json())
    cards = {c.id: c for c in BASE_TIER_1 + BASE_TIER_2 + BASE_TIER_3}
    nobles = {n.id: n for n in NOBLES}
    card_list = lambda ids: [None if i is None else cards[i] for i in ids]
    noble_list = lambda ids: [None if i is None else nobles[i] for i in ids]
    players = [Player(
        id=p["id"], points=p["points"],
        gems={c: p["gems"][c.name] for c in GemColor},
        bonuses={c: p["bonuses"][c.name] for c in list(GemColor)[:5]},
        reserved_cards=card_list(p["reserved_card_ids"]),
        reserved_card_hidden=p["reserved_card_hidden"],
        purchased_cards=card_list(p["purchased_card_ids"]),
        nobles=noble_list(p["noble_ids"]),
    ) for p in snapshot["players"]]
    return GameState(
        node_type=NodeType[snapshot["node_type"]], players=players,
        bank={c: snapshot["bank"][c.name] for c in GemColor},
        nobles=noble_list(snapshot["noble_ids"]),
        visible_cards={int(t): card_list(ids) for t, ids in snapshot["visible_card_ids"].items()},
        decks={int(t): card_list(ids) for t, ids in snapshot["deck_card_ids"].items()},
        current_player=snapshot["current_player"], turn_number=snapshot["turn_number"],
        winners=snapshot["winner_ids"], game_over=snapshot["game_over"],
        end_triggered=snapshot["end_triggered"], noble_taken=snapshot["noble_taken"],
    )


def observation(native):
    """A fresh float32 array matching Model 4's 258-feature input."""
    return np.asarray(native.observation(), dtype=np.float32)


def reset(seed=None):
    """Keep NumPy's existing shuffle/RNG behavior for exact seeded-start compatibility."""
    from splendor_v1.env.env import SplendorEnv
    env = SplendorEnv(num_players=2)
    env.reset(seed=seed)
    return from_python(env.state)
