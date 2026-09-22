from splendor_v1.env.core.enums import GemColor


STATE_SCHEMA_VERSION = 1


# ============================================================
# ENUM / DICTIONARY HELPERS
# ============================================================

def _serialize_enum_key_dict(data):
    """
    Convert dictionaries such as:

        {
            GemColor.WHITE: 2,
            GemColor.BLUE: 1,
            ...
        }

    into stable plain-data dictionaries:

        {
            "WHITE": 2,
            "BLUE": 1,
            ...
        }

    Enum names are preferable to raw Enum objects because
    they don't require pickle to reconstruct the original
    Python enum class.
    """

    return {
        key.name: int(value)
        for key, value in data.items()
    }


def _serialize_card_ids(cards):
    """
    Preserve card identity, ordering,
    and empty slots.
    """

    return [
        (
            None
            if card is None
            else int(card.id)
        )
        for card in cards
    ]


def _serialize_noble_ids(nobles):
    """
    Preserve noble identity, ordering,
    and empty slots.
    """

    return [
        (
            None
            if noble is None
            else int(noble.id)
        )
        for noble in nobles
    ]

# ============================================================
# PLAYER
# ============================================================

def serialize_player(player):
    """
    Convert Player into plain replay-safe data.

    Card/Noble objects are represented by their stable IDs.
    """

    if (
        len(player.reserved_cards)
        != len(player.reserved_card_hidden)
    ):
        raise RuntimeError(
            "reserved_cards and reserved_card_hidden "
            "have different lengths."
        )


    return {

        "id":
            int(player.id),

        "points":
            int(player.points),

        # ----------------------------------------------------
        # Resources
        # ----------------------------------------------------

        "gems":
            _serialize_enum_key_dict(
                player.gems
            ),

        "bonuses":
            _serialize_enum_key_dict(
                player.bonuses
            ),

        # ----------------------------------------------------
        # Cards
        # ----------------------------------------------------

        # --------------------------------------------
        # RESERVED CARDS
        # --------------------------------------------

        "reserved_card_ids":
            _serialize_card_ids(
                player.reserved_cards
            ),

        # Parallel hidden-status information.
        #
        # Example:
        #
        # reserved_card_ids:
        #     [41, 63]
        #
        # reserved_card_hidden:
        #     [False, True]
        #
        "reserved_card_hidden":
            [
                bool(hidden)
                for hidden
                in player.reserved_card_hidden
            ],

        # --------------------------------------------
        # PURCHASED CARDS
        # --------------------------------------------

        "purchased_card_ids":
            _serialize_card_ids(
                player.purchased_cards
            ),

        # ----------------------------------------------------
        # Nobles
        # ----------------------------------------------------

        "noble_ids":
            _serialize_noble_ids(
                player.nobles
            ),
    }


# ============================================================
# GAME STATE
# ============================================================

def serialize_state(state):
    """
    Convert GameState into a stable plain-data snapshot.

    IMPORTANT:
    The full remaining deck ORDER is preserved.

    This means two states that look identical from the
    player's perspective but have different unseen deck
    compositions remain distinct in the replay archive.

    Hidden information is archived here for reconstruction,
    but does NOT need to be exposed to the neural-network
    observation encoder.
    """

    node_type = state.node_type

    if hasattr(node_type, "name"):
        node_type = node_type.name

    # --------------------------------------------------------
    # PLAYERS
    # --------------------------------------------------------

    players = [
        serialize_player(player)
        for player in state.players
    ]

    # --------------------------------------------------------
    # BANK
    # --------------------------------------------------------

    bank = _serialize_enum_key_dict(
        state.bank
    )

    # --------------------------------------------------------
    # VISIBLE CARDS
    #
    # Preserve tier and slot ordering.
    #
    # Example:
    #
    # {
    #     1: [12, 31, 4, 28],
    #     2: [...],
    #     3: [...]
    # }
    # --------------------------------------------------------

    visible_cards = {
        int(tier): _serialize_card_ids(
            cards
        )
        for tier, cards
        in state.visible_cards.items()
    }

    # --------------------------------------------------------
    # DECKS
    #
    # IMPORTANT:
    # We preserve EVERY remaining card ID IN ORDER.
    #
    # This makes the archived state fully distinguish
    # different hidden deck states.
    # --------------------------------------------------------

    decks = {
        int(tier): _serialize_card_ids(
            cards
        )
        for tier, cards
        in state.decks.items()
    }

    # --------------------------------------------------------
    # NOBLES
    # --------------------------------------------------------

    nobles = _serialize_noble_ids(
        state.nobles
    )

    # --------------------------------------------------------
    # FINAL STATE
    # --------------------------------------------------------

    return {

        "state_schema_version":
            STATE_SCHEMA_VERSION,

        "node_type":
            node_type,

        "players":
            players,

        "bank":
            bank,

        "noble_ids":
            nobles,

        "visible_card_ids":
            visible_cards,

        "deck_card_ids":
            decks,

        "current_player":
            int(state.current_player),

        "turn_number":
            int(state.turn_number),

        "winner_ids":
            [
                int(player_id)
                for player_id
                in state.winners
            ],

        "game_over":
            bool(state.game_over),

        "end_triggered":
            bool(state.end_triggered),

        "noble_taken":
            bool(state.noble_taken),
    }