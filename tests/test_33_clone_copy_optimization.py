from splendor_v1.env.core.enums import NodeType, GemColor
from splendor_v1.env.core.player import Player
from splendor_v1.env.env import GameState
import pytest

@pytest.fixture
def sample_player():
    reserved_card = object()
    purchased_card = object()
    noble = object()

    return Player(
        id=0,

        gems={
            color: 0
            for color in GemColor
        },

        bonuses={
            color: 0
            for color in GemColor
            if color != GemColor.GOLD
        },

        reserved_cards=[
            reserved_card,
        ],

        purchased_cards=[
            purchased_card,
        ],

        nobles=[
            noble,
        ],

        points=3,
    )

@pytest.fixture
def sample_game_state():
    player_0 = Player(
        id=0,
        gems={
            color: 0
            for color in GemColor
        },
        bonuses={
            color: 0
            for color in GemColor
            if color != GemColor.GOLD
            
        },
        reserved_cards=[
            object(),
        ],
        purchased_cards=[
            object(),
        ],
        nobles=[
            object(),
        ],
        points=3,
    )

    player_1 = Player(
        id=1,
        gems={
            color: 0
            for color in GemColor
        },
        bonuses={
            color: 0
            for color in GemColor
            if color != GemColor.GOLD

        },
        reserved_cards=[
            object(),
        ],
        purchased_cards=[
            object(),
        ],
        nobles=[
            object(),
        ],
        points=5,
    )

    return GameState(
        node_type=next(iter(NodeType)),

        players=[
            player_0,
            player_1,
        ],

        bank={
            color: (
            5
            if color == GemColor.GOLD
            else 4
            )
            for color in GemColor

        },

        nobles=[
            object(),
            object(),
        ],

        visible_cards={
            1: [object() for _ in range(4)],
            2: [object() for _ in range(4)],
            3: [object() for _ in range(4)],
        },

        
        decks={
            1: [object(), object(), object()],
            2: [object(), object(), object()],
            3: [object(), object(), object()],
        },

        current_player=0,
        turn_number=5,

        winners=[],

        game_over=False,
        end_triggered=False,
        noble_taken=False,
    )

def test_player_clone_creates_independent_containers(
    sample_player,
):
    clone = sample_player.clone()

    assert clone is not sample_player

    assert clone.gems is not sample_player.gems
    assert clone.bonuses is not sample_player.bonuses

    assert (
        clone.reserved_cards
        is not sample_player.reserved_cards
    )

    assert (
        clone.purchased_cards
        is not sample_player.purchased_cards
    )

    assert (
        clone.nobles
        is not sample_player.nobles
    )


def test_player_clone_preserves_values(
    sample_player,
):
    clone = sample_player.clone()

    assert clone.id == sample_player.id
    assert clone.gems == sample_player.gems
    assert clone.bonuses == sample_player.bonuses

    assert (
        clone.reserved_cards
        == sample_player.reserved_cards
    )

    assert (
        clone.purchased_cards
        == sample_player.purchased_cards
    )

    assert clone.nobles == sample_player.nobles
    assert clone.points == sample_player.points


def test_player_clone_mutation_does_not_affect_original(
    sample_player,
):
    clone = sample_player.clone()

    clone.gems[GemColor.WHITE] += 1
    clone.bonuses[GemColor.BLUE] += 1
    clone.points += 3

    clone.reserved_cards.clear()
    clone.purchased_cards.clear()
    clone.nobles.clear()

    assert (
        clone.gems[GemColor.WHITE]
        != sample_player.gems[GemColor.WHITE]
    )

    assert (
        clone.bonuses[GemColor.BLUE]
        != sample_player.bonuses[GemColor.BLUE]
    )

    assert clone.points != sample_player.points

    assert sample_player.reserved_cards
    assert sample_player.purchased_cards
    assert sample_player.nobles

def test_player_clone_shares_card_and_noble_objects(
    sample_player,
):
    clone = sample_player.clone()

    if sample_player.reserved_cards:
        assert (
            clone.reserved_cards[0]
            is sample_player.reserved_cards[0]
        )

    if sample_player.purchased_cards:
        assert (
            clone.purchased_cards[0]
            is sample_player.purchased_cards[0]
        )

    if sample_player.nobles:
        assert (
            clone.nobles[0]
            is sample_player.nobles[0]
        )

def test_game_state_clone_creates_independent_containers(
    sample_game_state,
):
    clone = sample_game_state.clone()

    assert clone is not sample_game_state

    assert clone.players is not sample_game_state.players
    assert clone.bank is not sample_game_state.bank
    assert clone.nobles is not sample_game_state.nobles

    assert (
        clone.visible_cards
        is not sample_game_state.visible_cards
    )

    assert clone.decks is not sample_game_state.decks
    assert clone.winners is not sample_game_state.winners

def test_game_state_clone_clones_players(
    sample_game_state,
):
    clone = sample_game_state.clone()

    for original_player, cloned_player in zip(
        sample_game_state.players,
        clone.players,
    ):
        assert cloned_player is not original_player

        assert (
            cloned_player.gems
            is not original_player.gems
        )

def test_game_state_clone_copies_card_containers(
    sample_game_state,
):
    clone = sample_game_state.clone()

    for tier in sample_game_state.visible_cards:
        assert (
            clone.visible_cards[tier]
            is not sample_game_state.visible_cards[tier]
        )

    for tier in sample_game_state.decks:
        assert (
            clone.decks[tier]
            is not sample_game_state.decks[tier]
        )

def test_game_state_clone_mutation_does_not_affect_original(
    sample_game_state,
):
    clone = sample_game_state.clone()

    clone.bank[GemColor.WHITE] += 1

    clone.players[0].gems[
        GemColor.BLUE
    ] += 1

    clone.winners.append(99)

    assert (
        clone.bank[GemColor.WHITE]
        != sample_game_state.bank[
            GemColor.WHITE
        ]
    )

    assert (
        clone.players[0].gems[
            GemColor.BLUE
        ]
        != sample_game_state.players[0].gems[
            GemColor.BLUE
        ]
    )

    assert 99 not in sample_game_state.winners

def test_game_state_clone_card_list_mutation_is_independent(
    sample_game_state,
):
    clone = sample_game_state.clone()

    tier = next(iter(sample_game_state.visible_cards))

    original_length = len(
        sample_game_state.visible_cards[tier]
    )

    clone.visible_cards[tier].pop()

    assert (
        len(sample_game_state.visible_cards[tier])
        == original_length
    )

def test_game_state_clone_shares_card_and_noble_objects(
    sample_game_state,
):
    clone = sample_game_state.clone()

    for tier, cards in sample_game_state.visible_cards.items():
        if cards:
            assert (
                clone.visible_cards[tier][0]
                is cards[0]
            )

    for tier, cards in sample_game_state.decks.items():
        if cards:
            assert (
                clone.decks[tier][0]
                is cards[0]
            )

    if sample_game_state.nobles:
        assert (
            clone.nobles[0]
            is sample_game_state.nobles[0]
        )