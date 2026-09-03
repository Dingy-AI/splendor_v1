from splendor_v1.env.env import SplendorEnv
from splendor_v1.env.core.enums import GemColor


def get_all_cards(state):

    cards = []

    for tier_cards in state.visible_cards.values():
        for card in tier_cards:
            if card is not None:
                cards.append(card)

    for tier_cards in state.decks.values():
        for card in tier_cards:
            if card is not None:
                cards.append(card)

    return cards


def test_encode_card_matches_original():

    env = SplendorEnv()
    env.reset()

    state = env.state
    encoder = env.observation_encoder

    cards = get_all_cards(state)

    for card in cards:

        slow_encoding = encoder.slow_encode_card(
            card
        )

        cached_encoding = encoder._encode_card(
            card
        )

        assert slow_encoding == cached_encoding


def test_encode_none_matches_original():

    env = SplendorEnv()
    env.reset()

    encoder = env.observation_encoder

    slow_encoding = encoder.slow_encode_card(
        None
    )

    cached_encoding = encoder._encode_card(
        None
    )

    assert slow_encoding == cached_encoding


def test_encode_card_uses_cache():

    env = SplendorEnv()
    env.reset()

    state = env.state
    encoder = env.observation_encoder

    card = state.visible_cards[1][0]

    first_encoding = encoder._encode_card(
        card
    )

    second_encoding = encoder._encode_card(
        card
    )

    assert first_encoding is second_encoding


def test_encode_none_uses_cached_empty_encoding():

    env = SplendorEnv()
    env.reset()

    encoder = env.observation_encoder

    first_encoding = encoder._encode_card(
        None
    )

    second_encoding = encoder._encode_card(
        None
    )

    assert first_encoding is second_encoding


def test_encode_card_cache_keyed_by_card_id():

    env = SplendorEnv()
    env.reset()

    state = env.state
    encoder = env.observation_encoder

    card = state.visible_cards[1][0]

    encoding = encoder._encode_card(
        card
    )

    assert card.id in encoder._card_encoding_cache

    assert (
        encoder._card_encoding_cache[card.id]
        is encoding
    )


def test_encode_card_structure():

    env = SplendorEnv()
    env.reset()

    state = env.state
    encoder = env.observation_encoder

    card = state.visible_cards[1][0]

    encoding = encoder._encode_card(
        card
    )

    assert len(encoding) == 11

    expected_costs = [
        card.cost.get(color, 0)
        for color in GemColor
        if color != GemColor.GOLD
    ]

    expected_bonus = [
        1.0
        if card.bonus_color == color
        else 0.0
        for color in GemColor
        if color != GemColor.GOLD
    ]

    expected = (
        expected_costs
        + expected_bonus
        + [card.points]
    )

    assert encoding == expected