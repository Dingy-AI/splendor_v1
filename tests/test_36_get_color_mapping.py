from splendor_v1.env.env import SplendorEnv


def get_all_cards(state):
    cards = []

    for tier in (1, 2, 3):
        cards.extend(
            state.visible_cards[tier]
        )

        cards.extend(
            state.decks[tier]
        )

    return cards


def test_get_color_mapping_matches_slow_version():

    env = SplendorEnv()
    env.reset()

    state = env.state

    cards = get_all_cards(state)

    for card in cards:

        slow_result = env.slow_get_color_mapping(
            card
        )

        fast_result = env.get_color_mapping(
            card
        )

        assert fast_result == slow_result


def test_get_color_mapping_matches_for_all_tiers():

    env = SplendorEnv()
    env.reset()

    state = env.state

    for tier in (1, 2, 3):

        cards = (
            state.visible_cards[tier]
            + state.decks[tier]
        )

        for card in cards:

            assert (
                env.get_color_mapping(card)
                ==
                env.slow_get_color_mapping(card)
            )


def test_get_color_mapping_uses_cache():

    env = SplendorEnv()
    env.reset()

    state = env.state

    card = state.visible_cards[1][0]

    first_result = env.get_color_mapping(
        card
    )

    second_result = env.get_color_mapping(
        card
    )

    assert first_result is second_result


def test_get_color_mapping_cache_keyed_by_card_id():

    env = SplendorEnv()
    env.reset()

    state = env.state

    card = state.visible_cards[1][0]

    result = env.get_color_mapping(
        card
    )

    assert card.id in env._color_mapping_cache

    assert (
        env._color_mapping_cache[card.id]
        is result
    )


def test_get_color_mapping_survives_state_clone():

    env = SplendorEnv()
    env.reset()

    state = env.state

    card = state.visible_cards[1][0]

    original_result = env.get_color_mapping(
        card
    )

    cloned_state = state.clone()

    cloned_card = cloned_state.visible_cards[1][0]

    cloned_result = env.get_color_mapping(
        cloned_card
    )

    assert cloned_result == original_result

    assert cloned_result is original_result


def test_get_color_mapping_cache_persists_after_reset():

    env = SplendorEnv()
    env.reset()

    state = env.state

    card = state.visible_cards[1][0]

    result = env.get_color_mapping(
        card
    )

    card_id = card.id

    assert card_id in env._color_mapping_cache

    env.reset()

    assert card_id in env._color_mapping_cache

    assert (
        env._color_mapping_cache[card_id]
        is result
    )