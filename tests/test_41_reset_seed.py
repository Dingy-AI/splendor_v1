from splendor_v1.env.env import SplendorEnv

def test_reset_same_seed_generates_same_game():
    env = SplendorEnv(num_players=2)

    env.reset(seed=420)
    state_1 = env.state

    # Save the randomized components before resetting again
    visible_cards_1 = {
        tier: [card.id for card in cards]
        for tier, cards in env.state.visible_cards.items()
    }
        
    decks_1 = {
        tier: [card.id for card in cards]
        for tier, cards in env.state.decks.items()
    }


    nobles_1 = [
        noble.id for noble in state_1.nobles
    ]

    # Reset with the exact same seed
    env.reset(seed=420)
    state_2 = env.state

    visible_cards_2 = {
        tier: [card.id for card in cards]
        for tier, cards in env.state.visible_cards.items()
    }

    decks_2 = {
        tier: [card.id for card in cards]
        for tier, cards in env.state.decks.items()
    }


    nobles_2 = [
        noble.id for noble in state_2.nobles
    ]

    assert visible_cards_1 == visible_cards_2
    assert decks_1 == decks_2
    assert nobles_1 == nobles_2

def test_reset_different_seed_generates_different_game():
    env = SplendorEnv(num_players=2)

    env.reset(seed=420)

    visible_cards_1 = {
        tier: [card.id for card in cards]
        for tier, cards in env.state.visible_cards.items()
    }
        


    env.reset(seed=421)

    visible_cards_2 = {
        tier: [card.id for card in cards]
        for tier, cards in env.state.visible_cards.items()
    }


    assert visible_cards_1 != visible_cards_2