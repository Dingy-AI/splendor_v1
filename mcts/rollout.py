import random 
from splendor_v1.env.core.actions import ActionType
from splendor_v1.env.core.constants import COLOR_ORDER
def random_rollout(

        env,
        child,
        root_player,
        return_state=False,
        compute_observation=False
    ):

        rollout_state = child.state.clone()

        steps = 0

        while True:

            actions = env._legal_actions(rollout_state)

            if not actions:
                return -1.0

            action = random.choice(actions)

            obs, reward, terminated, truncated, info = env.step(
                action,
                state=rollout_state,
                compute_observation=compute_observation
            )

            steps += 1

            if terminated or truncated:
                break

            if steps >= 500:
                return 0.0

        if return_state:
            return rollout_state

        winners = info["winners"]

        return 1.0 if root_player in winners else 0.0

def heuristic_rollout(
    env,
    child,
    root_player,
    max_steps=40,
    compute_observation=False
):

    rollout_state = child.state.clone()

    for _ in range(max_steps):

        actions = env._legal_actions(rollout_state)

        if not actions:
            return -1.0

        action = heuristic_action(
            rollout_state,
            actions
        )

        _, _, terminated, truncated, info = env.step(
            action,
            state=rollout_state,
            compute_observation=compute_observation
        )

        if terminated or truncated:

            winners = info["winners"]

            return (
                1.0
                if root_player in winners
                else 0.0
            )

    # No winner after max_steps
    return heuristic_value(
        rollout_state,
        root_player
    )


def heuristic_action(
    state,
    actions
):

    buy_actions = [
        action
        for action in actions
        if action.action_type in (
            ActionType.BUY_VISIBLE,
            ActionType.BUY_RESERVED,
        )
    ]

    if buy_actions:
        return random.choice(buy_actions)

    noble_actions = [
        action
        for action in actions
        if action.action_type == ActionType.TAKE_NOBLE
    ]

    if noble_actions:
        return random.choice(noble_actions)

    discard_actions = [
        action
        for action in actions
        if action.action_type == ActionType.DISCARD_GEMS
    ]

    # Overflow is effectively forced
    if discard_actions:
        return random.choice(discard_actions)

    take_actions = [
        action
        for action in actions
        if action.action_type == ActionType.TAKE_GEMS
    ]

    reserve_actions = [
        action
        for action in actions
        if action.action_type in (
            ActionType.RESERVE_VISIBLE,
            ActionType.RESERVE_TOP_DECK,
        )
    ]

    if take_actions and random.random() < 0.85:
        return random.choice(take_actions)

    if reserve_actions:
        return random.choice(reserve_actions)

    return random.choice(actions)

def heuristic_value(
    state,
    root_player
):

    root_points = state.players[root_player].points

    opponent_points = max(
        player.points
        for i, player in enumerate(state.players)
        if i != root_player
    )

    value = (
        root_points - opponent_points
    ) / 15.0

    return max(
        -1.0,
        min(1.0, value)
    )


def score_buy_action(env, state, action):

    player = state.players[state.current_player]

    if action.action_type == ActionType.BUY_VISIBLE:
        card = state.visible_cards[action.tier][action.slot]

    elif action.action_type == ActionType.BUY_RESERVED:
        card = player.reserved_cards[action.reserved_index]

    else:
        return float("-inf")

    return card.points * 3.0 + 1.0

def card_deficit(player, card):

    deficit = 0

    for color in COLOR_ORDER:

        required = max(
            0,
            card.cost.get(color, 0)
            - player.bonuses.get(color, 0)
        )

        missing = max(
            0,
            required
            - player.gems.get(color, 0)
        )

        deficit += missing

    return deficit

def score_take_action(env, state, action):

    player = state.players[state.current_player]

    cards = []

    for tier_cards in state.visible_cards.values():
        cards.extend(
            card for card in tier_cards
            if card is not None
        )

    cards.extend(player.reserved_cards)

    if not cards:
        return 0.0

    before = min(
        card_deficit(player, card)
        for card in cards
    )

    # Cheap temporary gem counts
    temp_gems = player.gems.copy()

    for color in action.gem_colors:
        temp_gems[color] += 1

    class TempPlayer:
        pass

    temp_player = TempPlayer()
    temp_player.gems = temp_gems
    temp_player.bonuses = player.bonuses

    after = min(
        card_deficit(temp_player, card)
        for card in cards
    )

    return before - after

def heuristic_action_upgraded(env, state, actions):

    scored = []

    for action in actions:

        if action.action_type in (
            ActionType.BUY_VISIBLE,
            ActionType.BUY_RESERVED,
        ):
            score = score_buy_action(
                env,
                state,
                action,
            )

        elif action.action_type == ActionType.TAKE_GEMS:
            score = score_take_action(
                env,
                state,
                action,
            )

        elif action.action_type in (
            ActionType.RESERVE_VISIBLE,
            ActionType.RESERVE_TOP_DECK,
        ):
            score = 0.5

        elif action.action_type == ActionType.TAKE_NOBLE:
            score = 10.0

        elif action.action_type == ActionType.DISCARD_GEMS:
            score = 0.0

        else:
            score = 0.0

        scored.append(
            (score, action)
        )

    # Keep some stochasticity
    if random.random() < 0.15:
        return random.choice(actions)

    return max(
        scored,
        key=lambda x: x[0]
    )[1]


def heuristic_rollout_v2(
    env,
    child,
    root_player,
    max_steps=40,
):

    rollout_state = child.state.clone()

    for _ in range(max_steps):

        actions = env._legal_actions(rollout_state)

        # Dead-end state
        if not actions:
            return -1.0

        # Choose a more sensible action
        action = heuristic_action_upgraded(
            env,
            rollout_state,
            actions,
        )

        _, _, terminated, truncated, info = env.step(
            action,
            state=rollout_state,
        )

        if terminated or truncated:

            winners = info["winners"]

            return (
                1.0
                if root_player in winners
                else 0.0
            )

    # No terminal result after max_steps
    return heuristic_value(
        rollout_state,
        root_player,
    )