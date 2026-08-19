import random 
from splendor_v1.env.core.actions import ActionType

def random_rollout(

        env,
        child,
        root_player,
        return_state=False
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
                state=rollout_state
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
            state=rollout_state
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