# splendor_v1/training/self_play.py

import numpy as np

from splendor_v1.env.core.action_constants import ACTION_SPACE_SIZE


def root_visit_policy(env, root):

    target_policy = np.zeros(
        ACTION_SPACE_SIZE,
        dtype=np.float32,
    )

    total_visits = sum(
        child.visits
        for child in root.children
    )

    if total_visits == 0:
        raise ValueError(
            "Root has no child visits." \
            ""
        )

    for child in root.children:

        action_id = env.action_to_id(
            child.action
        )

        target_policy[action_id] = (
            child.visits / total_visits
        )

    return target_policy

def play_self_play_game(
    env,
    mcts,
    replay_buffer,
):

    env.reset()

    game_history = []

    terminated = False
    truncated = False
    info = {}

    while not terminated and not truncated:

        state = env.state

        observation = (
            env.observation_encoder.encoder(
                state
            )
        )

        action, root = mcts.search(
            env,
            state,
            return_root=True,
        )

        if action is None:
            break

        target_policy = root_visit_policy(
            env,
            root,
        )

        game_history.append(
            (
                observation,
                target_policy,
                state.current_player,
            )
        )

        
        _, _, terminated, truncated, info = env.step(action)

    if not terminated:
        return

    winners = info["winners"]

    for (
        observation,
        target_policy,
        player,
    ) in game_history:

        if len(winners) != 1:
            target_value = 0.0

        elif player in winners:
            target_value = 1.0

        else:
            target_value = -1.0

        replay_buffer.add(
            observation,
            target_policy,
            target_value,
        )

    return {
        "winners": winners,
        "game_length": len(game_history),
    }