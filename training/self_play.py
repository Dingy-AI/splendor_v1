# splendor_v1/training/self_play.py

import numpy as np
from collections import defaultdict
from splendor_v1.env.core.constants import MAX_GEMS
from splendor_v1.env.core.action_constants import ACTION_SPACE_SIZE
import time 
from collections import Counter
import math
from splendor_v1.env.core.enums import NodeType, ActionType
import torch

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

def get_child_for_action(
    root,
    action,
):
    for child in root.children:

        if child.action == action:
            return child

    return None

def old_play_self_play_game(
    env,
    mcts,
    replay_buffer,
):

    action_counts = {
        0: Counter(),
        1: Counter(),
    }

    env.reset()

    game_history = []

    terminated = False
    truncated = False
    info = {}

    mcts_time = 0.0

    turn_count = 0
    MAX_TURNS = 300

    while not terminated and not truncated:
        turn_count += 1
        if turn_count > MAX_TURNS:
            print(
                f"Game aborted after "
                f"{MAX_TURNS} turns"
            )
            break

        if turn_count % 25 == 0:
            print(
                f"Turn {turn_count}: "
                f"player={env.state.current_player}, "
                f"node={env.state.node_type}"
            )



        state = env.state

        observation = (
            env.observation_encoder.encoder(
                state
            )
        )
        mcts_start = time.perf_counter()


        _, root = mcts.search(
            env,
            state,
            return_root=True,
        )

        mcts_time += (
            time.perf_counter()
            - mcts_start
        )

        if not root.children:
            break

        action = select_self_play_action(
            root,
            temperature=1.0,
        )

        if action is None:
            break

        target_policy = root_visit_policy(
            env,
            root,
        )

        player = state.current_player

        action_counts[player][
            action.action_type
        ] += 1

        game_history.append(
            (
                observation,
                target_policy,
                player,
            )
        )

        
        _, _, terminated, truncated, info = env.step(action)

    if terminated:
        winners = info["winners"]

        for (observation,target_policy,player) in game_history:


            if player in winners:
                target_value = 1.0

            else:
                target_value = -1.0

            replay_buffer.add(
                observation,
                target_policy,
                target_value,
            )


        return {
            "completed": True,
            "winners": winners,
            "game_length": len(game_history),
            "mcts_time": mcts_time,
        }
    else:
        
        return {
            "completed": False,
            "mcts_time": mcts_time
        }

def select_self_play_action(
    root,
    temperature=1.0,
):
    if not root.children:
        raise ValueError(
            "Cannot select action from root with no children."
        )

    visits = np.array(
        [
            child.visits
            for child in root.children
        ],
        dtype=np.float64,
    )

    if temperature == 0:
        best_index = np.argmax(visits)
        return root.children[best_index].action

    if temperature < 0:
        raise ValueError(
            "Temperature must be >= 0."
        )

    adjusted_visits = (
        visits ** (1.0 / temperature)
    )

    total = adjusted_visits.sum()

    if total == 0:
        raise ValueError(
            "Root children have no visits."
        )

    probabilities = (
        adjusted_visits / total
    )

    index = np.random.choice(
        len(root.children),
        p=probabilities,
    )

    return root.children[index].action

def play_self_play_game(
    env,
    mcts,
    replay_buffer,
    policy_debug_samples=None,
    game_index=None,
    seed=None,
    teacher_mode=False
):

    action_counts = {
        0: Counter(),
        1: Counter(),
    }

    env.reset(seed)

    game_history = []

    terminated = False
    truncated = False
    info = {}

    mcts_time = 0.0

    turn_count = 0
    MAX_TURNS = 300

    current_root = None

    while not terminated and not truncated:

        turn_count += 1

        if turn_count > MAX_TURNS:
            print(
                f"Game aborted after "
                f"{MAX_TURNS} turns"
            )
            break

        # if turn_count % 25 == 0:
        #     print(
        #         f"Turn {turn_count}: "
        #         f"player={env.state.current_player}, "
        #         f"node={env.state.node_type}"
        #     )

        state = env.state

        observation = (
            env.observation_encoder.encoder(
                state
            )
        )

        old_player = state.current_player


        # -------------------------
        # MCTS search
        # -------------------------
        mcts_start = time.perf_counter()

        _, root = mcts.search(
            env,
            state,
            root=current_root,
            return_root=True,
            add_root_noise=True,
            teacher_mode=teacher_mode
        )



        mcts_time += (
            time.perf_counter()
            - mcts_start
        )

        if not root.children:
            break

        # -------------------------
        # Select actual self-play move
        # -------------------------
        action = select_self_play_action(
            root,
            temperature=1.0,
        )

        if action is None:
            break

        # Find the child corresponding to
        # the action actually selected.
        #
        # We cannot simply use best_child
        # returned by search because
        # temperature=1.0 may sample a
        # different action.
        next_root = None

        for child in root.children:

            if child.action == action:
                next_root = child
                break

        if next_root is None:
            raise ValueError(
                "Selected self-play action "
                "does not have a matching "
                "child in the MCTS root."
            )

        # -------------------------
        # Training targets
        # -------------------------
        target_policy = root_visit_policy(
            env,
            root,
        )

        #DEBUG HELPER
        if (
            policy_debug_samples is not None
            and state.turn_number % 20 == 0
        ):
            legal_action_ids = [
                env.action_to_id(child.action)
                for child in root.children
            ]
            policy_debug_samples.append({
                "game_index": game_index,
                "turn": state.turn_number,
                "observation": observation.copy(),
                "target_policy": target_policy.copy(),
                "legal_action_ids": legal_action_ids.copy(),
            })
                

        player = old_player

        action_counts[player][
            action.action_type
        ] += 1

        game_history.append(
            (
                observation,
                target_policy,
                player,
            )
        )

        # -------------------------
        # Play actual move
        # -------------------------
        (
            _,
            _,
            terminated,
            truncated,
            info,
        ) = env.step(action)

        new_player = env.state.current_player

        # -------------------------
        # Tree reuse
        # -------------------------

        # The selected child is now the
        # root of the next MCTS search.
        current_root = next_root

        # Detach the old tree so it can
        # be garbage collected.
        current_root.parent = None

        # MCTS node values are stored from
        # the previous root player's
        # perspective.
        #
        # If the player changed, convert
        # the entire reused subtree to the
        # new player's perspective.
        #
        # This intentionally DOES NOT flip
        # during forced discard or noble
        # decisions when current_player
        # remains unchanged.
        if new_player != old_player:

            mcts.flip_tree_values(
                current_root
            )

    # -------------------------
    # Store completed game
    # -------------------------
    if terminated:

        winners = info["winners"]

        for (
            observation,
            target_policy,
            player,
        ) in game_history:

            if player in winners:
                target_value = 1.0
            else:
                target_value = -1.0

            replay_buffer.add(
                (observation,
                target_policy,
                target_value)
            )

        return {
            "completed": True,
            "winners": winners,
            "game_length": len(game_history),
            "positions_added": len(game_history),            
            "mcts_time": mcts_time,
        }

    else:

        return {
            "completed": False,
            "mcts_time": mcts_time,
        }

