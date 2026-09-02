# splendor_v1/training/self_play.py

import numpy as np
from collections import defaultdict
from splendor_v1.env.core.constants import MAX_GEMS
from splendor_v1.env.core.action_constants import ACTION_SPACE_SIZE
import time 
from collections import Counter
import math

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

        # Debug root visit distribution

        # player = state.players[
        #     state.current_player
        # ]

        # total_gems = sum(
        #     player.gems.values()
        # )
        # if total_gems >= MAX_GEMS:


        #     prior_by_type = defaultdict(float)
        #     visits_by_type = defaultdict(int)
        #     value_by_type = defaultdict(float)
        #     actions_by_type = defaultdict(int)

        #     exploration_by_type = defaultdict(float)
        #     puct_by_type = defaultdict(float)

        #     total_visits = sum(
        #         child.visits
        #         for child in root.children
        #     )

        #     for child in root.children:
        #         action_type = child.action.action_type

        #         prior_by_type[action_type] += (
        #             child.prior
        #         )

        #         visits_by_type[action_type] += (
        #             child.visits
        #         )

        #         value_by_type[action_type] += (
        #             child.value
        #         )

        #         actions_by_type[action_type] += 1

        #         # Q for this individual child
        #         if child.visits > 0:
        #             child_q = (
        #                 child.value
        #                 / child.visits
        #             )
        #         else:
        #             child_q = 0.0

        #         # Same exploration formula used by PUCT
        #         c_puct = 1.5
        #         child_exploration = (
        #             c_puct
        #             * child.prior
        #             * math.sqrt(
        #                 max(root.visits, 1)
        #             )
        #             / (1 + child.visits)
        #         )

        #         child_puct = (
        #             child_q
        #             + child_exploration
        #         )

        #         # Weight by visits for action-type summary
        #         weight = max(child.visits, 1)

        #         exploration_by_type[action_type] += (
        #             child_exploration * weight
        #         )

        #         puct_by_type[action_type] += (
        #             child_puct * weight
        #         )

        #     print(
        #         f"\nPlayer {state.current_player} "
        #         f"at {total_gems} gems:"
        #     )

        #     action_types = sorted(
        #         prior_by_type.keys(),
        #         key=lambda action_type: (
        #             visits_by_type[action_type]
        #         ),
        #         reverse=True,
        #     )

        #     for action_type in action_types:

        #         visits = visits_by_type[
        #             action_type
        #         ]

        #         if visits > 0:
        #             q_value = (
        #                 value_by_type[action_type]
        #                 / visits
        #             )
        #         else:
        #             q_value = 0.0

        #         if total_visits > 0:
        #             target_policy = (
        #                 visits / total_visits
        #             )
        #         else:
        #             target_policy = 0.0

        #         weight = sum(
        #             max(child.visits, 1)
        #             for child in root.children
        #             if (
        #                 child.action.action_type
        #                 == action_type
        #             )
        #         )

        #         avg_exploration = (
        #             exploration_by_type[action_type]
        #             / weight
        #         )

        #         avg_puct = (
        #             puct_by_type[action_type]
        #             / weight
        #         )

        #         print(
        #             f"  {action_type.name}: "
        #             f"actions={actions_by_type[action_type]}, "
        #             f"prior={prior_by_type[action_type]:.3f}, "
        #             f"target={target_policy:.3f}, "
        #             f"visits={visits}, "
        #             f"Q={q_value:.3f}, "
        #             f"U={avg_exploration:.3f}, "
        #             f"PUCT={avg_puct:.3f}"
        #         )


        #     sorted_children = sorted(
        #         root.children,
        #         key=lambda child: child.visits,
        #         reverse=True,
        #     )

        #     print(
        #         f"\nPlayer {state.current_player} "
        #         f"at {total_gems} gems:"
        #     )

        #     for child in sorted_children[:10]:
        #         print(
        #             f"  visits={child.visits}, "
        #             f"prior={child.prior:.3f}, "
        #             f"action={child.action}"
        #         )



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


        print("Winners:", winners)

        for _, _, player in game_history[:10]:
            target_value = (
                1.0 if player in winners else -1.0
            )

            print(
                f"Player={player}, "
                f"Target value={target_value}"
            )

        print("\nSelf-play action counts:")

        game_action_counts = Counter()

        for counts in action_counts.values():
            game_action_counts.update(counts)

        game_total = sum(
            game_action_counts.values()
        )

        print(
            f"Game total: "
            f"{game_total} actions"
        )

        for action_type, count in (
            game_action_counts.items()
        ):
            percentage = (
                count / game_total * 100
                if game_total > 0
                else 0
            )

            print(
                f"  {action_type.name}: "
                f"{count} "
                f"({percentage:.1f}%)"
            )


        # Per-player breakdown

        for player in (0, 1):

            total = sum(
                action_counts[player].values()
            )

            print(
                f"\nPlayer {player}: "
                f"{total} actions"
            )

            for action_type, count in (
                action_counts[player].items()
            ):
                percentage = (
                    count / total * 100
                    if total > 0
                    else 0
                )

                print(
                    f"  {action_type.name}: "
                    f"{count} "
                    f"({percentage:.1f}%)"
                )





        return {
            "completed": True,
            "winners": winners,
            "game_length": len(game_history),
            "mcts_time": mcts_time,
        }
    else:


        print("\nSelf-play action counts:")

        game_action_counts = Counter()

        for counts in action_counts.values():
            game_action_counts.update(counts)

        game_total = sum(
            game_action_counts.values()
        )

        print(
            f"Game total: "
            f"{game_total} actions"
        )

        for action_type, count in (
            game_action_counts.items()
        ):
            percentage = (
                count / game_total * 100
                if game_total > 0
                else 0
            )

            print(
                f"  {action_type.name}: "
                f"{count} "
                f"({percentage:.1f}%)"
            )


        # Per-player breakdown

        for player in (0, 1):

            total = sum(
                action_counts[player].values()
            )

            print(
                f"\nPlayer {player}: "
                f"{total} actions"
            )

            for action_type, count in (
                action_counts[player].items()
            ):
                percentage = (
                    count / total * 100
                    if total > 0
                    else 0
                )

                print(
                    f"  {action_type.name}: "
                    f"{count} "
                    f"({percentage:.1f}%)"
                )


        
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