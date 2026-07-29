import random
from collections import Counter

from splendor_v1.env.env import SplendorEnv
from splendor_v1.env.core.enums import ActionType

#python -m splendor_v1.env_analysis.env_validation    


def validate_environment(num_games=1000):

    stats = {
        "completed_games": 0,
        "soft_locked_games": 0,
        "soft_lock_seeds": [],
        "soft_lock_turn": [],
        "soft_lock_state": [],
        "info":"",
        "turns": [],
        "winner_points": [],
        "winner_cards": [],
        "nobles_taken": [],
        "cards_bought": [],
        "reserved_cards_bought": [],
        "overflow_events": [],
        "action_counts": Counter(),
    }

    MAX_TURNS = 300

    for seed in range(num_games):
        print("PLAYING GAME: ", seed)

        
        random.seed(seed)

        env = SplendorEnv()
        env.reset()

        turns = 0
        reserved_buys = 0
        overflow_events = 0

        terminated = False

        while not terminated and turns < MAX_TURNS:


            legal_actions = env._legal_actions(env.state)

            if not legal_actions:

                terminated = True
                stats["info"] = "stalemate"

                print(
                    f"""No legal actions!
                        Seed: {seed}
                        Turn: {turns}

                        State:
                        {env.state}
                        """)

                stats["soft_locked_games"] += 1
                stats["soft_lock_seeds"].append(seed)
                stats["soft_lock_turn"].append(turns)
                stats["soft_lock_state"].append(env.state)
                continue
            else:
                action = random.choice(legal_actions)

                stats["action_counts"][action.action_type] += 1

                if action.action_type == ActionType.BUY_RESERVED:
                    reserved_buys += 1

                if action.action_type == ActionType.DISCARD_GEMS:
                    overflow_events += 1

                obs, reward, terminated, truncated, info = env.step(action)

                turns += 1

        state = env.state

        winner = max(
            state.players,
            key=lambda p: (p.points, len(p.purchased_cards))
        )

        nobles_taken = sum(
            len(player.nobles)
            for player in state.players
        )

        total_cards_bought = sum(
            len(player.purchased_cards)
            for player in state.players
        )

        if terminated:
            stats["completed_games"] += 1


        stats["turns"].append(turns)
        stats["winner_points"].append(winner.points)
        stats["winner_cards"].append(len(winner.purchased_cards))
        stats["nobles_taken"].append(nobles_taken)
        stats["cards_bought"].append(total_cards_bought)
        stats["reserved_cards_bought"].append(reserved_buys)
        stats["overflow_events"].append(overflow_events)

    return stats

from statistics import mean


def print_validation_summary(stats):

    print(f"Games: {len(stats['turns'])}")

    print(f"Average turns: {mean(stats['turns']):.2f}")
    print(f"Min turns: {min(stats['turns'])}")
    print(f"Max turns: {max(stats['turns'])}")

    print()

    print(f"Average winner points: {mean(stats['winner_points']):.2f}")
    print(f"Average winner cards: {mean(stats['winner_cards']):.2f}")

    print()

    print(f"Average nobles taken: {mean(stats['nobles_taken']):.2f}")
    print(f"Average cards bought: {mean(stats['cards_bought']):.2f}")
    print(f"Average reserved cards bought: {mean(stats['reserved_cards_bought']):.2f}")
    print(f"Average overflow events: {mean(stats['overflow_events']):.2f}")

    print("\nAction counts:")

    total_actions = sum(stats["action_counts"].values())

    for action_type, count in sorted(
        stats["action_counts"].items(),
        key=lambda x: x[1],
        reverse=True,
    ):
        percent = 100 * count / total_actions
        print(f"{action_type.name:<20} {count:>8} ({percent:.2f}%)")

stats = validate_environment(1000)
print_validation_summary(stats)
