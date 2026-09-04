from splendor_v1.env.env import SplendorEnv

def evaluate_agents(
    agent_a,
    agent_b,
    num_games=100,
    max_steps=300,
    debug_mode=True
):

    results = {
        "agent_a_wins": 0,
        "agent_b_wins": 0,
        "ties": 0,
        "deadlocks": 0,
        "games_played": 0,
        "total_steps": 0,
        "aborted": 0
    }

    for game_index in range(num_games):
        if debug_mode:
            print("Starting: ", game_index)

        # -------------------------
        # Alternate player positions
        # -------------------------

        if game_index % 2 == 0:

            # A = Player 0
            # B = Player 1
            result = play_game(
                agent_a,
                agent_b,
                max_steps=max_steps,
            )

            agent_a_index = 0
            agent_b_index = 1

        else:

            # B = Player 0
            # A = Player 1
            result = play_game(
                agent_b,
                agent_a,
                max_steps=max_steps,
            )

            agent_a_index = 1
            agent_b_index = 0
        print("Game: ", game_index, " completed with results ", result)
        # -------------------------
        # Record statistics
        # -------------------------

        results["games_played"] += 1
        results["total_steps"] += result["steps"]

        if result["deadlock"]:
            results["deadlocks"] += 1

            if debug_mode:
                print("Game Deadlocked.")
                print("Game Results: ", results)

            continue

        if result["aborted"]:
            results["aborted"] += 1

            if debug_mode:
                print("Game Aborted: Max Steps Reached.")
                print("Game Results: ", results)

            continue

        winners = result["winners"]

        if len(winners) > 1:
            results["ties"] += 1

        elif agent_a_index in winners:
            results["agent_a_wins"] += 1

        elif agent_b_index in winners:
            results["agent_b_wins"] += 1
        if debug_mode:
            print("Game Results: ", results)
    # -------------------------
    # Calculate summary stats
    # -------------------------

    completed_games = (
        results["agent_a_wins"]
        + results["agent_b_wins"]
        + results["ties"]
    )

    results["completed_games"] = completed_games

    if completed_games > 0:

        results["agent_a_win_rate"] = (
            results["agent_a_wins"]
            / completed_games
        )

        results["agent_b_win_rate"] = (
            results["agent_b_wins"]
            / completed_games
        )

    else:
        results["agent_a_win_rate"] = 0.0
        results["agent_b_win_rate"] = 0.0

    results["average_steps"] = (
        results["total_steps"]
        / results["games_played"]
    )
    
    return results


def play_game(
    agent_0,
    agent_1,
    max_steps=300,
):

    env = SplendorEnv()
    env.reset()

    agents = {
        0: agent_0,
        1: agent_1,
    }

    terminated = False
    truncated = False
    steps = 0
    info = {}

    while not (terminated or truncated):

        legal_actions = env._legal_actions(env.state)

        # Game reached a dead end
        if not legal_actions:
            return {
                "winners": [],
                "steps": steps,
                "deadlock": True,
                "aborted": False,
            }

        current_player = env.state.current_player
        agent = agents[current_player]

        action = agent.select_action(
            env,
            env.state,
        )

        # Agent should NEVER return an illegal action
        if action not in legal_actions:
            raise ValueError(
                f"Player {current_player} returned "
                f"illegal action: {action}"
            )

        # print(
        #     f"Step={steps}, "
        #     f"Player={current_player}, "
        #     f"Agent={type(agent).__name__}, "
        #     f"Action={action}, "
        #     f"Scores="
        #     f"{[p.points for p in env.state.players]}"
        # )

        obs, reward, terminated, truncated, info = env.step(action)

        steps += 1

        if steps >= max_steps:
            return {
                "winners": [],
                "steps": steps,
                "deadlock": False,
                "aborted": True
            }

    return {
        "winners": info["winners"],
        "steps": steps,
        "deadlock": False,
        "aborted": False
    }