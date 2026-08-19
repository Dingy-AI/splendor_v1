from splendor_v1.agents.random_agent import RandomAgent
from splendor_v1.agents.mcts_agent import MCTSAgent

from splendor_v1.evaluation.evaluate_agents import evaluate_agents

def main_random():

    random_agent = RandomAgent()
    mctsagent =MCTSAgent(simulations=10)

    results = evaluate_agents(
        agent_a=random_agent,
        agent_b=random_agent,
        num_games=50,
        print_mode = True
    )

    print("\nEvaluation Results")
    print("------------------")

    for key, value in results.items():
        print(f"{key}: {value}")


def main_random_mcts():

    random_agent = RandomAgent()
    mctsagent =MCTSAgent(simulations=25)

    results = evaluate_agents(
        agent_a=random_agent,
        agent_b=mctsagent,
        num_games=20,
        debug = False
    )

    print("\nEvaluation Results")
    print("------------------")

    for key, value in results.items():
        print(f"{key}: {value}")


def main_random_mcts_once_sim_5():

    random_agent = RandomAgent()
    mctsagent =MCTSAgent(simulations=5)

    results = evaluate_agents(
        agent_a=random_agent,
        agent_b=mctsagent,
        num_games=1,
        debug_mode = True
    )

    print("\nEvaluation Results")
    print("------------------")

    for key, value in results.items():
        print(f"{key}: {value}")




if __name__ == "__main__":
    # main_random_mcts_once_sim_5()
    main_random_mcts_once_sim_5()