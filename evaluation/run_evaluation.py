from splendor_v1.agents.random_agent import RandomAgent
from splendor_v1.agents.mcts_agent import MCTSAgent

from splendor_v1.evaluation.evaluate_agents import evaluate_agents

def main():

    random_agent = RandomAgent()
    mctsagent =MCTSAgent(simulations=10)

    results = evaluate_agents(
        agent_a=random_agent,
        agent_b=random_agent,
        num_games=1
    )

    print("\nEvaluation Results")
    print("------------------")

    for key, value in results.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()