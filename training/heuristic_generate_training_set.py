from splendor_v1.training.heuristic_replay_generator import HeuristicReplayGenerator
from splendor_v1.env.env import SplendorEnv
from splendor_v1.agents.heuristic_agent_3 import HeuristicAgent3
from splendor_v1.training.replay_buffer import ReplayBuffer
from splendor_v1.agents.heuristic_agent_2 import HeuristicAgent2


import pickle


env = SplendorEnv()
agent = HeuristicAgent2()

replay_buffer = ReplayBuffer(500_000)

generator = HeuristicReplayGenerator(
    env=env,
    agent=agent,
    replay_buffer=replay_buffer,
)

num_games = 1000

# num_games=10

for game in range(num_games):

    num_positions = generator.generate_game(
        seed=game,
    )

    print(
        f"Game {game + 1}/{num_games} "
        f"- {num_positions} positions "
        f"- buffer size: {len(replay_buffer.buffer)}"
    )


data = {
    "capacity": replay_buffer.capacity,
    "buffer": replay_buffer.buffer,
    "position": replay_buffer.position,
}

with open(
    "splendor_v1/training/data/heuristic_replay_buffer_2.pkl",
    "wb",
) as f:
    pickle.dump(
        data,
        f,
        protocol=pickle.HIGHEST_PROTOCOL,
    )