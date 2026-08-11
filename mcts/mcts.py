import random 
from splendor_v1.mcts.node import Node

class MCTS:

    def __init__(self, env, simulations=1000):

        self.env = env

        self.simulations = simulations

    def search(self, state):

        root = Node(
            state=state.clone()
        )

        for _ in range(self.simulations):


            try:
                leaf = self.select(root)

                child = self.expand(leaf)

                value = self.random_rollout(
                    self.env,
                    child.state
                )

                self.backup(child, value)


            except Exception as e:
                print(f"MCTS simulation failed: {e}")
                continue

        return self.best_action(root)

    def select(self):

        pass

    def expand(self):

        pass

    def random_rollout(self, env):

        rollout_env = env.clone()

        root_player = rollout_env.state.current_player

        terminated = False
        steps = 0

        while not terminated:

            actions = rollout_env._legal_actions(rollout_env.state)

            if not actions:
                raise RuntimeError(
                    "Rollout reached state with no legal actions"
                )

            action = random.choice(actions)

            obs, reward, terminated, truncated, info = rollout_env.step(action)
            steps += 1

            if steps > 500:
                terminated = True
                return 0.0

        winners = info['winners']

        return 1.0 if root_player in winners else 0.0


    def backup(self):

        pass