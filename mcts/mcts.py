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

    def random_rollout(self, env, state):

        rollout_state = state.clone()

        root_player = rollout_state.current_player

        while not env._check_terminated(state):

            actions = env._legal_actions(rollout_state)


            if len(actions) == 0:
                # early termination due to no legal moves
                # return 0
                raise RuntimeError(
                    f"Rollout deadlock at turn {rollout_state.turn}"
                )

            action = random.choice(actions)


            env.step(action, state=rollout_state)

        winners = env._compute_winners(rollout_state)

        return 1.0 if root_player in winners else 0.0


    def backup(self):

        pass