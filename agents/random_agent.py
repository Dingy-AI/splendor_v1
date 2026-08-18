import random

class RandomAgent:

    def select_action(self, env, state):
        return random.choice(
            env._legal_actions(state)
        )