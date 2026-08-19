from splendor_v1.mcts.mcts import MCTS

class MCTSAgent:
    def __init__(self, simulations=100):
        self.mcts = MCTS(simulations=simulations, rollout_type="random")

    def select_action(self, env, state):
        # return self.mcts.search(env, state, debug=True)
        return self.mcts.search(env, state, debug=False)