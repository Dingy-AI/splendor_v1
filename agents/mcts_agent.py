from splendor_v1.mcts.mcts import MCTS

class MCTSAgent:
    def __init__(self, simulations=100, rollout_type="random"):
        self.mcts = MCTS(simulations=simulations, rollout_type=rollout_type)

    def select_action(self, env, state):
        # return self.mcts.search(env, state, debug=True)
        return self.mcts.search(env, state, debug=False)