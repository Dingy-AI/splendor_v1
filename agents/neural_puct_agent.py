from splendor_v1.mcts.mcts import MCTS


class NeuralPUCTAgent:

    def __init__(
        self,
        model,
        simulations=5,
    ):
        self.model = model

        self.mcts = MCTS(
            simulations=simulations,
            rollout_type="neural",
            selection_type="puct",
            model=model,
        )

    def select_action(
        self,
        env,
        state,
    ):
        _, root = self.mcts.search(
            env,
            state,
            return_root=True,
        )

        if not root.children:
            raise ValueError(
                "NeuralPUCTAgent could not find "
                "any legal child actions."
            )

        best_child = max(
            root.children,
            key=lambda child: child.visits,
        )

        return best_child.action