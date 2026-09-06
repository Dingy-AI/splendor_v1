from splendor_v1.mcts.mcts import MCTS
from collections import defaultdict
from splendor_v1.env.core.enums import NodeType

class NeuralPUCTAgent:

    def __init__(
        self,
        model,
        simulations=20,
        debug_mode=False,
        teacher_mode = False,
    ):

        self.teacher_mode = teacher_mode
        self.model = model
        self.debug_mode = debug_mode
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
            teacher_mode=self.teacher_mode
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

        if self.debug_mode:

            state = root.state
            player = state.players[state.current_player]

            print(f"Score: {player.points}")
            print(f"Gems: {player.gems}")
            print(f"Bonuses: {player.bonuses}")
            print(
                f"Reserved cards: {player.reserved_cards}"
            )            
            self.mcts.print_root_debug(
                env=env,
                root=root,
                chosen_action=best_child.action,
                top_k=5,
            )

        return best_child.action