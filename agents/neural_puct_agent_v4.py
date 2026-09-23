from splendor_v1.mcts.mcts_v4 import MCTS


class NeuralPUCTAgentV4:

    def __init__(
        self,
        model,
        simulations=20,
        debug_mode=False,
        teacher_mode=False,
        name=None,
        c_puct=3.0,
        add_root_noise=False,
    ):
        self.name = name
        self.teacher_mode = teacher_mode
        self.model = model
        self.debug_mode = debug_mode
        self.add_root_noise = add_root_noise

        # Evaluation agent: make sure dropout / training-only
        # behavior is disabled.
        self.model.eval()

        self.mcts = MCTS(
            simulations=simulations,
            rollout_type="neural",
            selection_type="puct",
            model=model,
            c_puct=c_puct,
        )

    def select_action(
        self,
        env,
        state,
    ):
        # Defensive: keep the network in evaluation mode even if
        # external code temporarily switched it to train mode.
        self.model.eval()

        action, root = self.mcts.search(
            env,
            state,
            return_root=True,
            teacher_mode=self.teacher_mode,
            add_root_noise=self.add_root_noise,
        )

        if action is None:
            raise ValueError(
                "NeuralPUCTAgentV4 could not find "
                "a legal action."
            )

        if not root.children:
            raise ValueError(
                "NeuralPUCTAgentV4 could not find "
                "any legal child actions."
            )

        if self.debug_mode:

            root_state = root.state

            player = (
                root_state.players[
                    root_state.current_player
                ]
            )

            print(
                f"Score: {player.points}"
            )

            print(
                f"Gems: {player.gems}"
            )

            print(
                f"Bonuses: {player.bonuses}"
            )

            print(
                f"Reserved cards: "
                f"{player.reserved_cards}"
            )

            self.mcts.print_root_debug(
                env=env,
                root=root,
                chosen_action=action,
                top_k=5,
            )

        return action
