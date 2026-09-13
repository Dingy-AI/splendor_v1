import numpy as np

from splendor_v1.env.core.action_constants import ACTION_SPACE_SIZE


class ModelReplayGenerator:

    def __init__(
        self,
        env,
        mcts,
        replay_buffer,
        add_root_noise=False,
    ):
        self.env = env
        self.mcts = mcts
        self.replay_buffer = replay_buffer
        self.add_root_noise = add_root_noise


    # ==================================================
    # GENERATE ONE GAME
    # ==================================================

    def generate_game(self, seed=None):

        # -------------------------
        # RESET
        # -------------------------

        if seed is None:
            result = self.env.reset()
        else:
            result = self.env.reset(seed=seed)

        if hasattr(result, "current_player"):
            state = result
        else:
            state = self.env.state

        trajectory = []

        terminated = False

        # -------------------------
        # PLAY GAME
        # -------------------------

        while not terminated:

            player = state.current_player

            # ------------------------------------------
            # Encode current state
            # ------------------------------------------

            observation = (
                self.env
                .observation_encoder
                .encoder(state)
            )

            # ------------------------------------------
            # Run MCTS using the FROZEN model
            # ------------------------------------------

            action, root = self.mcts.search(
                self.env,
                state,
                return_root=True,
                add_root_noise=self.add_root_noise,
                teacher_mode=False,
            )

            # ------------------------------------------
            # Convert visit counts -> policy target
            # ------------------------------------------

            policy = self._get_policy_from_root(
                root
            )

            trajectory.append(
                (
                    observation.copy(),
                    policy.copy(),
                    player,
                )
            )

            # ------------------------------------------
            # Play strongest MCTS action
            #
            # IMPORTANT:
            # return exact Action object from tree.
            # Do NOT reconstruct with id_to_action().
            # ------------------------------------------

            if action is None:
                raise RuntimeError(
                    "MCTS returned no action "
                    "before termination."
                )

            result = self.env.step(action)

            # ------------------------------------------
            # Gymnasium-style termination handling
            # ------------------------------------------

            if isinstance(result, tuple):

                if len(result) == 5:

                    _, _, terminated, truncated, _ = result

                    terminated = (
                        terminated
                        or truncated
                    )

                elif len(result) == 4:

                    _, _, terminated, _ = result

            state = self.env.state

        # ==================================================
        # ASSIGN FINAL VALUES
        # ==================================================

        winners = state.winners

        if not winners:
            raise RuntimeError(
                "Game terminated without a winner."
            )

        for observation, policy, player in trajectory:

            if player in winners:
                value = 1.0
            else:
                value = -1.0

            self.replay_buffer.add(
                (
                    observation,
                    policy,
                    value,
                )
            )

        return len(trajectory)


    # ==================================================
    # ROOT -> POLICY
    # ==================================================

    def _get_policy_from_root(self, root):

        policy = np.zeros(
            ACTION_SPACE_SIZE,
            dtype=np.float32,
        )

        total_visits = 0

        for child in root.children:

            visits = child.visits

            if visits <= 0:
                continue

            action = child.action

            action_id = self.env.action_to_id(
                action
            )

            # Multiple exact actions can map to
            # the same neural-network action ID.
            policy[action_id] += visits

            total_visits += visits

        if total_visits <= 0:
            raise RuntimeError(
                "MCTS root has no visited children."
            )

        policy /= total_visits

        return policy


    # ==================================================
    # ROOT -> ACTION
    # ==================================================

    def _select_action_from_root(self, root):

        if not root.children:
            return None

        best_child = max(
            root.children,
            key=lambda child: child.visits,
        )

        return best_child.action


    # ==================================================
    # VISIT COUNT HELPER
    # ==================================================

    @staticmethod
    def _get_visit_count(node):

        # Change this helper if your Node uses
        # a different field name.

        if hasattr(node, "visit_count"):
            return node.visit_count

        if hasattr(node, "visits"):
            return node.visits

        if hasattr(node, "N"):
            return node.N

        raise AttributeError(
            "Could not find visit count "
            "on MCTS Node."
        )