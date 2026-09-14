import numpy as np

from splendor_v1.env.core.enums import NodeType
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

    def generate_game(self, seed=None, max_steps=200):

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
        step_count = 0
        main_decision_count = 0

        while not terminated:

            step_count += 1

            if step_count > max_steps:
                raise RuntimeError(
                    f"Game exceeded max_steps={max_steps} "
                    f"for seed={seed}."
                )

            player = state.current_player

            observation = (
                self.env
                .observation_encoder
                .encoder(state)
            )

            action, root = self.mcts.search(
                self.env,
                state,
                return_root=True,
                add_root_noise=self.add_root_noise,
                teacher_mode=False,
            )



            # ---------------------------------------
            # Temperature schedule
            # ---------------------------------------

            if state.node_type == NodeType.MAIN_DECISION:

                if main_decision_count < 8:
                    temperature = 0.5
                else:
                    temperature = 0.0

                main_decision_count += 1

            else:
                # Forced discard/noble decisions don't
                # really need exploration.
                temperature = 0.0



            policy = self._get_policy_from_root(
                root
            )

            # ---------------------------------------
            # Select actual move
            # ---------------------------------------

            action = self._sample_action_from_root(
                root,
                temperature,
            )


            trajectory.append(
                (
                    observation.copy(),
                    policy.copy(),
                    player,
                )
            )

            if action is None:
                raise RuntimeError(
                    f"MCTS returned no action "
                    f"for seed={seed}, "
                    f"step={step_count}."
                )

            result = self.env.step(action)

            if isinstance(result, tuple):

                if len(result) == 5:
                    _, _, terminated, truncated, _ = result
                    terminated = terminated or truncated

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


    def _sample_action_from_root(
        self,
        root,
        temperature,
    ):

        children = [
            child
            for child in root.children
            if child.visits > 0
        ]

        if not children:
            return None

        # Deterministic
        if temperature <= 1e-8:
            return max(
                children,
                key=lambda child: child.visits,
            ).action

        visits = np.array(
            [child.visits for child in children],
            dtype=np.float64,
        )

        probs = visits ** (1.0 / temperature)
        probs /= probs.sum()

        index = np.random.choice(
            len(children),
            p=probs,
        )

        return children[index].action