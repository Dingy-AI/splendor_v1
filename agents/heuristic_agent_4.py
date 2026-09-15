import random
import numpy as np

from splendor_v1.agents.heuristic_agent_3 import HeuristicAgent3


class HeuristicAgent4(HeuristicAgent3):

    def __init__(
        self,
        num_rollouts=8,
        max_rollout_steps=200,
        name=None

    ):

        self.name = name
        self.num_rollouts = num_rollouts
        self.max_rollout_steps = max_rollout_steps


    # ============================================================
    # SELECT ACTION
    # ============================================================

    def select_action(
        self,
        env,
        state,
    ):

        scored_candidates = self._get_scored_candidates(
            env,
            state,
        )

        if not scored_candidates:
            return None

        root_player = state.current_player


        # --------------------------------------------------------
        # Sample hidden worlds ONCE.
        #
        # Every candidate action is evaluated against the SAME
        # possible hidden deck orders.
        #
        # This greatly reduces rollout variance and prevents one
        # candidate from simply getting luckier deck samples.
        # --------------------------------------------------------

        sampled_states = [
            self._sample_hidden_world(state)
            for _ in range(self.num_rollouts)
        ]


        evaluated = []

        for action, heuristic_score in scored_candidates:

            rollout_values = []

            for sampled_state in sampled_states:

                value = self._evaluate_with_rollout(
                    env=env,
                    sampled_state=sampled_state,
                    first_action=action,
                    root_player=root_player,
                )

                rollout_values.append(value)


            mean_rollout_value = float(
                np.mean(rollout_values)
            )


            evaluated.append(
                (
                    action,
                    mean_rollout_value,
                    heuristic_score,
                )
            )


        # Primary:
        #     estimated rollout win value
        #
        # Secondary:
        #     H3 heuristic score
        best_action, _, _ = max(
            evaluated,
            key=lambda x: (
                x[1],
                x[2],
            ),
        )

        return best_action


    # ============================================================
    # SAMPLE A POSSIBLE HIDDEN FUTURE
    # ============================================================

    def _sample_hidden_world(
        self,
        state,
    ):

        sampled_state = state.clone()


        # --------------------------------------------------------
        # IMPORTANT:
        #
        # Visible cards remain untouched.
        #
        # Only shuffle cards that are still hidden inside the
        # decks.
        #
        # This assumes:
        #
        #     sampled_state.decks[tier]
        #
        # contains only the unrevealed cards for that tier.
        # --------------------------------------------------------

        for tier in sampled_state.decks:

            deck = sampled_state.decks[tier]

            random.shuffle(deck)


        return sampled_state


    # ============================================================
    # ROLLOUT
    # ============================================================

    def _evaluate_with_rollout(
        self,
        env,
        sampled_state,
        first_action,
        root_player,
    ):

        rollout_env = env.clone()

        # Start from one randomized possible world.
        rollout_env.state = sampled_state.clone()


        # --------------------------------------------------------
        # Play candidate move
        # --------------------------------------------------------

        rollout_env.step(
            first_action
        )


        # --------------------------------------------------------
        # H3 controls the rest of the simulated game.
        #
        # Explicitly call H3 so we don't recursively call H4.
        # --------------------------------------------------------

        steps = 0

        while not rollout_env._check_terminated(
            rollout_env.state
        ):

            steps += 1

            if steps > self.max_rollout_steps:
                return 0.0


            rollout_state = rollout_env.state

            action = HeuristicAgent3.select_action(
                self,
                rollout_env,
                rollout_state,
            )


            if action is None:
                return 0.0


            rollout_env.step(
                action
            )


        # --------------------------------------------------------
        # Terminal result from original player's perspective
        # --------------------------------------------------------

        winners = rollout_env.state.winners

        if not winners:
            return 0.0

        if root_player in winners:
            return 1.0

        return -1.0