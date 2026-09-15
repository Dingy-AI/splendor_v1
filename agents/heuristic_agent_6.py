import random
import numpy as np

from splendor_v1.agents.heuristic_agent_5 import HeuristicAgent5


class HeuristicAgent6(HeuristicAgent5):
    """
    HeuristicAgent6 = HeuristicAgent5 + H4-style terminal rollouts.

    Root behavior:
        - Inherit H5's candidate generation.
        - For every H5 candidate action, evaluate the action over
          multiple determinizations of the hidden decks.
        - After the root action, BOTH players follow H5 until the
          game terminates.
        - Choose the action with the best mean terminal result.
        - Use the original H5 score only as a tiebreaker.

    Conceptually:

        H6(s) = argmax_a E[ terminal result |
                            take a,
                            then both players follow H5 ]

    Important:
        - H6 does NOT recursively call itself during rollouts.
        - Rollout actions are selected explicitly with H5 logic.
        - Every root candidate is tested against the SAME sampled
          hidden worlds ("common random numbers"), reducing noise.
        - Hidden deck order is shuffled so H6 cannot exploit the
          real future deck sequence.
        - This implementation matches the project's confirmed API:

              env.step(action, state)

    This version is intended for 2-player Splendor.
    """

    def __init__(
        self,
        num_rollouts=8,
        max_rollout_steps=200,
        random_seed=None,
        name=None

    ):
        self.name=name
        self.num_rollouts = max(1, num_rollouts)
        self.max_rollout_steps = max_rollout_steps
        self.rng = random.Random(random_seed)

    # ============================================================
    # PUBLIC API
    # ============================================================

    def select_action(
        self,
        env,
        state,
    ):
        evaluated = self._get_rollout_evaluated_candidates(
            env,
            state,
        )

        if not evaluated:
            return None

        best_action, _, _ = max(
            evaluated,
            key=lambda item: (
                item[1],
                item[2],
            ),
        )

        return best_action

    def get_policy(
        self,
        env,
        state,
        action_size=1139,
        temperature=0.25,
    ):
        """
        Soft policy over rollout-evaluated H5 candidates.

        Rollout values are approximately in [-1, 1], so H5's
        temperature=10 would be far too flat here.
        """

        if temperature <= 0:
            raise ValueError(
                "temperature must be greater than 0"
            )

        evaluated = self._get_rollout_evaluated_candidates(
            env,
            state,
        )

        policy = np.zeros(
            action_size,
            dtype=np.float32,
        )

        if not evaluated:
            return policy

        actions = [
            action
            for action, _, _ in evaluated
        ]

        rollout_values = np.array(
            [
                rollout_value
                for _, rollout_value, _ in evaluated
            ],
            dtype=np.float64,
        )

        if not np.any(np.isfinite(rollout_values)):

            probs = np.ones(
                len(actions),
                dtype=np.float64,
            )

            probs /= probs.sum()

        else:

            rollout_values = np.where(
                np.isfinite(rollout_values),
                rollout_values,
                -1e9,
            )

            logits = (
                rollout_values
                / temperature
            )

            logits -= np.max(
                logits
            )

            probs = np.exp(
                logits
            )

            probs /= probs.sum()

        for action, prob in zip(
            actions,
            probs,
        ):

            action_id = env.action_to_id(
                action
            )

            policy[action_id] += prob

        return policy

    # ============================================================
    # ROOT ROLLOUT EVALUATION
    # ============================================================

    def _get_rollout_evaluated_candidates(
        self,
        env,
        state,
    ):
        """
        Get H5's candidate actions, then rank them by full terminal
        rollouts where both players use H5.

        Returns:
            [
                (
                    action,
                    mean_rollout_value,
                    original_h5_score,
                ),
                ...
            ]
        """

        h5_candidates = (
            HeuristicAgent5._get_scored_candidates(
                self,
                env,
                state,
            )
        )

        if not h5_candidates:
            return []

        # If H5 only has one possible candidate, there is no need
        # to spend rollout compute.
        if len(h5_candidates) == 1:

            action, h5_score = (
                h5_candidates[0]
            )

            return [
                (
                    action,
                    0.0,
                    h5_score,
                )
            ]

        root_player = (
            state.current_player
        )

        # --------------------------------------------------------
        # Sample hidden worlds ONCE.
        #
        # Every root candidate is evaluated against the same
        # determinizations for a fairer comparison.
        # --------------------------------------------------------

        sampled_worlds = (
            self._sample_hidden_worlds(
                state
            )
        )

        evaluated = []

        for action, h5_score in h5_candidates:

            rollout_values = []

            for sampled_state in sampled_worlds:

                value = self._rollout_from_action(
                    env=env,
                    sampled_state=sampled_state,
                    first_action=action,
                    root_player=root_player,
                )

                rollout_values.append(
                    value
                )

            mean_value = float(
                np.mean(
                    rollout_values
                )
            )

            evaluated.append(
                (
                    action,
                    mean_value,
                    h5_score,
                )
            )

        return evaluated

    # ============================================================
    # TERMINAL H5 ROLLOUT
    # ============================================================

    def _rollout_from_action(
        self,
        env,
        sampled_state,
        first_action,
        root_player,
    ):
        """
        Play first_action, then let H5 control BOTH players until
        terminal or max_rollout_steps.

        H5 is called explicitly so H6 never recursively launches
        more H6 rollouts.
        """

        sim_env = env.clone()

        rollout_state = (
            sampled_state.clone()
        )

        # --------------------------------------------------------
        # Root candidate action.
        # --------------------------------------------------------

        rollout_state = self._apply_action(
            sim_env,
            rollout_state,
            first_action,
        )

        steps = 1

        # --------------------------------------------------------
        # Continue with H5 policy for both players.
        # --------------------------------------------------------

        while (
            steps < self.max_rollout_steps
            and not sim_env._check_terminated(
                rollout_state
            )
        ):

            action = self._select_h5_action(
                sim_env,
                rollout_state,
            )

            if action is None:
                return 0.0

            rollout_state = self._apply_action(
                sim_env,
                rollout_state,
                action,
            )

            steps += 1

        if sim_env._check_terminated(
            rollout_state
        ):

            return self._terminal_value(
                rollout_state,
                root_player,
            )

        # Safety cutoff / deadlock.
        return 0.0

    # ============================================================
    # H5 ROLLOUT POLICY
    # ============================================================

    def _select_h5_action(
        self,
        env,
        state,
    ):
        """
        Select one move using H5 only.

        Do NOT call self.select_action() here, because that would
        invoke H6 recursively.
        """

        scored_actions = (
            HeuristicAgent5._get_scored_candidates(
                self,
                env,
                state,
            )
        )

        if not scored_actions:
            return None

        return max(
            scored_actions,
            key=lambda item: item[1],
        )[0]

    # ============================================================
    # HIDDEN-WORLD DETERMINIZATION
    # ============================================================

    def _sample_hidden_worlds(
        self,
        state,
    ):
        """
        Create num_rollouts plausible hidden-deck worlds.

        Visible market cards remain fixed.
        The still-hidden cards in state.decks are shuffled.

        If other hidden/private information exists separately in
        GameState, that can be determinized later as well.
        """

        worlds = []

        for _ in range(
            self.num_rollouts
        ):

            sampled_state = (
                state.clone()
            )

            self._shuffle_hidden_decks(
                sampled_state
            )

            worlds.append(
                sampled_state
            )

        return worlds

    def _shuffle_hidden_decks(
        self,
        state,
    ):

        decks = state.decks

        if isinstance(
            decks,
            dict,
        ):

            for tier in decks:

                self.rng.shuffle(
                    decks[tier]
                )

        else:

            for deck in decks:

                self.rng.shuffle(
                    deck
                )

    # ============================================================
    # ENVIRONMENT STEP
    # ============================================================

    def _apply_action(
        self,
        env,
        state,
        action,
    ):
        """
        Project environment API:

            env.step(action, state)

        The supplied state is cloned before mutation so each
        hypothetical rollout branch remains isolated.
        """

        next_state = (
            state.clone()
        )

        result = env.step(
            action,
            next_state,
        )

        # The environment is expected to mutate next_state.
        # If it instead returns a GameState directly, accept that
        # too without breaking compatibility.
        if (
            result is not None
            and hasattr(
                result,
                "current_player",
            )
            and hasattr(
                result,
                "players",
            )
        ):
            return result

        return next_state

    # ============================================================
    # TERMINAL RESULT
    # ============================================================

    def _terminal_value(
        self,
        state,
        root_player,
    ):
        """
        +1 = root player wins
         0 = draw / no winner
        -1 = root player loses
        """

        winners = getattr(
            state,
            "winners",
            None,
        )

        if not winners:
            return 0.0

        if len(winners) > 1:
            return 0.0

        if root_player in winners:
            return 1.0

        return -1.0
