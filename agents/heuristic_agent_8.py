import math
import random
import numpy as np

from splendor_v1.env.core.constants import COLOR_ORDER
from splendor_v1.env.core.enums import GemColor, NodeType
from splendor_v1.agents.heuristic_agent_5 import HeuristicAgent5


class HeuristicAgent8(HeuristicAgent5):
    """
    HeuristicAgent8 = faster H6-style rollout teacher.

    H6's expensive structure was:

        each real decision
            x every H5 root candidate
            x N hidden-world rollouts
            x H5-vs-H5 all the way to terminal
            x state clone on every simulated action

    H8 keeps the useful parts:
        - H5 root candidate set
        - common hidden-world determinizations
        - H5 controls both players in the rollout
        - rollout values rank root actions
        - soft policy can still be generated from rollout values

    But H8 speeds this up in three ways:

        1. TRUNCATED ROLLOUT
           Stop after a configurable number of MAIN_DECISION moves
           instead of always playing to terminal.

        2. LEAF BOOTSTRAP VALUE
           If the truncated rollout is not terminal, estimate the
           position with a cheap static H5-informed value function.

        3. IN-PLACE PRIVATE ROLLOUT STATE
           Clone the sampled root state once per branch, then mutate
           only that private state. H6 cloned the whole GameState on
           every simulated action.

        4. SMALL H5 ACTION CACHE
           Cache H5's selected action for identical PUBLIC decision
           states encountered across hidden-world samples. The key
           intentionally ignores hidden deck order because H5's
           action choice does not inspect that order.

    H8 does NOT:
        - prune H5 root candidates
        - recursively call H8 inside a rollout
        - inspect the true hidden deck order
        - require multiprocessing / thread-safety

    This makes H8 a clean speed/quality experiment against H6.

    Confirmed environment API:

        env.step(action, state)

    Designed for 2-player Splendor.
    """

    def __init__(
        self,
        num_rollouts=8,
        rollout_decisions=8,
        max_rollout_steps=80,
        policy_temperature=0.25,
        random_seed=None,
        use_action_cache=True,
    ):
        self.num_rollouts = max(
            1,
            int(num_rollouts),
        )

        self.rollout_decisions = max(
            0,
            int(rollout_decisions),
        )

        self.max_rollout_steps = max(
            1,
            int(max_rollout_steps),
        )

        self.policy_temperature = float(
            policy_temperature
        )

        self.rng = random.Random(
            random_seed
        )

        self.use_action_cache = bool(
            use_action_cache
        )

        self._h5_action_cache = {}
        self._leaf_value_cache = {}

        self.last_rollout_stats = {}

    # ============================================================
    # PUBLIC API
    # ============================================================

    def select_action(
        self,
        env,
        state,
    ):
        evaluated = (
            self._get_rollout_evaluated_candidates(
                env,
                state,
            )
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
        temperature=None,
    ):
        """
        Soft policy over rollout-evaluated H5 candidates.

        Values are approximately in [-1, 1].

        This preserves the useful teacher signal instead of
        collapsing every state to a one-hot chosen move.
        """

        if temperature is None:
            temperature = (
                self.policy_temperature
            )

        if temperature <= 0:
            raise ValueError(
                "temperature must be greater than 0"
            )

        evaluated = (
            self._get_rollout_evaluated_candidates(
                env,
                state,
            )
        )

        policy = np.zeros(
            action_size,
            dtype=np.float32,
        )

        if not evaluated:
            return policy

        actions = [
            action
            for action, _, _
            in evaluated
        ]

        values = np.array(
            [
                rollout_value
                for _, rollout_value, _
                in evaluated
            ],
            dtype=np.float64,
        )

        if not np.any(
            np.isfinite(values)
        ):
            probs = np.ones(
                len(actions),
                dtype=np.float64,
            )
            probs /= probs.sum()

        else:
            values = np.where(
                np.isfinite(values),
                values,
                -1e9,
            )

            logits = (
                values
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

        # Caches are intentionally local to one root search.
        self._h5_action_cache = {}
        self._leaf_value_cache = {}

        stats = {
            "root_candidates": 0,
            "rollouts": 0,
            "env_steps": 0,
            "main_decisions": 0,
            "terminal_rollouts": 0,
            "leaf_evaluations": 0,
            "action_cache_hits": 0,
            "action_cache_misses": 0,
            "leaf_cache_hits": 0,
            "leaf_cache_misses": 0,
        }

        self._active_stats = stats

        try:
            h5_candidates = (
                HeuristicAgent5._get_scored_candidates(
                    self,
                    env,
                    state,
                )
            )

            if not h5_candidates:
                self.last_rollout_stats = stats
                return []

            stats["root_candidates"] = (
                len(h5_candidates)
            )

            # No counterfactual comparison is needed.
            if len(h5_candidates) == 1:
                action, h5_score = (
                    h5_candidates[0]
                )

                self.last_rollout_stats = stats

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

            # ----------------------------------------------------
            # Common random numbers:
            # every root candidate sees the same hidden worlds.
            # ----------------------------------------------------

            sampled_worlds = (
                self._sample_hidden_worlds(
                    state
                )
            )

            evaluated = []

            for action, h5_score in (
                h5_candidates
            ):
                rollout_values = []

                for sampled_state in (
                    sampled_worlds
                ):
                    value = (
                        self._rollout_from_action(
                            env=env,
                            sampled_state=sampled_state,
                            first_action=action,
                            root_player=root_player,
                        )
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

            self.last_rollout_stats = (
                dict(stats)
            )

            return evaluated

        finally:
            # Keep a copy for inspection but do not leave a stale
            # mutable stats reference active.
            if not self.last_rollout_stats:
                self.last_rollout_stats = (
                    dict(stats)
                )

            self._active_stats = None

    # ============================================================
    # TRUNCATED H5 ROLLOUT
    # ============================================================

    def _rollout_from_action(
        self,
        env,
        sampled_state,
        first_action,
        root_player,
    ):
        """
        Play the candidate action, then let H5 control both players.

        Unlike H6, stop after rollout_decisions MAIN_DECISION moves
        and bootstrap from a static leaf value.

        Forced DISCARD / NOBLE nodes do not consume strategic depth.
        """

        self._active_stats[
            "rollouts"
        ] += 1

        sim_env = env.clone()

        # One branch clone.
        #
        # From here onward H8 mutates only this private rollout
        # state in place rather than cloning at every action.
        rollout_state = (
            sampled_state.clone()
        )

        rollout_state = (
            self._step_in_place(
                sim_env,
                rollout_state,
                first_action,
            )
        )

        env_steps = 1
        main_decisions = 0

        # --------------------------------------------------------
        # Continue H5-vs-H5 for a bounded strategic horizon.
        # --------------------------------------------------------

        while (
            env_steps
            < self.max_rollout_steps
            and main_decisions
            < self.rollout_decisions
            and not sim_env._check_terminated(
                rollout_state
            )
        ):
            is_main_decision = (
                rollout_state.node_type
                == NodeType.MAIN_DECISION
            )

            action = (
                self._select_h5_action_cached(
                    sim_env,
                    rollout_state,
                )
            )

            if action is None:
                break

            rollout_state = (
                self._step_in_place(
                    sim_env,
                    rollout_state,
                    action,
                )
            )

            env_steps += 1

            if is_main_decision:
                main_decisions += 1

        self._active_stats[
            "env_steps"
        ] += env_steps

        self._active_stats[
            "main_decisions"
        ] += main_decisions

        # --------------------------------------------------------
        # Exact terminal result still takes precedence.
        # --------------------------------------------------------

        if sim_env._check_terminated(
            rollout_state
        ):
            self._active_stats[
                "terminal_rollouts"
            ] += 1

            return self._terminal_value(
                rollout_state,
                root_player,
            )

        # --------------------------------------------------------
        # Bootstrap at the cutoff.
        # --------------------------------------------------------

        self._active_stats[
            "leaf_evaluations"
        ] += 1

        return self._evaluate_leaf_state_cached(
            rollout_state,
            root_player,
        )

    # ============================================================
    # IN-PLACE PRIVATE STEP
    # ============================================================

    def _step_in_place(
        self,
        env,
        state,
        action,
    ):
        """
        Safe because every rollout owns its own cloned state.

        H6 cloned GameState here on every simulated action.
        H8 avoids that repeated deep-copy cost.
        """

        result = env.step(
            action,
            state,
        )

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

        return state

    # ============================================================
    # H5 ROLLOUT POLICY + CACHE
    # ============================================================

    def _select_h5_action_cached(
        self,
        env,
        state,
    ):
        """
        Cache H5's deterministic choice for equivalent public states.

        Hidden deck ORDER is intentionally absent from the key.
        H5 does not inspect future order when scoring the current
        move; only current public/private player state and deck
        availability matter.
        """

        if not self.use_action_cache:
            return self._select_h5_action(
                env,
                state,
            )

        key = self._state_cache_key(
            state
        )

        cached = (
            self._h5_action_cache.get(
                key
            )
        )

        if cached is not None:
            self._active_stats[
                "action_cache_hits"
            ] += 1

            return cached

        self._active_stats[
            "action_cache_misses"
        ] += 1

        action = self._select_h5_action(
            env,
            state,
        )

        if action is not None:
            self._h5_action_cache[
                key
            ] = action

        return action

    def _select_h5_action(
        self,
        env,
        state,
    ):
        """
        Explicit H5 call prevents recursive H8 rollouts.
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
            key=lambda item:
                item[1],
        )[0]

    # ============================================================
    # STATIC LEAF VALUE
    # ============================================================

    def _evaluate_leaf_state_cached(
        self,
        state,
        root_player,
    ):
        key = (
            root_player,
            self._state_cache_key(
                state
            ),
        )

        if key in self._leaf_value_cache:
            self._active_stats[
                "leaf_cache_hits"
            ] += 1

            return self._leaf_value_cache[
                key
            ]

        self._active_stats[
            "leaf_cache_misses"
        ] += 1

        value = self._evaluate_leaf_state(
            state,
            root_player,
        )

        self._leaf_value_cache[
            key
        ] = value

        return value

    def _evaluate_leaf_state(
        self,
        state,
        root_player,
    ):
        """
        Cheap H5-informed position evaluation in [-1, 1].

        It deliberately avoids another search.

        Components:
            - point lead
            - permanent-bonus / engine lead
            - best accessible card opportunity
            - noble proximity
            - small gem-resource term

        The final tanh keeps the scale compatible with terminal
        rollout values of -1 / 0 / +1.
        """

        if len(state.players) != 2:
            return 0.0

        opponent_index = (
            1 - root_player
        )

        root = state.players[
            root_player
        ]

        opponent = state.players[
            opponent_index
        ]

        root_points = getattr(
            root,
            "points",
            0,
        )

        opponent_points = getattr(
            opponent,
            "points",
            0,
        )

        # Strongest and most reliable static signal.
        point_term = (
            root_points
            - opponent_points
        ) / 6.0

        # Permanent engine.
        root_bonus_count = sum(
            root.bonuses[color]
            for color in COLOR_ORDER
        )

        opponent_bonus_count = sum(
            opponent.bonuses[color]
            for color in COLOR_ORDER
        )

        bonus_term = (
            root_bonus_count
            - opponent_bonus_count
        ) / 20.0

        # Immediate / near-term card opportunity.
        root_target = (
            self._best_card_opportunity(
                state,
                root,
            )
        )

        opponent_target = (
            self._best_card_opportunity(
                state,
                opponent,
            )
        )

        target_term = (
            root_target
            - opponent_target
        ) / 60.0

        # Noble closeness.
        root_noble = (
            self._best_noble_progress(
                state,
                root,
            )
        )

        opponent_noble = (
            self._best_noble_progress(
                state,
                opponent,
            )
        )

        noble_term = (
            root_noble
            - opponent_noble
        ) / 8.0

        # Gems matter, but much less than actual points/engine.
        root_gems = sum(
            root.gems.values()
        )

        opponent_gems = sum(
            opponent.gems.values()
        )

        gem_term = (
            root_gems
            - opponent_gems
        ) / 30.0

        raw = (
            point_term
            + bonus_term
            + target_term
            + noble_term
            + gem_term
        )

        return float(
            math.tanh(raw)
        )

    def _best_card_opportunity(
        self,
        state,
        player,
    ):
        """
        H5-flavored estimate of the player's best currently known
        card opportunity.

        Visible cards + that player's own reserved cards.
        """

        cards = []

        for tier in (1, 2, 3):
            cards.extend(
                card
                for card
                in state.visible_cards[tier]
                if card is not None
            )

        cards.extend(
            card
            for card
            in player.reserved_cards
            if card is not None
        )

        if not cards:
            return 0.0

        best = float("-inf")

        for card in cards:
            distance = self._distance_to_card(
                player,
                card,
            )

            # Reuse H5's card valuation directly.
            card_value = self._score_card(
                player,
                card,
                state.nobles,
                state=state,
            )

            opportunity = (
                card_value
                / (distance + 1)
            )

            current_points = getattr(
                player,
                "points",
                0,
            )

            # Immediate win opportunities should dominate a
            # truncated leaf.
            if (
                distance == 0
                and current_points
                + card.points
                >= 15
            ):
                opportunity += 100.0

            best = max(
                best,
                opportunity,
            )

        if best == float("-inf"):
            return 0.0

        return best

    def _best_noble_progress(
        self,
        state,
        player,
    ):
        """
        Larger is better.

        Convert each noble into:
            required bonuses - current bonuses

        and use the closest noble as a small strategic signal.
        """

        best = None

        for noble in state.nobles:
            if noble is None:
                continue

            missing = 0

            for color in COLOR_ORDER:
                missing += max(
                    0,
                    noble.requirement[color]
                    - player.bonuses[color],
                )

            if (
                best is None
                or missing < best
            ):
                best = missing

        if best is None:
            return 0.0

        # 0 missing is excellent.
        # Larger missing produces less progress value.
        return 5.0 / (
            best + 1.0
        )

    # ============================================================
    # HIDDEN-WORLD DETERMINIZATION
    # ============================================================

    def _sample_hidden_worlds(
        self,
        state,
    ):
        """
        Same common-random-number idea as H6.
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
    # CACHE KEY
    # ============================================================

    def _state_cache_key(
        self,
        state,
    ):
        """
        Lightweight structural key for H5's CURRENT action choice.

        Deliberately includes:
            - player resources / bonuses / points / reservations
            - visible market
            - nobles
            - bank
            - node type / current player
            - deck SIZES

        Deliberately excludes:
            - hidden deck ORDER

        That exclusion is what permits cache reuse across sampled
        hidden worlds before different hidden cards are revealed.
        """

        bank_key = tuple(
            self._mapping_value(
                state.bank,
                color,
            )
            for color in list(
                COLOR_ORDER
            ) + [GemColor.GOLD]
        )

        visible_key = tuple(
            self._card_key(card)
            for tier in (1, 2, 3)
            for card
            in state.visible_cards[tier]
        )

        noble_key = tuple(
            self._noble_key(noble)
            for noble in state.nobles
        )

        players_key = tuple(
            self._player_key(player)
            for player in state.players
        )

        decks_key = self._deck_sizes_key(
            state.decks
        )

        return (
            getattr(
                state.current_player,
                "value",
                state.current_player,
            ),
            getattr(
                state.node_type,
                "value",
                state.node_type,
            ),
            getattr(
                state,
                "turn_number",
                None,
            ),
            getattr(
                state,
                "end_round_start",
                None,
            ),
            getattr(
                state,
                "max_gems",
                None,
            ),
            bank_key,
            visible_key,
            noble_key,
            players_key,
            decks_key,
        )

    def _player_key(
        self,
        player,
    ):
        gems_key = tuple(
            self._mapping_value(
                player.gems,
                color,
            )
            for color in list(
                COLOR_ORDER
            ) + [GemColor.GOLD]
        )

        bonuses_key = tuple(
            self._mapping_value(
                player.bonuses,
                color,
            )
            for color in COLOR_ORDER
        )

        reserved_key = tuple(
            self._card_key(card)
            for card in player.reserved_cards
        )

        purchased_count = None

        for attr in (
            "purchased_cards",
            "cards",
        ):
            value = getattr(
                player,
                attr,
                None,
            )

            if value is not None:
                try:
                    purchased_count = len(
                        value
                    )
                except TypeError:
                    purchased_count = None

                if purchased_count is not None:
                    break

        return (
            getattr(
                player,
                "points",
                0,
            ),
            gems_key,
            bonuses_key,
            reserved_key,
            purchased_count,
        )

    def _card_key(
        self,
        card,
    ):
        if card is None:
            return None

        cost_key = tuple(
            self._mapping_value(
                card.cost,
                color,
            )
            for color in COLOR_ORDER
        )

        bonus_color = getattr(
            card,
            "bonus_color",
            None,
        )

        return (
            getattr(
                card,
                "points",
                0,
            ),
            getattr(
                bonus_color,
                "value",
                bonus_color,
            ),
            cost_key,
        )

    def _noble_key(
        self,
        noble,
    ):
        if noble is None:
            return None

        requirement = getattr(
            noble,
            "requirement",
            {},
        )

        return (
            getattr(
                noble,
                "points",
                3,
            ),
            tuple(
                self._mapping_value(
                    requirement,
                    color,
                )
                for color in COLOR_ORDER
            ),
        )

    def _deck_sizes_key(
        self,
        decks,
    ):
        if isinstance(
            decks,
            dict,
        ):
            return tuple(
                (
                    tier,
                    len(decks[tier]),
                )
                for tier in sorted(
                    decks.keys()
                )
            )

        return tuple(
            len(deck)
            for deck in decks
        )

    def _mapping_value(
        self,
        mapping,
        key,
    ):
        try:
            return mapping.get(
                key,
                0,
            )
        except AttributeError:
            try:
                return mapping[key]
            except Exception:
                return 0

    # ============================================================
    # TERMINAL RESULT
    # ============================================================

    def _terminal_value(
        self,
        state,
        root_player,
    ):
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

    # ============================================================
    # DEBUG / PROFILING
    # ============================================================

    def get_last_rollout_stats(
        self,
    ):
        """
        Example after select_action/get_policy:

            {
                "root_candidates": 7,
                "rollouts": 56,
                "env_steps": 490,
                "main_decisions": 430,
                ...
            }

        Useful for comparing H8 depth/runtime against H6.
        """

        return dict(
            self.last_rollout_stats
        )
