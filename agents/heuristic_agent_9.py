import random
import numpy as np

from splendor_v1.agents.heuristic_agent_3 import HeuristicAgent3
from splendor_v1.env.core.constants import COLOR_ORDER
from splendor_v1.env.core.enums import GemColor, NodeType
from splendor_v1.env.core.actions import ActionType


class HeuristicAgent9:
    """
    HeuristicAgent9
    =================

    Phase-aware broad-root search + cheap H3 terminal rollouts.

    The design goal is a middle ground between H4 and H6:

        H4:
            restricted H3 root candidates
            + H3 full terminal rollout

        H6:
            broader / opponent-aware H5 root candidates
            + expensive H5 full terminal rollout

        H9:
            broader PHASE-AWARE root candidate selection
            + cheap H3 full terminal rollout

    H9 does NOT try to decide the move with the phase heuristic.
    The phase heuristic only decides which moves deserve expensive
    rollout calculation.

    Pipeline:

        all legal actions
            ↓
        cheap H3-style action scoring
            ↓
        detect EARLY / MID / LATE phase
            ↓
        phase-aware BUY / TAKE / RESERVE emphasis
            ↓
        remove clearly inferior actions within each category
            ↓
        waterfall allocation up to num_calc_moves
            ↓
        same sampled hidden worlds for every candidate
            ↓
        H3 vs H3 all the way to terminal
            ↓
        choose best average terminal result

    Important properties
    --------------------

    1. num_calc_moves limits expensive root calculations.

       Example:
           22 plausible opening actions
           num_calc_moves = 8

       Only up to 8 normal candidates receive terminal rollouts.

    2. Allocation is NOT a hard quota.

       Example early-game target:
           BUY:     4
           TAKE:    3
           RESERVE: 1

       If only 3 BUY actions survive the quality filter, the unused
       slot rolls into TAKE / RESERVE / best remaining candidates.

    3. BUY includes BOTH:
           BUY_VISIBLE
           BUY_RESERVED

       Reserved-card purchases are never ignored.

    4. Phase affects candidate emphasis, not final evaluation.

       EARLY:
           Tier 1 BUY emphasis
           TAKE emphasis
           Tier 2/3 reserve is excluded by default

       MID:
           Tier 2 BUY emphasis
           useful Tier 1 / Tier 3 buys remain possible

       LATE:
           Tier 2 / Tier 3 BUY emphasis
           more tactical RESERVE capacity

    5. Candidate quality uses a RELATIVE threshold.

       An action is considered plausible if:

           score >= best_score_in_group - candidate_score_tolerance

       This avoids a brittle global cutoff like "score >= 20".

    6. Immediate threshold-crossing BUY actions are always preserved,
       even if they would otherwise fall outside num_calc_moves.

       The rollout still decides whether ending the round from that
       position actually wins.

    7. Continuation policy is explicitly HeuristicAgent3.

       H9 never recursively calls itself during rollouts.

    Environment API expected:

        env.step(action, state)

    Intended for 2-player Splendor, but the H3 continuation itself
    can operate normally if the environment supports more players.
    """

    EARLY = "early"
    MID = "mid"
    LATE = "late"

    WIN_POINTS = 15

    def __init__(
        self,
        num_rollouts=8,
        num_calc_moves=8,
        candidate_score_tolerance=8.0,
        max_rollout_steps=200,
        random_seed=None,

        # Phase thresholds.
        early_points_max=3,
        early_bonuses_max=4,
        late_points_min=10,

        # Whether to spend unused rollout slots on actions that were
        # below the plausibility threshold.
        fill_budget_with_low_score=False,
    ):
        self.num_rollouts = max(
            1,
            int(num_rollouts),
        )

        self.num_calc_moves = max(
            1,
            int(num_calc_moves),
        )

        self.candidate_score_tolerance = float(
            candidate_score_tolerance
        )

        self.max_rollout_steps = max(
            1,
            int(max_rollout_steps),
        )

        self.rng = random.Random(
            random_seed
        )

        self.early_points_max = int(
            early_points_max
        )

        self.early_bonuses_max = int(
            early_bonuses_max
        )

        self.late_points_min = int(
            late_points_min
        )

        self.fill_budget_with_low_score = bool(
            fill_budget_with_low_score
        )

        # Cheap continuation policy.
        self.h3 = HeuristicAgent3()

        # Debug information from most recent root calculation.
        self.last_candidate_debug = None
        self.last_rollout_debug = None

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

        # Terminal result is primary.
        # Cheap candidate score only breaks rollout ties.
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
        Soft policy over rollout-evaluated root candidates.

        Rollout values live approximately in [-1, 1], so a much
        smaller temperature than the raw heuristic policy is useful.
        """

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

        rollout_values = np.array(
            [
                rollout_value
                for _, rollout_value, _
                in evaluated
            ],
            dtype=np.float64,
        )

        if not np.any(
            np.isfinite(
                rollout_values
            )
        ):
            probs = np.ones(
                len(actions),
                dtype=np.float64,
            )

            probs /= probs.sum()

        else:
            rollout_values = np.where(
                np.isfinite(
                    rollout_values
                ),
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
    # PHASE DETECTION
    # ============================================================

    def _get_game_phase(
        self,
        state,
    ):
        """
        State-based rather than turn-based.

        We use the most advanced player because if the opponent is
        already near 15, the game is strategically late even if our
        own score is still modest.
        """

        max_points = max(
            getattr(
                player,
                "points",
                0,
            )
            for player in state.players
        )

        max_bonuses = max(
            sum(
                player.bonuses[color]
                for color in COLOR_ORDER
            )
            for player in state.players
        )

        if (
            max_points
            >= self.late_points_min
        ):
            return self.LATE

        if (
            max_points
            <= self.early_points_max
            and max_bonuses
            <= self.early_bonuses_max
        ):
            return self.EARLY

        return self.MID

    # ============================================================
    # ROOT CANDIDATE GENERATION
    # ============================================================

    def _get_root_candidates(
        self,
        env,
        state,
    ):
        """
        Return:
            [
                (
                    action,
                    cheap_selection_score,
                ),
                ...
            ]

        Only these candidates receive full terminal rollouts.
        """

        legal_actions = env._legal_actions(
            state
        )

        if not legal_actions:
            return []

        # Forced transition nodes are not strategic root choices.
        # Let H3 handle them directly.
        if state.node_type != NodeType.MAIN_DECISION:
            h3_scored = (
                self.h3._get_scored_candidates(
                    env,
                    state,
                )
            )

            self.last_candidate_debug = {
                "phase": None,
                "legal_count": len(
                    legal_actions
                ),
                "selected_count": len(
                    h3_scored
                ),
                "selected": h3_scored,
                "reason": "forced_node",
            }

            return h3_scored

        phase = self._get_game_phase(
            state
        )

        records = []

        for action in legal_actions:
            record = self._make_action_record(
                state=state,
                action=action,
                phase=phase,
            )

            if record is not None:
                records.append(
                    record
                )

        if not records:
            return []

        # --------------------------------------------------------
        # Immediate threshold-crossing BUY actions are mandatory.
        #
        # Reaching 15 does not always guarantee final victory because
        # Splendor finishes the round, so these still go through the
        # same terminal rollout comparison.
        # --------------------------------------------------------

        mandatory = [
            record
            for record in records
            if record[
                "mandatory"
            ]
        ]

        selected_ids = {
            self._action_identity(
                record["action"]
            )
            for record in mandatory
        }

        selected = list(
            mandatory
        )

        # If mandatory actions already exceed the normal budget,
        # keep all of them.
        normal_budget = max(
            0,
            self.num_calc_moves
            - len(selected),
        )

        # --------------------------------------------------------
        # Build group-specific plausible sets.
        #
        # Relative threshold is applied independently to BUY,
        # TAKE, and RESERVE so their different heuristic scales do
        # not need to be globally calibrated.
        # --------------------------------------------------------

        groups = {
            "buy": [],
            "take": [],
            "reserve": [],
            "other": [],
        }

        for record in records:
            action_id = self._action_identity(
                record["action"]
            )

            if action_id in selected_ids:
                continue

            groups[
                record["group"]
            ].append(
                record
            )

        plausible = {}

        for group_name, group_records in (
            groups.items()
        ):
            plausible[group_name] = (
                self._filter_plausible_group(
                    group_records
                )
            )

        # --------------------------------------------------------
        # Phase allocation targets.
        #
        # These are SOFT targets.
        # Any unfilled capacity rolls over.
        # --------------------------------------------------------

        targets = self._phase_targets(
            phase
        )

        group_order = self._phase_group_order(
            phase
        )

        # First pass: give each strategic group its preferred share.
        for group_name in group_order:
            if normal_budget <= 0:
                break

            target = targets.get(
                group_name,
                0,
            )

            if target <= 0:
                continue

            candidates = [
                record
                for record
                in plausible[
                    group_name
                ]
                if self._action_identity(
                    record["action"]
                )
                not in selected_ids
            ]

            candidates.sort(
                key=lambda record:
                    record[
                        "selection_score"
                    ],
                reverse=True,
            )

            add_count = min(
                target,
                normal_budget,
                len(candidates),
            )

            for record in candidates[
                :add_count
            ]:
                selected.append(
                    record
                )

                selected_ids.add(
                    self._action_identity(
                        record["action"]
                    )
                )

            normal_budget -= (
                add_count
            )

        # --------------------------------------------------------
        # Second pass: WATERFALL / ROLLOVER.
        #
        # If BUY only supplied 3 good actions, the unused BUY slots
        # are not lost. Fill them with the strongest remaining
        # plausible TAKE / RESERVE / BUY candidates.
        # --------------------------------------------------------

        if normal_budget > 0:
            leftovers = []

            for group_name in (
                "buy",
                "take",
                "reserve",
                "other",
            ):
                for record in plausible[
                    group_name
                ]:
                    if self._action_identity(
                        record["action"]
                    ) in selected_ids:
                        continue

                    leftovers.append(
                        record
                    )

            leftovers.sort(
                key=lambda record:
                    record[
                        "selection_score"
                    ],
                reverse=True,
            )

            for record in leftovers:
                if normal_budget <= 0:
                    break

                selected.append(
                    record
                )

                selected_ids.add(
                    self._action_identity(
                        record["action"]
                    )
                )

                normal_budget -= 1

        # --------------------------------------------------------
        # Optional third pass:
        # use clearly weaker actions only if the caller explicitly
        # wants to spend the whole num_calc_moves budget.
        # --------------------------------------------------------

        if (
            normal_budget > 0
            and self.fill_budget_with_low_score
        ):
            low_score_leftovers = [
                record
                for record in records
                if (
                    not record["mandatory"]
                    and self._action_identity(
                        record["action"]
                    )
                    not in selected_ids
                )
            ]

            low_score_leftovers.sort(
                key=lambda record:
                    record[
                        "selection_score"
                    ],
                reverse=True,
            )

            for record in low_score_leftovers:
                if normal_budget <= 0:
                    break

                selected.append(
                    record
                )

                selected_ids.add(
                    self._action_identity(
                        record["action"]
                    )
                )

                normal_budget -= 1

        # --------------------------------------------------------
        # Debug snapshot.
        # --------------------------------------------------------

        self.last_candidate_debug = {
            "phase": phase,
            "legal_count": len(
                legal_actions
            ),
            "record_count": len(
                records
            ),
            "mandatory_count": len(
                mandatory
            ),
            "selected_count": len(
                selected
            ),
            "num_calc_moves": (
                self.num_calc_moves
            ),
            "candidate_score_tolerance": (
                self.candidate_score_tolerance
            ),
            "targets": dict(
                targets
            ),
            "plausible_counts": {
                group_name: len(
                    group_records
                )
                for group_name, group_records
                in plausible.items()
            },
            "selected": [
                {
                    "action":
                        record["action"],
                    "group":
                        record["group"],
                    "tier":
                        record["tier"],
                    "base_score":
                        record["base_score"],
                    "phase_bonus":
                        record["phase_bonus"],
                    "selection_score":
                        record["selection_score"],
                    "mandatory":
                        record["mandatory"],
                }
                for record in selected
            ],
        }

        return [
            (
                record["action"],
                record[
                    "selection_score"
                ],
            )
            for record in selected
        ]

    # ============================================================
    # ACTION RECORD / CHEAP SCORING
    # ============================================================

    def _make_action_record(
        self,
        state,
        action,
        phase,
    ):
        group = self._action_group(
            action
        )

        tier = self._action_tier(
            state,
            action,
        )

        # Early-game design choice:
        # do not spend root rollout budget reserving Tier 2 / Tier 3.
        if (
            phase == self.EARLY
            and group == "reserve"
            and tier in (2, 3)
        ):
            return None

        base_score = (
            self._cheap_h3_action_score(
                state,
                action,
            )
        )

        if not np.isfinite(
            base_score
        ):
            return None

        phase_bonus = (
            self._phase_bonus(
                phase=phase,
                group=group,
                tier=tier,
                action=action,
            )
        )

        selection_score = (
            base_score
            + phase_bonus
        )

        mandatory = (
            self._is_threshold_crossing_buy(
                state,
                action,
            )
        )

        return {
            "action":
                action,

            "group":
                group,

            "tier":
                tier,

            "base_score":
                float(
                    base_score
                ),

            "phase_bonus":
                float(
                    phase_bonus
                ),

            "selection_score":
                float(
                    selection_score
                ),

            "mandatory":
                mandatory,
        }

    def _cheap_h3_action_score(
        self,
        state,
        action,
    ):
        """
        Prefer H3's own scoring methods if exposed by the current
        project version.

        Fallbacks reproduce the H3 logic discussed during design.
        """

        action_type = (
            action.action_type
        )

        if action_type in (
            ActionType.BUY_VISIBLE,
            ActionType.BUY_RESERVED,
        ):
            method = getattr(
                self.h3,
                "_score_buy_action",
                None,
            )

            if method is not None:
                return method(
                    state,
                    action,
                )

            return self._fallback_buy_score(
                state,
                action,
            )

        if (
            action_type
            == ActionType.TAKE_GEMS
        ):
            method = getattr(
                self.h3,
                "_score_take_gems",
                None,
            )

            if method is not None:
                return method(
                    state,
                    action,
                )

            return self._fallback_take_score(
                state,
                action,
            )

        if action_type in (
            ActionType.RESERVE_VISIBLE,
            ActionType.RESERVE_TOP_DECK,
        ):
            method = getattr(
                self.h3,
                "_score_reserve_action",
                None,
            )

            if method is not None:
                return method(
                    state,
                    action,
                )

            return self._fallback_reserve_score(
                state,
                action,
            )

        # Forced / special transitions.
        return 0.0

    # ============================================================
    # PHASE BIASES
    # ============================================================

    def _phase_targets(
        self,
        phase,
    ):
        """
        Soft initial allocation for num_calc_moves.

        Rollover fills unused slots afterward.
        """

        if phase == self.EARLY:
            return {
                "buy": 4,
                "take": 3,
                "reserve": 1,
            }

        if phase == self.MID:
            return {
                "buy": 5,
                "take": 2,
                "reserve": 1,
            }

        # LATE
        return {
            "buy": 5,
            "take": 1,
            "reserve": 2,
        }

    def _phase_group_order(
        self,
        phase,
    ):
        if phase == self.LATE:
            return [
                "buy",
                "reserve",
                "take",
                "other",
            ]

        return [
            "buy",
            "take",
            "reserve",
            "other",
        ]

    def _phase_bonus(
        self,
        phase,
        group,
        tier,
        action,
    ):
        """
        Phase bias is used ONLY to decide which moves receive
        rollout calculation.

        It does not directly determine the selected game action.
        """

        if group == "buy":

            # Reserved buys remain valid in every phase.
            # If original tier is unavailable, tier=None receives
            # no tier bonus but keeps its H3 score.
            if phase == self.EARLY:
                if tier == 1:
                    return 8.0
                if tier == 2:
                    return 1.0
                if tier == 3:
                    return -6.0
                return 0.0

            if phase == self.MID:
                if tier == 1:
                    return 1.0
                if tier == 2:
                    return 8.0
                if tier == 3:
                    return 3.0
                return 2.0

            # LATE
            if tier == 1:
                return -4.0
            if tier == 2:
                return 5.0
            if tier == 3:
                return 10.0
            return 4.0

        if group == "take":
            if phase == self.EARLY:
                return 6.0

            if phase == self.MID:
                return 4.0

            return 1.0

        if group == "reserve":
            # RESERVE_TOP_DECK usually has tier=None.
            if phase == self.EARLY:
                if tier == 1:
                    return 2.0
                return -4.0

            if phase == self.MID:
                if tier == 1:
                    return 0.0
                if tier == 2:
                    return 3.0
                if tier == 3:
                    return 1.0
                return -2.0

            # LATE
            if tier == 1:
                return -2.0
            if tier == 2:
                return 2.0
            if tier == 3:
                return 5.0
            return -1.0

        return 0.0

    # ============================================================
    # RELATIVE QUALITY FILTER
    # ============================================================

    def _filter_plausible_group(
        self,
        records,
    ):
        """
        Keep actions reasonably close to the best action in their
        own broad category.

        This avoids comparing BUY score scale directly to TAKE or
        RESERVE score scale.
        """

        if not records:
            return []

        best_score = max(
            record[
                "selection_score"
            ]
            for record in records
        )

        cutoff = (
            best_score
            - self.candidate_score_tolerance
        )

        plausible = [
            record
            for record in records
            if (
                record[
                    "selection_score"
                ]
                >= cutoff
            )
        ]

        plausible.sort(
            key=lambda record:
                record[
                    "selection_score"
                ],
            reverse=True,
        )

        return plausible

    # ============================================================
    # ROLLOUT EVALUATION
    # ============================================================

    def _get_rollout_evaluated_candidates(
        self,
        env,
        state,
    ):
        candidates = (
            self._get_root_candidates(
                env,
                state,
            )
        )

        if not candidates:
            return []

        if len(candidates) == 1:
            action, score = candidates[0]

            self.last_rollout_debug = {
                "candidate_count": 1,
                "rollout_count": 0,
            }

            return [
                (
                    action,
                    0.0,
                    score,
                )
            ]

        root_player = (
            state.current_player
        )

        # Common random numbers:
        # every candidate sees exactly the same hidden worlds.
        sampled_worlds = (
            self._sample_hidden_worlds(
                state
            )
        )

        evaluated = []

        total_rollouts = 0

        for action, selection_score in (
            candidates
        ):
            values = []

            for sampled_state in (
                sampled_worlds
            ):
                value = self._rollout_from_action(
                    env=env,
                    sampled_state=sampled_state,
                    first_action=action,
                    root_player=root_player,
                )

                values.append(
                    value
                )

                total_rollouts += 1

            mean_value = float(
                np.mean(
                    values
                )
            )

            evaluated.append(
                (
                    action,
                    mean_value,
                    selection_score,
                )
            )

        self.last_rollout_debug = {
            "candidate_count": len(
                candidates
            ),
            "num_rollouts_per_candidate": (
                self.num_rollouts
            ),
            "total_terminal_rollouts": (
                total_rollouts
            ),
            "evaluated": [
                {
                    "action":
                        action,
                    "mean_rollout_value":
                        value,
                    "selection_score":
                        score,
                }
                for action, value, score
                in evaluated
            ],
        }

        return evaluated

    # ============================================================
    # FULL TERMINAL H3 ROLLOUT
    # ============================================================

    def _rollout_from_action(
        self,
        env,
        sampled_state,
        first_action,
        root_player,
    ):
        """
        First root action may come from BUY / TAKE / RESERVE.

        After that, both players use ordinary H3 all the way to
        terminal. This preserves the cheap/full-terminal behavior
        that made H4 strong.
        """

        sim_env = env.clone()

        # One private branch clone.
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

        steps = 1

        while (
            steps
            < self.max_rollout_steps
            and not sim_env._check_terminated(
                rollout_state
            )
        ):
            action = self.h3.select_action(
                sim_env,
                rollout_state,
            )

            if action is None:
                return 0.0

            rollout_state = (
                self._step_in_place(
                    sim_env,
                    rollout_state,
                    action,
                )
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
    # HIDDEN-WORLD SAMPLING
    # ============================================================

    def _sample_hidden_worlds(
        self,
        state,
    ):
        worlds = []

        for _ in range(
            self.num_rollouts
        ):
            sampled = state.clone()

            self._shuffle_hidden_decks(
                sampled
            )

            worlds.append(
                sampled
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
    # PRIVATE BRANCH STEP
    # ============================================================

    def _step_in_place(
        self,
        env,
        state,
        action,
    ):
        """
        Every rollout owns its own state clone, so we can mutate
        that branch in place rather than deep-copying every step.
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
    # TERMINAL VALUE
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
    # ACTION CLASSIFICATION
    # ============================================================

    def _action_group(
        self,
        action,
    ):
        if action.action_type in (
            ActionType.BUY_VISIBLE,
            ActionType.BUY_RESERVED,
        ):
            return "buy"

        if (
            action.action_type
            == ActionType.TAKE_GEMS
        ):
            return "take"

        if action.action_type in (
            ActionType.RESERVE_VISIBLE,
            ActionType.RESERVE_TOP_DECK,
        ):
            return "reserve"

        return "other"

    def _action_tier(
        self,
        state,
        action,
    ):
        """
        Visible actions expose action.tier.

        For BUY_RESERVED, try public card metadata if the project
        stores original tier. If not available, return None and let
        the H3 score determine its importance.
        """

        tier = getattr(
            action,
            "tier",
            None,
        )

        if tier in (
            1,
            2,
            3,
        ):
            return tier

        card = self._get_card_from_action(
            state,
            action,
        )

        if card is None:
            return None

        for attr in (
            "tier",
            "level",
        ):
            value = getattr(
                card,
                attr,
                None,
            )

            if value in (
                1,
                2,
                3,
            ):
                return value

        return None

    def _action_identity(
        self,
        action,
    ):
        """
        Prefer action's own hash when available.

        Repr fallback keeps this utility independent of the
        project's action-id encoder.
        """

        try:
            hash(action)
            return action
        except TypeError:
            return repr(
                action
            )

    # ============================================================
    # THRESHOLD-CROSSING BUY
    # ============================================================

    def _is_threshold_crossing_buy(
        self,
        state,
        action,
    ):
        if action.action_type not in (
            ActionType.BUY_VISIBLE,
            ActionType.BUY_RESERVED,
        ):
            return False

        player = state.players[
            state.current_player
        ]

        card = self._get_card_from_action(
            state,
            action,
        )

        if card is None:
            return False

        projected_points = (
            getattr(
                player,
                "points",
                0,
            )
            + card.points
        )

        projected_points += (
            self._noble_points_after_card(
                player,
                card,
                state.nobles,
            )
        )

        return (
            projected_points
            >= self.WIN_POINTS
        )

    def _noble_points_after_card(
        self,
        player,
        card,
        nobles,
    ):
        best_points = 0

        bonus_color = (
            card.bonus_color
        )

        for noble in nobles:
            if noble is None:
                continue

            qualifies = True

            for color in COLOR_ORDER:
                bonus_count = (
                    player.bonuses[color]
                )

                if color == bonus_color:
                    bonus_count += 1

                if (
                    bonus_count
                    < noble.requirement[color]
                ):
                    qualifies = False
                    break

            if qualifies:
                best_points = max(
                    best_points,
                    getattr(
                        noble,
                        "points",
                        3,
                    ),
                )

        return best_points

    # ============================================================
    # CARD / DISTANCE HELPERS
    # ============================================================

    def _get_card_from_action(
        self,
        state,
        action,
    ):
        # Use H3 helper if available.
        method = getattr(
            self.h3,
            "_get_card_from_action",
            None,
        )

        if method is not None:
            return method(
                state,
                action,
            )

        player = state.players[
            state.current_player
        ]

        if action.action_type in (
            ActionType.BUY_VISIBLE,
            ActionType.RESERVE_VISIBLE,
        ):
            return state.visible_cards[
                action.tier
            ][
                action.slot
            ]

        if (
            action.action_type
            == ActionType.BUY_RESERVED
        ):
            return player.reserved_cards[
                action.reserved_index
            ]

        return None

    def _distance_to_card(
        self,
        player,
        card,
        gems=None,
    ):
        method = getattr(
            self.h3,
            "_distance_to_card",
            None,
        )

        if method is not None:
            if gems is None:
                return method(
                    player,
                    card,
                )

            return method(
                player,
                card,
                gems,
            )

        if gems is None:
            gems = player.gems

        missing = 0

        for color in COLOR_ORDER:
            required = max(
                0,
                card.cost.get(
                    color,
                    0,
                )
                - player.bonuses[color]
            )

            missing += max(
                0,
                required
                - gems[color],
            )

        missing = max(
            0,
            missing
            - gems[GemColor.GOLD],
        )

        return missing

    def _score_card_h3(
        self,
        player,
        card,
        nobles,
    ):
        method = getattr(
            self.h3,
            "_score_card",
            None,
        )

        if method is not None:
            return method(
                player,
                card,
                nobles,
            )

        effective_cost = sum(
            max(
                0,
                card.cost.get(
                    color,
                    0,
                )
                - player.bonuses[color]
            )
            for color in COLOR_ORDER
        )

        score = 0.0

        score += (
            card.points
            * 10.0
        )

        score -= (
            effective_cost
            * 0.5
        )

        score += 2.0

        bonus_color = (
            card.bonus_color
        )

        for noble in nobles:
            if noble is None:
                continue

            if (
                player.bonuses[
                    bonus_color
                ]
                < noble.requirement[
                    bonus_color
                ]
            ):
                score += 2.0

        return score

    # ============================================================
    # FALLBACK H3-LIKE ROOT SCORING
    # ============================================================

    def _fallback_buy_score(
        self,
        state,
        action,
    ):
        player = state.players[
            state.current_player
        ]

        card = self._get_card_from_action(
            state,
            action,
        )

        if card is None:
            return float(
                "-inf"
            )

        score = self._score_card_h3(
            player,
            card,
            state.nobles,
        )

        gold_payment = getattr(
            action,
            "gold_payment",
            None,
        )

        if gold_payment is not None:
            score -= sum(
                gold_payment
            )

        return score

    def _fallback_take_score(
        self,
        state,
        action,
    ):
        player = state.players[
            state.current_player
        ]

        gems_after = dict(
            player.gems
        )

        for color in action.gem_colors:
            gems_after[color] += 1

        cards = []

        for tier in (
            1,
            2,
            3,
        ):
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

        best_score = float(
            "-inf"
        )

        for card in cards:
            before = (
                self._distance_to_card(
                    player,
                    card,
                )
            )

            after = (
                self._distance_to_card(
                    player,
                    card,
                    gems_after,
                )
            )

            progress = (
                before
                - after
            )

            card_value = (
                self._score_card_h3(
                    player,
                    card,
                    state.nobles,
                )
            )

            score = (
                progress
                * 10.0
                + card_value
                / (
                    after + 1
                )
            )

            best_score = max(
                best_score,
                score,
            )

        if (
            best_score
            == float(
                "-inf"
            )
        ):
            return 0.0

        return best_score

    def _fallback_reserve_score(
        self,
        state,
        action,
    ):
        if (
            action.action_type
            == ActionType.RESERVE_TOP_DECK
        ):
            return -5.0

        player = state.players[
            state.current_player
        ]

        card = self._get_card_from_action(
            state,
            action,
        )

        if card is None:
            return float(
                "-inf"
            )

        card_score = (
            self._score_card_h3(
                player,
                card,
                state.nobles,
            )
        )

        distance = (
            self._distance_to_card(
                player,
                card,
            )
        )

        return (
            card_score
            - distance * 2.0
        )

    # ============================================================
    # DEBUG HELPERS
    # ============================================================

    def get_candidate_debug(
        self,
    ):
        return self.last_candidate_debug

    def get_rollout_debug(
        self,
    ):
        return self.last_rollout_debug
