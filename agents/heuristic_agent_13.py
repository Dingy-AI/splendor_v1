from collections import Counter

from splendor_v1.agents.heuristic_agent_12 import HeuristicAgent12
from splendor_v1.env.core.enums import NodeType


class HeuristicAgent13(HeuristicAgent12):
    """
    HeuristicAgent13
    =================

    Reserve-default / Noble-exception version of H12.

    H13 keeps the strongest architectural ideas from H12:

        1. Protect every H4/H3 root candidate.
        2. Add only a small number of strategic alternatives.
        3. Evaluate EVERY candidate with full H3-vs-H3 terminal rollouts.
        4. Never let the handcrafted strategy directly choose the move.

    The major change is strategy selection.

    H12:
        compare a Noble score against a Reserve score.

    H13:
        Reserve / high-point is the DEFAULT strategic lens.
        Noble / Kangaroo is a SPECIFIC board-state exception.

    This avoids requiring two handcrafted strategy scores to be
    calibrated onto the same numerical scale.

    ------------------------------------------------------------
    Default: RESERVE / HIGH-POINT
    ------------------------------------------------------------

    Unless the board clearly supports a two-noble route, H13 adds
    the reserve/high-point alternatives inherited from H12:

        - reserve valuable Tier 2 / Tier 3 cards
        - take gems toward high-point targets
        - preserve H4's original candidates

    "Reserve strategy" does NOT mean H13 must reserve.
    It only determines which extra candidates are allowed to
    challenge H4.

    ------------------------------------------------------------
    Exception: NOBLE / KANGAROO
    ------------------------------------------------------------

    H13 enters Noble mode only when:

        - there is a target PAIR of nobles,
        - the pair shares at least noble_min_overlap colors,
        - the game is not already too late,
        - AND one of several concrete board conditions is true.

    Noble exception triggers:

    A. Very strong overlap
        >= 3 shared noble colors
        AND at least 1 visible Tier-1 card in a shared color.

    B. Strong opening / board support
        >= 2 shared noble colors
        AND at least 2 visible Tier-1 cards in shared colors.

    C. Established noble route
        >= 2 shared noble colors
        AND current permanent bonuses are already close enough
            to BOTH selected nobles.

    These thresholds are configurable.

    ------------------------------------------------------------
    Example
    ------------------------------------------------------------

    Noble A:
        blue 3, red 3, green 3

    Noble B:
        blue 3, red 3, black 3

    Shared colors:
        blue, red

    If the board also contains two useful Tier-1 blue/red bonus
    cards, H13 can classify this as a Noble exception.

    Otherwise, Reserve remains the strategic default.

    ------------------------------------------------------------
    Rollout behavior
    ------------------------------------------------------------

    H13 inherits H12's full H3 terminal rollout.

    Thus:

        board-state strategy
            ↓
        proposes extra moves
            ↓
        H4 base moves + strategic extras
            ↓
        full terminal rollouts
            ↓
        best rollout result wins

    The strategy layer NEVER directly chooses the final action.
    """

    def __init__(
        self,
        num_rollouts=8,
        num_strategy_moves=2,
        max_rollout_steps=200,
        random_seed=None,

        # Noble exception requirements.
        noble_min_overlap=2,
        noble_max_points=9,

        # Trigger A:
        # extremely overlapping nobles need only one visible
        # shared-color Tier-1 card.
        noble_very_strong_overlap=3,
        noble_very_strong_min_shared_tier1=1,

        # Trigger B:
        # normal 2-color overlap needs stronger visible support.
        noble_board_min_shared_tier1=2,

        # Trigger C:
        # already-developed noble engine.
        noble_progress_max_total_missing=10,
        noble_progress_max_individual_missing=6,
    ):
        # H13 overrides H12's _choose_strategy entirely, so
        # strategy_margin/min_strategy_score are intentionally unused.
        super().__init__(
            num_rollouts=num_rollouts,
            num_strategy_moves=num_strategy_moves,
            strategy_margin=0.0,
            min_strategy_score=0.0,
            max_rollout_steps=max_rollout_steps,
            random_seed=random_seed,
        )

        self.noble_min_overlap = int(
            noble_min_overlap
        )

        self.noble_max_points = int(
            noble_max_points
        )

        self.noble_very_strong_overlap = int(
            noble_very_strong_overlap
        )

        self.noble_very_strong_min_shared_tier1 = int(
            noble_very_strong_min_shared_tier1
        )

        self.noble_board_min_shared_tier1 = int(
            noble_board_min_shared_tier1
        )

        self.noble_progress_max_total_missing = int(
            noble_progress_max_total_missing
        )

        self.noble_progress_max_individual_missing = int(
            noble_progress_max_individual_missing
        )

    # ============================================================
    # STRATEGY SELECTION
    # ============================================================

    def _choose_strategy(
        self,
        state,
    ):
        """
        Reserve is always the fallback/default.

        Noble is selected only if _is_noble_exception() finds
        concrete board-state evidence for the Kangaroo route.
        """

        noble_info = (
            self._score_noble_strategy(
                state
            )
        )

        # Reserve score is retained for diagnostics only.
        # It does NOT compete numerically with Noble anymore.
        reserve_info = (
            self._score_reserve_strategy(
                state,
                noble_info=noble_info,
            )
        )

        exception = (
            self._is_noble_exception(
                state=state,
                noble_info=noble_info,
            )
        )

        if exception[
            "is_exception"
        ]:
            strategy = self.NOBLE
            reason = exception[
                "reason"
            ]
        else:
            strategy = self.RESERVE
            reason = "reserve_default"

        return {
            "strategy":
                strategy,

            "reason":
                reason,

            # Keep these so existing debug tooling remains useful.
            "noble_score":
                noble_info.get(
                    "score",
                    0.0,
                ),

            "reserve_score":
                reserve_info.get(
                    "score",
                    0.0,
                ),

            "noble":
                noble_info,

            "reserve":
                reserve_info,

            # H13-specific fields.
            "noble_exception":
                exception,

            "default_strategy":
                self.RESERVE,
        }

    # ============================================================
    # NOBLE EXCEPTION
    # ============================================================

    def _is_noble_exception(
        self,
        state,
        noble_info,
    ):
        player = state.players[
            state.current_player
        ]

        target_nobles = noble_info.get(
            "target_nobles",
            [],
        )

        overlap_count = noble_info.get(
            "overlap_count",
            0,
        )

        shared_tier1_count = noble_info.get(
            "shared_tier1_count",
            0,
        )

        total_missing = noble_info.get(
            "total_missing",
            999,
        )

        individual_missing = noble_info.get(
            "individual_missing",
            [],
        )

        points = getattr(
            player,
            "points",
            0,
        )

        # --------------------------------------------------------
        # Basic gates
        # --------------------------------------------------------

        if len(
            target_nobles
        ) < 2:
            return self._noble_exception_result(
                False,
                "fewer_than_two_nobles",
                noble_info,
                points,
            )

        if (
            overlap_count
            < self.noble_min_overlap
        ):
            return self._noble_exception_result(
                False,
                "insufficient_noble_overlap",
                noble_info,
                points,
            )

        # Once already near the end, direct point conversion is
        # normally preferred to beginning/continuing a broad engine.
        if (
            points
            > self.noble_max_points
        ):
            return self._noble_exception_result(
                False,
                "too_late_for_noble_exception",
                noble_info,
                points,
            )

        # --------------------------------------------------------
        # Trigger A:
        # Very strong noble overlap.
        # --------------------------------------------------------

        very_strong_overlap = (
            overlap_count
            >= self.noble_very_strong_overlap
            and shared_tier1_count
            >= self.noble_very_strong_min_shared_tier1
        )

        if very_strong_overlap:
            return self._noble_exception_result(
                True,
                "very_strong_overlap",
                noble_info,
                points,
            )

        # --------------------------------------------------------
        # Trigger B:
        # 2+ overlapping colors with strong visible Tier-1 support.
        # --------------------------------------------------------

        board_supported_overlap = (
            overlap_count
            >= self.noble_min_overlap
            and shared_tier1_count
            >= self.noble_board_min_shared_tier1
        )

        if board_supported_overlap:
            return self._noble_exception_result(
                True,
                "shared_colors_with_tier1_support",
                noble_info,
                points,
            )

        # --------------------------------------------------------
        # Trigger C:
        # We are already sufficiently invested in the two-noble
        # route that abandoning it would be questionable.
        # --------------------------------------------------------

        worst_individual_missing = (
            max(
                individual_missing
            )
            if individual_missing
            else 999
        )

        established_noble_route = (
            overlap_count
            >= self.noble_min_overlap
            and total_missing
            <= self.noble_progress_max_total_missing
            and worst_individual_missing
            <= self.noble_progress_max_individual_missing
        )

        if established_noble_route:
            return self._noble_exception_result(
                True,
                "established_noble_progress",
                noble_info,
                points,
            )

        return self._noble_exception_result(
            False,
            "noble_conditions_not_specific_enough",
            noble_info,
            points,
        )

    def _noble_exception_result(
        self,
        is_exception,
        reason,
        noble_info,
        points,
    ):
        individual_missing = noble_info.get(
            "individual_missing",
            [],
        )

        return {
            "is_exception":
                bool(
                    is_exception
                ),

            "reason":
                reason,

            "points":
                points,

            "overlap_count":
                noble_info.get(
                    "overlap_count",
                    0,
                ),

            "overlap_colors":
                noble_info.get(
                    "overlap_colors",
                    set(),
                ),

            "shared_tier1_count":
                noble_info.get(
                    "shared_tier1_count",
                    0,
                ),

            "aligned_tier1_count":
                noble_info.get(
                    "aligned_tier1_count",
                    0,
                ),

            "total_missing":
                noble_info.get(
                    "total_missing",
                    None,
                ),

            "individual_missing":
                individual_missing,

            "worst_individual_missing":
                (
                    max(
                        individual_missing
                    )
                    if individual_missing
                    else None
                ),
        }

    # ============================================================
    # H13 DEBUG
    # ============================================================

    def get_h13_strategy_debug(
        self,
    ):
        """
        Return the most recent strategy-selection information.

        Example:
            {
                "strategy": "reserve",
                "reason": "reserve_default",
                "noble_exception": {...},
                ...
            }
        """

        return self.last_strategy_debug


class HeuristicAgent13Diagnostics(HeuristicAgent13):
    """
    Diagnostic version of H13.

    Gameplay logic is identical to HeuristicAgent13.
    It records how often:

        - Reserve/default strategy is selected
        - Noble exception activates
        - H4 base move wins the rollout
        - strategic extra wins the rollout
        - Noble/Reserve extras are actually executed
        - each Noble exception reason occurs

    Use:
        agent = HeuristicAgent13Diagnostics(...)
        ...
        print(agent.format_diagnostic_summary())
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(
            *args,
            **kwargs,
        )

        self.reset_diagnostic_stats()

    # ============================================================
    # DECISION WRAPPER
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

        action, rollout_value, cheap_score = max(
            evaluated,
            key=lambda item: (
                item[1],
                item[2],
            ),
        )

        self._record_diagnostic_decision(
            state=state,
            selected_action=action,
            rollout_value=rollout_value,
        )

        return action

    # ============================================================
    # RESET
    # ============================================================

    def reset_diagnostic_stats(
        self,
    ):
        self._diag13 = {
            "strategic_decisions":
                0,

            "forced_node_calls":
                0,

            "strategy_counts":
                Counter(),

            "source_counts":
                Counter(),

            "strategy_source_counts":
                Counter(),

            "noble_exception_reason_counts":
                Counter(),

            "extra_action_type_counts":
                Counter(),

            "extras_available":
                0,

            "extras_proposed":
                0,
        }

    # ============================================================
    # RECORD
    # ============================================================

    def _record_diagnostic_decision(
        self,
        state,
        selected_action,
        rollout_value,
    ):
        if (
            state.node_type
            != NodeType.MAIN_DECISION
        ):
            self._diag13[
                "forced_node_calls"
            ] += 1
            return

        self._diag13[
            "strategic_decisions"
        ] += 1

        strategy_info = (
            self.last_strategy_debug
            or {}
        )

        candidate_info = (
            self.last_candidate_debug
            or {}
        )

        strategy = strategy_info.get(
            "strategy",
            self.RESERVE,
        )

        self._diag13[
            "strategy_counts"
        ][strategy] += 1

        exception = strategy_info.get(
            "noble_exception",
            {},
        )

        if strategy == self.NOBLE:
            reason = exception.get(
                "reason",
                "unknown",
            )

            self._diag13[
                "noble_exception_reason_counts"
            ][reason] += 1

        base_candidates = candidate_info.get(
            "base_candidates",
            [],
        )

        extras = candidate_info.get(
            "strategy_extras",
            [],
        )

        self._diag13[
            "extras_proposed"
        ] += len(
            extras
        )

        if extras:
            self._diag13[
                "extras_available"
            ] += 1

        selected_id = (
            self._action_identity(
                selected_action
            )
        )

        base_ids = {
            self._action_identity(
                action
            )
            for action, _
            in base_candidates
        }

        extra_ids = {
            self._action_identity(
                action
            )
            for action, _
            in extras
        }

        if selected_id in base_ids:
            source = "h4_base"

        elif selected_id in extra_ids:
            source = "strategic_extra"

        else:
            source = "unknown"

        self._diag13[
            "source_counts"
        ][source] += 1

        self._diag13[
            "strategy_source_counts"
        ][
            (
                strategy,
                source,
            )
        ] += 1

        if source == "strategic_extra":
            action_type = getattr(
                selected_action,
                "action_type",
                None,
            )

            action_type_name = getattr(
                action_type,
                "name",
                str(
                    action_type
                ),
            )

            self._diag13[
                "extra_action_type_counts"
            ][action_type_name] += 1

    # ============================================================
    # SUMMARY
    # ============================================================

    def get_diagnostic_stats(
        self,
    ):
        decisions = self._diag13[
            "strategic_decisions"
        ]

        def pct(
            count,
        ):
            return (
                100.0
                * count
                / decisions
                if decisions
                else 0.0
            )

        strategy_counts = dict(
            self._diag13[
                "strategy_counts"
            ]
        )

        source_counts = dict(
            self._diag13[
                "source_counts"
            ]
        )

        return {
            "strategic_decisions":
                decisions,

            "forced_node_calls":
                self._diag13[
                    "forced_node_calls"
                ],

            "strategy_counts":
                strategy_counts,

            "strategy_percentages": {
                key:
                    pct(
                        value
                    )
                for key, value
                in strategy_counts.items()
            },

            "source_counts":
                source_counts,

            "source_percentages": {
                key:
                    pct(
                        value
                    )
                for key, value
                in source_counts.items()
            },

            "noble_exception_reason_counts":
                dict(
                    self._diag13[
                        "noble_exception_reason_counts"
                    ]
                ),

            "extra_action_type_counts":
                dict(
                    self._diag13[
                        "extra_action_type_counts"
                    ]
                ),

            "extras_available":
                self._diag13[
                    "extras_available"
                ],

            "extras_available_pct":
                pct(
                    self._diag13[
                        "extras_available"
                    ]
                ),

            "average_extras_proposed":
                (
                    self._diag13[
                        "extras_proposed"
                    ]
                    / decisions
                    if decisions
                    else 0.0
                ),

            "strategy_source_counts": {
                f"{strategy}|{source}":
                    count

                for (
                    strategy,
                    source,
                ), count

                in self._diag13[
                    "strategy_source_counts"
                ].items()
            },
        }

    def format_diagnostic_summary(
        self,
    ):
        stats = (
            self.get_diagnostic_stats()
        )

        decisions = stats[
            "strategic_decisions"
        ]

        lines = [
            "=== HeuristicAgent13 Diagnostics ===",
            f"Strategic decisions: {decisions}",
            f"Forced-node calls: {stats['forced_node_calls']}",
            "",
            "Strategy selected:",
        ]

        for strategy in (
            self.RESERVE,
            self.NOBLE,
        ):
            count = stats[
                "strategy_counts"
            ].get(
                strategy,
                0,
            )

            percentage = stats[
                "strategy_percentages"
            ].get(
                strategy,
                0.0,
            )

            lines.append(
                f"  {strategy}: "
                f"{count} "
                f"({percentage:.2f}%)"
            )

        lines.extend(
            [
                "",
                "Final selected action source:",
            ]
        )

        for source in (
            "h4_base",
            "strategic_extra",
            "unknown",
        ):
            count = stats[
                "source_counts"
            ].get(
                source,
                0,
            )

            if (
                source == "unknown"
                and count == 0
            ):
                continue

            percentage = stats[
                "source_percentages"
            ].get(
                source,
                0.0,
            )

            lines.append(
                f"  {source}: "
                f"{count} "
                f"({percentage:.2f}%)"
            )

        lines.extend(
            [
                "",
                "Strategy x selected source:",
            ]
        )

        for strategy in (
            self.RESERVE,
            self.NOBLE,
        ):
            base_count = stats[
                "strategy_source_counts"
            ].get(
                f"{strategy}|h4_base",
                0,
            )

            extra_count = stats[
                "strategy_source_counts"
            ].get(
                f"{strategy}|strategic_extra",
                0,
            )

            lines.append(
                f"  {strategy}: "
                f"H4={base_count}, "
                f"extra={extra_count}"
            )

        lines.extend(
            [
                "",
                "Noble exception triggers:",
            ]
        )

        reasons = stats[
            "noble_exception_reason_counts"
        ]

        if reasons:
            for reason, count in sorted(
                reasons.items()
            ):
                lines.append(
                    f"  {reason}: {count}"
                )
        else:
            lines.append(
                "  none"
            )

        lines.extend(
            [
                "",
                "Selected strategic extras by action type:",
            ]
        )

        action_counts = stats[
            "extra_action_type_counts"
        ]

        if action_counts:
            for action_type, count in sorted(
                action_counts.items()
            ):
                lines.append(
                    f"  {action_type}: {count}"
                )
        else:
            lines.append(
                "  none"
            )

        lines.extend(
            [
                "",
                "Strategic-extra availability:",
                (
                    "  Decisions with >=1 extra: "
                    f"{stats['extras_available']} "
                    f"({stats['extras_available_pct']:.2f}%)"
                ),
                (
                    "  Average extras proposed / decision: "
                    f"{stats['average_extras_proposed']:.3f}"
                ),
            ]
        )

        return "\n".join(
            lines
        )
