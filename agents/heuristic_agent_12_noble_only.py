from collections import Counter

from splendor_v1.agents.heuristic_agent_12 import HeuristicAgent12
from splendor_v1.env.core.enums import NodeType


class HeuristicAgent12NobleOnly(HeuristicAgent12):
    """
    HeuristicAgent12NobleOnly
    =========================

    Clean ablation of HeuristicAgent12.

    Keeps:
        - H4/H3 base candidate set
        - full H3-vs-H3 terminal rollouts
        - Noble / Kangaroo strategic extras
        - num_strategy_moves
        - min_strategy_score

    Removes:
        - Reserve strategy branch
        - Noble-vs-Reserve score competition
        - strategy_margin

    Decision architecture
    ---------------------

        H4 candidates
            ↓
        calculate Noble strategy score
            ↓
        if noble_score >= min_strategy_score:
            add Noble strategic extras
        else:
            stay Neutral / pure H4
            ↓
        full terminal rollout over all candidates
            ↓
        choose best rollout result

    This is intended to answer:

        "Was the Reserve branch actually helping H12,
         or was Noble + H4 enough?"

    Recommended comparison:
        H12NobleOnly_16 vs H12_16
        H12NobleOnly_16 vs H4_16
    """

    def __init__(
        self,
        num_rollouts=16,
        num_strategy_moves=2,
        min_strategy_score=8.0,
        max_rollout_steps=200,
        random_seed=None,
    ):
        # strategy_margin is unused because there is only one
        # strategic branch.
        super().__init__(
            num_rollouts=num_rollouts,
            num_strategy_moves=num_strategy_moves,
            strategy_margin=0.0,
            min_strategy_score=min_strategy_score,
            max_rollout_steps=max_rollout_steps,
            random_seed=random_seed,
        )

    # ============================================================
    # STRATEGY SELECTION
    # ============================================================

    def _choose_strategy(
        self,
        state,
    ):
        """
        Noble is the only strategic mode.

        If the Noble score does not clear min_strategy_score,
        fall back to Neutral, which means H4 only.
        """

        noble_info = (
            self._score_noble_strategy(
                state
            )
        )

        noble_score = noble_info[
            "score"
        ]

        if (
            noble_score
            >= self.min_strategy_score
        ):
            strategy = self.NOBLE
            reason = (
                "noble_score_above_threshold"
            )

        else:
            strategy = self.NEUTRAL
            reason = (
                "noble_score_below_threshold"
            )

        return {
            "strategy":
                strategy,

            "reason":
                reason,

            "noble_score":
                noble_score,

            # Kept for compatibility with existing debug code.
            "reserve_score":
                None,

            "noble":
                noble_info,

            "reserve":
                None,
        }

    # ============================================================
    # STRATEGY EXTRAS
    # ============================================================

    def _get_strategy_extras(
        self,
        env,
        state,
        strategy,
        excluded_actions,
    ):
        """
        Only Noble extras are ever allowed.

        Neutral returns no extras.
        """

        if strategy != self.NOBLE:
            return []

        legal_actions = env._legal_actions(
            state
        )

        excluded = {
            self._action_identity(
                action
            )
            for action in excluded_actions
        }

        scored = []

        for action in legal_actions:

            if (
                self._action_identity(
                    action
                )
                in excluded
            ):
                continue

            score = (
                self._score_noble_extra_action(
                    state,
                    action,
                )
            )

            if score is None:
                continue

            scored.append(
                (
                    action,
                    float(
                        score
                    ),
                )
            )

        scored.sort(
            key=lambda item:
                item[1],
            reverse=True,
        )

        return scored[
            :self.num_strategy_moves
        ]


class HeuristicAgent12NobleOnlyDiagnostics(
    HeuristicAgent12NobleOnly
):
    """
    Diagnostic version of HeuristicAgent12NobleOnly.

    Gameplay behavior is identical to the noble-only agent.
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
    # ACTION SELECTION
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

        self._record_decision(
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
        self._diag = {
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

            "extra_action_type_counts":
                Counter(),

            "extras_available":
                0,

            "extras_proposed":
                0,

            "noble_score_sum":
                0.0,

            "noble_score_observations":
                0,
        }

    # ============================================================
    # RECORD
    # ============================================================

    def _record_decision(
        self,
        state,
        selected_action,
        rollout_value,
    ):
        if (
            state.node_type
            != NodeType.MAIN_DECISION
        ):
            self._diag[
                "forced_node_calls"
            ] += 1
            return

        self._diag[
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
            self.NEUTRAL,
        )

        self._diag[
            "strategy_counts"
        ][strategy] += 1

        noble_score = strategy_info.get(
            "noble_score",
            None,
        )

        if noble_score is not None:
            self._diag[
                "noble_score_sum"
            ] += float(
                noble_score
            )

            self._diag[
                "noble_score_observations"
            ] += 1

        base_candidates = candidate_info.get(
            "base_candidates",
            [],
        )

        extras = candidate_info.get(
            "strategy_extras",
            [],
        )

        self._diag[
            "extras_proposed"
        ] += len(
            extras
        )

        if extras:
            self._diag[
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
            source = "noble_extra"

        else:
            source = "unknown"

        self._diag[
            "source_counts"
        ][source] += 1

        self._diag[
            "strategy_source_counts"
        ][
            (
                strategy,
                source,
            )
        ] += 1

        if source == "noble_extra":
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

            self._diag[
                "extra_action_type_counts"
            ][action_type_name] += 1

    # ============================================================
    # STRUCTURED STATS
    # ============================================================

    def get_diagnostic_stats(
        self,
    ):
        decisions = self._diag[
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

        score_n = self._diag[
            "noble_score_observations"
        ]

        strategy_counts = dict(
            self._diag[
                "strategy_counts"
            ]
        )

        source_counts = dict(
            self._diag[
                "source_counts"
            ]
        )

        return {
            "strategic_decisions":
                decisions,

            "forced_node_calls":
                self._diag[
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

            "strategy_source_counts": {
                f"{strategy}|{source}":
                    count

                for (
                    strategy,
                    source,
                ), count

                in self._diag[
                    "strategy_source_counts"
                ].items()
            },

            "extra_action_type_counts":
                dict(
                    self._diag[
                        "extra_action_type_counts"
                    ]
                ),

            "extras_available":
                self._diag[
                    "extras_available"
                ],

            "extras_available_pct":
                pct(
                    self._diag[
                        "extras_available"
                    ]
                ),

            "average_extras_proposed":
                (
                    self._diag[
                        "extras_proposed"
                    ]
                    / decisions
                    if decisions
                    else 0.0
                ),

            "average_noble_score":
                (
                    self._diag[
                        "noble_score_sum"
                    ]
                    / score_n
                    if score_n
                    else 0.0
                ),
        }

    # ============================================================
    # HUMAN-READABLE SUMMARY
    # ============================================================

    def format_diagnostic_summary(
        self,
    ):
        stats = (
            self.get_diagnostic_stats()
        )

        lines = [
            "=== HeuristicAgent12 Noble-Only Diagnostics ===",
            f"Strategic decisions: {stats['strategic_decisions']}",
            f"Forced-node calls: {stats['forced_node_calls']}",
            "",
            "Strategy selected:",
        ]

        for strategy in (
            self.NOBLE,
            self.NEUTRAL,
        ):
            count = stats[
                "strategy_counts"
            ].get(
                strategy,
                0,
            )

            pct = stats[
                "strategy_percentages"
            ].get(
                strategy,
                0.0,
            )

            lines.append(
                f"  {strategy}: "
                f"{count} "
                f"({pct:.2f}%)"
            )

        lines.extend(
            [
                "",
                "Final selected action source:",
            ]
        )

        for source in (
            "h4_base",
            "noble_extra",
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

            pct = stats[
                "source_percentages"
            ].get(
                source,
                0.0,
            )

            lines.append(
                f"  {source}: "
                f"{count} "
                f"({pct:.2f}%)"
            )

        lines.extend(
            [
                "",
                "Selected Noble extras by action type:",
            ]
        )

        extra_counts = stats[
            "extra_action_type_counts"
        ]

        if extra_counts:
            for action_type, count in sorted(
                extra_counts.items()
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
                "Noble-extra availability:",
                (
                    "  Decisions with >=1 Noble extra: "
                    f"{stats['extras_available']} "
                    f"({stats['extras_available_pct']:.2f}%)"
                ),
                (
                    "  Average Noble extras proposed / decision: "
                    f"{stats['average_extras_proposed']:.3f}"
                ),
                (
                    "  Average Noble classifier score: "
                    f"{stats['average_noble_score']:.3f}"
                ),
            ]
        )

        return "\n".join(
            lines
        )
