from collections import Counter

from splendor_v1.agents.heuristic_agent_15 import (
    HeuristicAgent15,
    HeuristicAgent15Diagnostics,
)
from splendor_v1.env.core.constants import COLOR_ORDER
from splendor_v1.env.core.enums import NodeType
from splendor_v1.env.core.actions import ActionType


class _PostDiscardTakeMixin:
    """
    H16 change relative to H15
    ==========================

    H15 evaluates TAKE_GEMS using the temporary hand immediately after
    taking gems, even if that hand exceeds the 10-token limit and will
    be forced to discard.

    H16 changes ONLY overflow-producing TAKE evaluation.

    If a TAKE does not overflow:
        use H15's exact TAKE scorer unchanged.

    If a TAKE does overflow:
        1. add the taken gems
        2. enumerate every legal way to discard back to max_gems
        3. score each resulting legal hand
        4. use the best post-discard result as the TAKE's cheap score

    This makes H16 a clean ablation of H15:

        H15
          +
        post-discard-aware TAKE evaluation

    No Noble, Tier-2, Tier-3, candidate-budget, reservation, or rollout
    logic is changed.
    """

    # ============================================================
    # TAKE SCORING OVERRIDE
    # ============================================================

    def _score_take_action(
        self,
        state,
        action,
        board_model,
    ):
        player = board_model[
            "player"
        ]

        gems_after_take = dict(
            player.gems
        )

        for color in action.gem_colors:
            gems_after_take[
                color
            ] += 1

        max_gems = int(
            getattr(
                state,
                "max_gems",
                10,
            )
        )

        total_after_take = sum(
            gems_after_take.values()
        )

        overflow = max(
            0,
            total_after_take
            - max_gems,
        )

        # --------------------------------------------------------
        # CLEAN ABLATION:
        #
        # If this TAKE does not overflow, H16 is EXACTLY H15.
        # --------------------------------------------------------

        if overflow <= 0:
            score = super()._score_take_action(
                state,
                action,
                board_model,
            )

            self._cache_h16_take_detail(
                action=action,
                detail={
                    "causes_overflow":
                        False,

                    "overflow_size":
                        0,

                    "score":
                        float(
                            score
                        ),

                    "discarded_counts":
                        {},

                    "rediscarded_gained_tokens":
                        0,

                    "final_gems":
                        gems_after_take,
                },
            )

            return score

        # --------------------------------------------------------
        # Overflow:
        #
        # Enumerate ALL legal post-discard hands and let the unified
        # route scorer decide which legal 10-token hand is best.
        # --------------------------------------------------------

        best = None

        for final_gems in (
            self._enumerate_post_discard_hands(
                gems_after_take,
                overflow,
            )
        ):
            score = (
                self._score_post_discard_take_hand(
                    state=state,
                    action=action,
                    board_model=board_model,
                    final_gems=final_gems,
                )
            )

            discarded_counts = (
                self._discard_counts_between(
                    gems_after_take,
                    final_gems,
                )
            )

            rediscarded = (
                self._count_rediscarded_gained_tokens(
                    action=action,
                    discarded_counts=discarded_counts,
                )
            )

            retained_route_delta = (
                self._route_weighted_gem_delta(
                    before_gems=player.gems,
                    after_gems=final_gems,
                    color_demand=board_model[
                        "color_demand"
                    ],
                )
            )

            # Score is primary.
            #
            # Tie-breaks:
            #   1. retain more route-weighted gem value
            #   2. avoid immediately throwing away newly taken gems
            rank = (
                float(
                    score
                ),
                float(
                    retained_route_delta
                ),
                -int(
                    rediscarded
                ),
            )

            if (
                best is None
                or rank
                > best[
                    "rank"
                ]
            ):
                best = {
                    "rank":
                        rank,

                    "score":
                        float(
                            score
                        ),

                    "final_gems":
                        final_gems,

                    "discarded_counts":
                        discarded_counts,

                    "rediscarded_gained_tokens":
                        int(
                            rediscarded
                        ),

                    "retained_route_delta":
                        float(
                            retained_route_delta
                        ),
                }

        # There should always be at least one legal discard plan.
        # Defensive fallback preserves H15 behavior if state encoding
        # is ever malformed.
        if best is None:
            score = super()._score_take_action(
                state,
                action,
                board_model,
            )

            self._cache_h16_take_detail(
                action=action,
                detail={
                    "causes_overflow":
                        True,

                    "overflow_size":
                        overflow,

                    "score":
                        float(
                            score
                        ),

                    "discarded_counts":
                        {},

                    "rediscarded_gained_tokens":
                        0,

                    "final_gems":
                        gems_after_take,

                    "fallback_to_h15":
                        True,
                },
            )

            return score

        self._cache_h16_take_detail(
            action=action,
            detail={
                "causes_overflow":
                    True,

                "overflow_size":
                    overflow,

                "score":
                    best[
                        "score"
                    ],

                "discarded_counts":
                    best[
                        "discarded_counts"
                    ],

                "rediscarded_gained_tokens":
                    best[
                        "rediscarded_gained_tokens"
                    ],

                "retained_route_delta":
                    best[
                        "retained_route_delta"
                    ],

                "final_gems":
                    best[
                        "final_gems"
                    ],
            },
        )

        return best[
            "score"
        ]

    # ============================================================
    # SCORE ONE LEGAL POST-DISCARD HAND
    # ============================================================

    def _score_post_discard_take_hand(
        self,
        state,
        action,
        board_model,
        final_gems,
    ):
        """
        Recompute H15's TAKE value from the legal hand that will exist
        AFTER all required discards.

        The structure deliberately mirrors H15._score_take_action().
        """

        player = board_model[
            "player"
        ]

        color_demand = board_model[
            "color_demand"
        ]

        # --------------------------------------------------------
        # 1. Direct strategic value of the NET hand change.
        #
        # H15 gives a small bonus for each color taken.
        #
        # H16 instead credits only what survives the forced discard
        # and penalizes route-relevant gems that had to be thrown
        # away.
        # --------------------------------------------------------

        score = (
            self._route_weighted_gem_delta(
                before_gems=player.gems,
                after_gems=final_gems,
                color_demand=color_demand,
            )
            * 0.35
        )

        # --------------------------------------------------------
        # 2. T1 / T2 bridge progress from the FINAL legal hand.
        # --------------------------------------------------------

        best_bridge = 0.0

        for tier in (
            1,
            2,
        ):
            for card in state.visible_cards[
                tier
            ]:
                if card is None:
                    continue

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
                        final_gems,
                    )
                )

                progress = (
                    before
                    - after
                )

                if progress <= 0:
                    continue

                demand = (
                    color_demand.get(
                        card.bonus_color,
                        0.0,
                    )
                )

                if tier == 1:
                    target_value = (
                        3.0
                        + demand
                        * 2.0
                    )

                else:
                    target_value = (
                        card.points
                        * self.tier2_point_weight
                        + demand
                        * 1.5
                    )

                best_bridge = max(
                    best_bridge,
                    target_value
                    + progress
                    * 4.0,
                )

        # --------------------------------------------------------
        # 3. Direct T3 progress from the FINAL legal hand.
        # --------------------------------------------------------

        best_t3 = 0.0

        for role in (
            "primary",
            "secondary",
        ):
            target = board_model[
                "tier3"
            ].get(
                role,
                None,
            )

            if target is None:
                continue

            card = target[
                "card"
            ]

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
                    final_gems,
                )
            )

            progress = (
                before
                - after
            )

            if progress <= 0:
                continue

            role_multiplier = (
                1.0
                if role
                == "primary"
                else 0.8
            )

            best_t3 = max(
                best_t3,
                target[
                    "route_score"
                ]
                * 0.30
                * role_multiplier
                + progress
                * 4.0,
            )

        score += max(
            best_bridge,
            best_t3,
        )

        # --------------------------------------------------------
        # 4. Two-player bank denial.
        #
        # Only credit denial if at least one gem of that color is
        # actually retained after discard.
        # --------------------------------------------------------

        score += (
            self._take_bank_denial_bonus_post_discard(
                state=state,
                action=action,
                before_gems=player.gems,
                final_gems=final_gems,
            )
        )

        return score

    # ============================================================
    # ENUMERATE LEGAL DISCARD OUTCOMES
    # ============================================================

    def _enumerate_post_discard_hands(
        self,
        gems_after_take,
        discard_count,
    ):
        """
        Yield every unique gem dictionary obtainable by discarding
        exactly discard_count tokens.

        Splendor has only six token colors and overflow after a TAKE is
        small, so exhaustive enumeration is cheap.
        """

        colors = [
            color
            for color, count
            in gems_after_take.items()
            if count > 0
        ]

        discard_plan = {}

        yield from (
            self._enumerate_discard_recursive(
                gems_after_take=gems_after_take,
                colors=colors,
                color_index=0,
                remaining=discard_count,
                discard_plan=discard_plan,
            )
        )

    def _enumerate_discard_recursive(
        self,
        gems_after_take,
        colors,
        color_index,
        remaining,
        discard_plan,
    ):
        if remaining == 0:
            final_gems = dict(
                gems_after_take
            )

            for color, count in (
                discard_plan.items()
            ):
                final_gems[
                    color
                ] -= count

            yield final_gems
            return

        if color_index >= len(
            colors
        ):
            return

        color = colors[
            color_index
        ]

        available = min(
            gems_after_take[
                color
            ],
            remaining,
        )

        # Try every possible quantity of this color.
        for count in range(
            available + 1
        ):
            if count > 0:
                discard_plan[
                    color
                ] = count

            elif color in discard_plan:
                del discard_plan[
                    color
                ]

            yield from (
                self._enumerate_discard_recursive(
                    gems_after_take=gems_after_take,
                    colors=colors,
                    color_index=color_index + 1,
                    remaining=remaining - count,
                    discard_plan=discard_plan,
                )
            )

        if color in discard_plan:
            del discard_plan[
                color
            ]

    # ============================================================
    # POST-DISCARD HELPERS
    # ============================================================

    def _discard_counts_between(
        self,
        before_discard,
        after_discard,
    ):
        result = {}

        for color, before_count in (
            before_discard.items()
        ):
            count = (
                before_count
                - after_discard.get(
                    color,
                    0,
                )
            )

            if count > 0:
                result[
                    color
                ] = count

        return result

    def _count_rediscarded_gained_tokens(
        self,
        action,
        discarded_counts,
    ):
        gained = Counter(
            action.gem_colors
        )

        rediscarded = 0

        for color, gained_count in (
            gained.items()
        ):
            rediscarded += min(
                gained_count,
                discarded_counts.get(
                    color,
                    0,
                ),
            )

        return rediscarded

    def _route_weighted_gem_delta(
        self,
        before_gems,
        after_gems,
        color_demand,
    ):
        value = 0.0

        for color in COLOR_ORDER:
            delta = (
                after_gems.get(
                    color,
                    0,
                )
                - before_gems.get(
                    color,
                    0,
                )
            )

            value += (
                delta
                * color_demand.get(
                    color,
                    0.0,
                )
            )

        return value

    def _take_bank_denial_bonus_post_discard(
        self,
        state,
        action,
        before_gems,
        final_gems,
    ):
        bonus = 0.0

        for color in set(
            action.gem_colors
        ):
            if color not in COLOR_ORDER:
                continue

            try:
                bank_count = (
                    state.bank[
                        color
                    ]
                )
            except Exception:
                continue

            # In 2p, take-two is legal only when all four are in the
            # bank. Retaining at least one token keeps that option
            # disabled for the opponent.
            if (
                bank_count == 4
                and final_gems.get(
                    color,
                    0,
                )
                > before_gems.get(
                    color,
                    0,
                )
            ):
                bonus += (
                    self.bank_denial_weight
                )

        return bonus

    # ============================================================
    # TAKE-DETAIL CACHE FOR DIAGNOSTICS
    # ============================================================

    def _cache_h16_take_detail(
        self,
        action,
        detail,
    ):
        if not hasattr(
            self,
            "_h16_take_detail_cache",
        ):
            self._h16_take_detail_cache = {}

        self._h16_take_detail_cache[
            self._action_identity(
                action
            )
        ] = detail

    def _get_h16_take_detail(
        self,
        action,
    ):
        return getattr(
            self,
            "_h16_take_detail_cache",
            {},
        ).get(
            self._action_identity(
                action
            )
        )


class HeuristicAgent16(
    _PostDiscardTakeMixin,
    HeuristicAgent15,
):
    """
    Heuristic 16
    ============

    H15 Unified Route Planner
        +
    post-discard-aware TAKE evaluation.

    Everything else is inherited from H15.

    This is intentionally a narrow experiment so H16 vs H15 measures
    whether respecting the 10-token cap during candidate scoring
    improves strength.
    """

    pass


class HeuristicAgent16Diagnostics(
    _PostDiscardTakeMixin,
    HeuristicAgent15Diagnostics,
):
    """
    H15 diagnostics plus H16-specific information about selected TAKEs.

    Existing H15 diagnostics remain available through inheritance.
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

        self._reset_h16_take_diagnostics()

    def _reset_h16_take_diagnostics(
        self,
    ):
        self._diag16_take = {
            "selected_take_count":
                0,

            "selected_take_overflow_count":
                0,

            "predicted_discard_tokens":
                0,

            "predicted_rediscarded_gained_tokens":
                0,

            "predicted_discard_color_counts":
                Counter(),

            "overflow_size_counts":
                Counter(),
        }

    def select_action(
        self,
        env,
        state,
    ):
        action = super().select_action(
            env,
            state,
        )

        if (
            action is None
            or state.node_type
            != NodeType.MAIN_DECISION
            or action.action_type
            != ActionType.TAKE_GEMS
        ):
            return action

        self._diag16_take[
            "selected_take_count"
        ] += 1

        detail = (
            self._get_h16_take_detail(
                action
            )
        )

        if not detail:
            return action

        if detail.get(
            "causes_overflow",
            False,
        ):
            self._diag16_take[
                "selected_take_overflow_count"
            ] += 1

            overflow_size = int(
                detail.get(
                    "overflow_size",
                    0,
                )
            )

            self._diag16_take[
                "overflow_size_counts"
            ][overflow_size] += 1

            discarded_counts = detail.get(
                "discarded_counts",
                {},
            )

            discarded_total = sum(
                discarded_counts.values()
            )

            self._diag16_take[
                "predicted_discard_tokens"
            ] += discarded_total

            self._diag16_take[
                "predicted_rediscarded_gained_tokens"
            ] += int(
                detail.get(
                    "rediscarded_gained_tokens",
                    0,
                )
            )

            for color, count in (
                discarded_counts.items()
            ):
                color_name = getattr(
                    color,
                    "name",
                    str(
                        color
                    ),
                )

                self._diag16_take[
                    "predicted_discard_color_counts"
                ][color_name] += count

        return action

    def get_diagnostic_stats(
        self,
    ):
        stats = super().get_diagnostic_stats()

        take_count = self._diag16_take[
            "selected_take_count"
        ]

        overflow_count = self._diag16_take[
            "selected_take_overflow_count"
        ]

        stats[
            "h16_post_discard_take"
        ] = {
            "selected_take_count":
                take_count,

            "selected_take_overflow_count":
                overflow_count,

            "selected_take_overflow_pct":
                (
                    100.0
                    * overflow_count
                    / take_count
                    if take_count
                    else 0.0
                ),

            "predicted_discard_tokens":
                self._diag16_take[
                    "predicted_discard_tokens"
                ],

            "predicted_rediscarded_gained_tokens":
                self._diag16_take[
                    "predicted_rediscarded_gained_tokens"
                ],

            "predicted_discard_color_counts":
                dict(
                    self._diag16_take[
                        "predicted_discard_color_counts"
                    ]
                ),

            "overflow_size_counts":
                dict(
                    self._diag16_take[
                        "overflow_size_counts"
                    ]
                ),
        }

        return stats

    def format_diagnostic_summary(
        self,
    ):
        base = super().format_diagnostic_summary()

        # Rename inherited title for readability.
        base = base.replace(
            "HeuristicAgent15 Unified Route Planner Diagnostics",
            "HeuristicAgent16 Post-Discard Route Planner Diagnostics",
            1,
        )

        stats = self.get_diagnostic_stats()[
            "h16_post_discard_take"
        ]

        lines = [
            base,
            "",
            "H16 post-discard TAKE diagnostics:",
            (
                "  Selected TAKE actions: "
                f"{stats['selected_take_count']}"
            ),
            (
                "  Selected TAKEs that still intentionally overflow: "
                f"{stats['selected_take_overflow_count']} "
                f"({stats['selected_take_overflow_pct']:.2f}%)"
            ),
            (
                "  Predicted tokens discarded after selected TAKEs: "
                f"{stats['predicted_discard_tokens']}"
            ),
            (
                "  Predicted newly-gained tokens immediately re-discarded: "
                f"{stats['predicted_rediscarded_gained_tokens']}"
            ),
            (
                "  Predicted discard colors: "
                f"{stats['predicted_discard_color_counts']}"
            ),
            (
                "  Selected overflow sizes: "
                f"{stats['overflow_size_counts']}"
            ),
        ]

        return "\n".join(
            lines
        )
