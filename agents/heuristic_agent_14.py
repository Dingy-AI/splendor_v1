from collections import Counter

from splendor_v1.agents.heuristic_agent_12 import HeuristicAgent12
from splendor_v1.env.core.constants import COLOR_ORDER
from splendor_v1.env.core.enums import GemColor, NodeType
from splendor_v1.env.core.actions import ActionType


class HeuristicAgent14(HeuristicAgent12):
    """
    HeuristicAgent14 — "Kangaroo Reservation"
    ==========================================

    Protected-H4 agent with TWO INDEPENDENT strategic route planners:

        1. Kangaroo / Noble route
        2. Tier-3 Reservation route

    Neither planner directly chooses the move.

    Every H4 candidate is protected.
    Strategic planners only ADD challengers.
    Full H3-vs-H3 terminal rollouts remain the final authority.

    ----------------------------------------------------------------
    HIGH-LEVEL ARCHITECTURE
    ----------------------------------------------------------------

        H4 / H3 base candidates
                ↓
        evaluate Noble route independently
        evaluate Tier-3 route independently
                ↓
        Noble qualifies?
            → add Noble challengers

        Tier-3 route qualifies?
            → GUARANTEE reserve rollout for primary T3
            → if compatible secondary T3 exists, also guarantee
              its reserve rollout
            → add best T1/T2 bridge challengers
                ↓
        dedupe / cap optional extras
                ↓
        full H3 terminal rollout over EVERYTHING
                ↓
        choose best terminal result

    ----------------------------------------------------------------
    TIER-3 RESERVATION ROUTE
    ----------------------------------------------------------------

    The Tier-3 planner does NOT merely ask:

        "Is this an expensive card with many points?"

    It asks:

        "Can my current engine + gems + visible Tier-1 bridge cards
         realistically grow into this Tier-3 card?"

    For each Tier-3 target and each normal color:

        uncovered[color] =
            max(
                0,
                cost[color]
                - current permanent bonuses[color]
                - current gems[color]
            )

    Gold is then allocated flexibly against the largest remaining
    deficits.

    Visible/owned Tier-1 cards are used to estimate the permanent
    bonus bridge that can still be built.

    Tier-2 cards are ALSO evaluated, but as:

        bridge + points

    rather than as the thing that determines Tier-3 feasibility.

    This gives the intended hierarchy:

        Tier 1 = engine
        Tier 2 = bridge + points
        Tier 3 = main scoring destination

    ----------------------------------------------------------------
    PRIMARY + SECONDARY TIER-3 TARGETS
    ----------------------------------------------------------------

    H14 first finds ONE best viable Tier-3 target.

    That target creates the route.

    Then H14 searches for a secondary Tier-3 card that:

        - is itself viable
        - is no more than secondary_score_tolerance worse
          than the primary
        - shares enough remaining engine demand with the primary

    A secondary target is therefore a route-strengthening signal,
    NOT a hard prerequisite.

    One excellent Tier-3 card can create a valid route.
    A second compatible Tier-3 card makes the engine investment
    much more reusable.

    ----------------------------------------------------------------
    MANDATORY RESERVE INVESTIGATION
    ----------------------------------------------------------------

    If the Tier-3 route qualifies:

        the legal RESERVE_VISIBLE action for the primary Tier-3
        is ALWAYS inserted into the rollout candidate set.

    If a qualifying secondary exists and is visible:

        its RESERVE_VISIBLE action is also inserted.

    These mandatory reserve investigations are not pruned by the
    optional-strategy candidate budget.

    The terminal rollout can still reject them.

    Route qualification decides:
        "This reserve deserves investigation."

    Terminal rollout decides:
        "Should we actually reserve it now?"

    ----------------------------------------------------------------
    CROSS-ROUTE SYNERGY
    ----------------------------------------------------------------

    Noble and Tier-3 routes can both qualify.

    A Tier-1/Tier-2 card whose bonus color:

        - advances the selected Noble pair
        AND
        - bridges the Tier-3 route

    receives additional shortlist priority.

    This lets a single engine investment support both strategic
    destinations.

    ----------------------------------------------------------------
    DEFAULTS
    ----------------------------------------------------------------

    num_rollouts = 16

    This matches the strong H12 test configuration, but is fully
    configurable.
    """

    def __init__(
        self,
        num_rollouts=16,
        max_rollout_steps=200,
        random_seed=None,

        # Noble route.
        min_noble_score=8.0,
        num_noble_moves=2,

        # Tier-3 route.
        num_tier3_bridge_moves=2,
        max_strategy_extras=4,

        # Primary Tier-3 viability.
        tier3_min_coverage_ratio=0.55,
        tier3_max_residual_after_tier1=5.0,

        # Secondary Tier-3 qualification.
        secondary_score_tolerance=4.0,
        secondary_min_overlap_ratio=0.35,

        # Relative bridge weights.
        tier1_bridge_weight=2.5,
        tier2_bridge_weight=1.4,
        tier2_point_weight=1.5,
        direct_tier3_progress_weight=1.0,

        # Cross-route bonus.
        cross_route_synergy_bonus=4.0,
    ):
        # H14 owns route selection itself, so H12's single-strategy
        # classifier is not used.
        super().__init__(
            num_rollouts=num_rollouts,
            num_strategy_moves=0,
            strategy_margin=0.0,
            min_strategy_score=0.0,
            max_rollout_steps=max_rollout_steps,
            random_seed=random_seed,
        )

        self.min_noble_score = float(
            min_noble_score
        )

        self.num_noble_moves = max(
            0,
            int(num_noble_moves),
        )

        self.num_tier3_bridge_moves = max(
            0,
            int(num_tier3_bridge_moves),
        )

        self.max_strategy_extras = max(
            0,
            int(max_strategy_extras),
        )

        self.tier3_min_coverage_ratio = float(
            tier3_min_coverage_ratio
        )

        self.tier3_max_residual_after_tier1 = float(
            tier3_max_residual_after_tier1
        )

        self.secondary_score_tolerance = float(
            secondary_score_tolerance
        )

        self.secondary_min_overlap_ratio = float(
            secondary_min_overlap_ratio
        )

        self.tier1_bridge_weight = float(
            tier1_bridge_weight
        )

        self.tier2_bridge_weight = float(
            tier2_bridge_weight
        )

        self.tier2_point_weight = float(
            tier2_point_weight
        )

        self.direct_tier3_progress_weight = float(
            direct_tier3_progress_weight
        )

        self.cross_route_synergy_bonus = float(
            cross_route_synergy_bonus
        )

    # ============================================================
    # ROOT CANDIDATES
    # ============================================================

    def _get_root_candidates(
        self,
        env,
        state,
    ):
        """
        Protect H4 candidates, then independently add Noble and
        Tier-3 route challengers.
        """

        base_candidates = (
            self.h3._get_scored_candidates(
                env,
                state,
            )
        )

        if not base_candidates:
            return []

        if (
            state.node_type
            != NodeType.MAIN_DECISION
        ):
            self.last_strategy_debug = {
                "noble_qualified": False,
                "tier3_qualified": False,
                "reason": "forced_node",
            }

            self.last_candidate_debug = {
                "base_count": len(
                    base_candidates
                ),
                "extra_count": 0,
                "total_count": len(
                    base_candidates
                ),
                "base_candidates": base_candidates,
                "extra_records": [],
            }

            return base_candidates

        excluded_ids = {
            self._action_identity(
                action
            )
            for action, _
            in base_candidates
        }

        # --------------------------------------------------------
        # 1. Noble route
        # --------------------------------------------------------

        noble_info = (
            self._score_noble_strategy(
                state
            )
        )

        noble_qualified = (
            noble_info.get(
                "score",
                0.0,
            )
            >= self.min_noble_score
        )

        noble_records = []

        if (
            noble_qualified
            and self.num_noble_moves > 0
        ):
            noble_records = (
                self._collect_noble_extra_records(
                    env=env,
                    state=state,
                    excluded_ids=excluded_ids,
                    noble_info=noble_info,
                )
            )

        # --------------------------------------------------------
        # 2. Tier-3 route
        # --------------------------------------------------------

        tier3_plan = (
            self._build_tier3_route(
                state
            )
        )

        tier3_qualified = (
            tier3_plan[
                "qualified"
            ]
        )

        mandatory_t3_records = []
        bridge_records = []

        if tier3_qualified:
            mandatory_t3_records = (
                self._collect_mandatory_tier3_reserves(
                    env=env,
                    state=state,
                    tier3_plan=tier3_plan,
                    excluded_ids=excluded_ids,
                )
            )

            bridge_records = (
                self._collect_tier3_bridge_records(
                    env=env,
                    state=state,
                    tier3_plan=tier3_plan,
                    noble_info=(
                        noble_info
                        if noble_qualified
                        else None
                    ),
                    excluded_ids=excluded_ids,
                )
            )

        # --------------------------------------------------------
        # 3. Merge optional extras.
        #
        # Mandatory Tier-3 reserve investigations bypass this cap.
        # --------------------------------------------------------

        optional_records = (
            noble_records
            + bridge_records
        )

        optional_records = (
            self._merge_extra_records(
                optional_records
            )
        )

        mandatory_t3_records = (
            self._merge_extra_records(
                mandatory_t3_records
            )
        )

        mandatory_ids = {
            self._action_identity(
                record["action"]
            )
            for record in mandatory_t3_records
        }

        optional_records = [
            record
            for record in optional_records
            if self._action_identity(
                record["action"]
            )
            not in mandatory_ids
        ]

        # Both-route moves get a shortlist bonus.
        for record in optional_records:
            sources = record[
                "sources"
            ]

            if (
                "noble" in sources
                and "tier3_bridge" in sources
            ):
                record[
                    "score"
                ] += (
                    self.cross_route_synergy_bonus
                )

        optional_records.sort(
            key=lambda record:
                record[
                    "score"
                ],
            reverse=True,
        )

        optional_records = optional_records[
            :self.max_strategy_extras
        ]

        extra_records = (
            mandatory_t3_records
            + optional_records
        )

        # --------------------------------------------------------
        # 4. Protect H4 candidates and append extras.
        # --------------------------------------------------------

        combined = list(
            base_candidates
        )

        existing = set(
            excluded_ids
        )

        final_extra_records = []

        for record in extra_records:
            action = record[
                "action"
            ]

            identity = (
                self._action_identity(
                    action
                )
            )

            if identity in existing:
                continue

            combined.append(
                (
                    action,
                    float(
                        record[
                            "score"
                        ]
                    ),
                )
            )

            existing.add(
                identity
            )

            final_extra_records.append(
                record
            )

        # --------------------------------------------------------
        # 5. Debug snapshots.
        # --------------------------------------------------------

        self.last_strategy_debug = {
            "noble_qualified":
                noble_qualified,

            "noble_score":
                noble_info.get(
                    "score",
                    0.0,
                ),

            "noble":
                noble_info,

            "tier3_qualified":
                tier3_qualified,

            "tier3":
                tier3_plan,

            "both_qualified":
                (
                    noble_qualified
                    and tier3_qualified
                ),
        }

        self.last_candidate_debug = {
            "base_count":
                len(
                    base_candidates
                ),

            "extra_count":
                len(
                    final_extra_records
                ),

            "total_count":
                len(
                    combined
                ),

            "base_candidates":
                base_candidates,

            # Compatibility with older diagnostics.
            "strategy_extras": [
                (
                    record[
                        "action"
                    ],
                    record[
                        "score"
                    ],
                )
                for record
                in final_extra_records
            ],

            "extra_records":
                final_extra_records,

            "mandatory_tier3_reserve_count":
                sum(
                    1
                    for record
                    in final_extra_records
                    if record.get(
                        "mandatory",
                        False,
                    )
                ),
        }

        return combined

    # ============================================================
    # NOBLE EXTRAS
    # ============================================================

    def _collect_noble_extra_records(
        self,
        env,
        state,
        excluded_ids,
        noble_info,
    ):
        legal_actions = env._legal_actions(
            state
        )

        records = []

        for action in legal_actions:

            if (
                self._action_identity(
                    action
                )
                in excluded_ids
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

            records.append(
                {
                    "action":
                        action,

                    "score":
                        float(
                            score
                        ),

                    "sources":
                        {
                            "noble"
                        },

                    "mandatory":
                        False,
                }
            )

        records.sort(
            key=lambda record:
                record[
                    "score"
                ],
            reverse=True,
        )

        return records[
            :self.num_noble_moves
        ]

    # ============================================================
    # TIER-3 ROUTE PLANNER
    # ============================================================

    def _build_tier3_route(
        self,
        state,
    ):
        """
        Find one viable primary Tier-3 target, then look for a
        compatible secondary target that is close enough in route
        quality.
        """

        player = state.players[
            state.current_player
        ]

        tier1_supply = (
            self._tier1_bridge_supply(
                state,
                player,
            )
        )

        tier2_support = (
            self._tier2_bridge_supply(
                state,
                player,
            )
        )

        target_records = (
            self._collect_tier3_targets(
                state=state,
                player=player,
                tier1_supply=tier1_supply,
                tier2_support=tier2_support,
            )
        )

        viable_targets = [
            target
            for target in target_records
            if target[
                "viable"
            ]
        ]

        if not viable_targets:
            return {
                "qualified":
                    False,

                "primary":
                    None,

                "secondary":
                    None,

                "targets":
                    target_records,

                "tier1_supply":
                    tier1_supply,

                "tier2_support":
                    tier2_support,

                "bridge_priority":
                    {
                        color: 0.0
                        for color
                        in COLOR_ORDER
                    },
            }

        viable_targets.sort(
            key=lambda target:
                target[
                    "route_score"
                ],
            reverse=True,
        )

        primary = viable_targets[
            0
        ]

        secondary = None
        best_secondary_rank = None

        for candidate in viable_targets[
            1:
        ]:
            score_gap = (
                primary[
                    "route_score"
                ]
                - candidate[
                    "route_score"
                ]
            )

            if (
                score_gap
                > self.secondary_score_tolerance
            ):
                continue

            overlap = (
                self._tier3_target_overlap(
                    primary,
                    candidate,
                )
            )

            if (
                overlap[
                    "ratio"
                ]
                < self.secondary_min_overlap_ratio
            ):
                continue

            rank = (
                overlap[
                    "ratio"
                ],
                overlap[
                    "shared_demand"
                ],
                candidate[
                    "route_score"
                ],
            )

            if (
                best_secondary_rank is None
                or rank
                > best_secondary_rank
            ):
                best_secondary_rank = (
                    rank
                )

                secondary = dict(
                    candidate
                )

                secondary[
                    "overlap_with_primary"
                ] = overlap

        bridge_priority = (
            self._build_tier3_bridge_priority(
                primary=primary,
                secondary=secondary,
            )
        )

        return {
            "qualified":
                True,

            "primary":
                primary,

            "secondary":
                secondary,

            "targets":
                target_records,

            "tier1_supply":
                tier1_supply,

            "tier2_support":
                tier2_support,

            "bridge_priority":
                bridge_priority,
        }

    # ============================================================
    # TIER-3 TARGET COLLECTION / SCORING
    # ============================================================

    def _collect_tier3_targets(
        self,
        state,
        player,
        tier1_supply,
        tier2_support,
    ):
        targets = []

        # Visible Tier-3 cards.
        for slot, card in enumerate(
            state.visible_cards[
                3
            ]
        ):
            if card is None:
                continue

            targets.append(
                self._score_tier3_target(
                    player=player,
                    card=card,
                    tier1_supply=tier1_supply,
                    tier2_support=tier2_support,
                    source="visible",
                    slot=slot,
                )
            )

        # Already-reserved Tier-3 cards remain strategic targets.
        for reserved_index, card in enumerate(
            player.reserved_cards
        ):
            if card is None:
                continue

            tier = self._card_tier(
                card
            )

            if tier != 3:
                continue

            targets.append(
                self._score_tier3_target(
                    player=player,
                    card=card,
                    tier1_supply=tier1_supply,
                    tier2_support=tier2_support,
                    source="reserved",
                    reserved_index=reserved_index,
                )
            )

        targets.sort(
            key=lambda target:
                target[
                    "route_score"
                ],
            reverse=True,
        )

        return targets

    def _score_tier3_target(
        self,
        player,
        card,
        tier1_supply,
        tier2_support,
        source,
        slot=None,
        reserved_index=None,
    ):
        """
        Tier-3 viability is anchored on:

            current bonuses
            + current gems
            + gold
            + realistically obtainable Tier-1 bonus bridge

        Tier-2 support improves route quality but does not decide
        basic viability.
        """

        raw_profile = (
            self._uncovered_color_profile(
                player,
                card,
            )
        )

        post_gold_profile = (
            self._apply_gold_to_profile(
                raw_profile,
                self._gold_count(
                    player
                ),
            )
        )

        total_cost = sum(
            card.cost.get(
                color,
                0,
            )
            for color in COLOR_ORDER
        )

        current_residual = sum(
            post_gold_profile.values()
        )

        current_coverage = max(
            0.0,
            total_cost
            - current_residual,
        )

        tier1_coverage = 0.0
        residual_after_tier1 = 0.0

        for color in COLOR_ORDER:
            demand = (
                post_gold_profile[
                    color
                ]
            )

            supply = (
                tier1_supply[
                    color
                ]
            )

            covered = min(
                demand,
                supply,
            )

            tier1_coverage += (
                covered
            )

            residual_after_tier1 += max(
                0.0,
                demand
                - supply,
            )

        estimated_coverage = (
            current_coverage
            + tier1_coverage
        )

        coverage_ratio = (
            estimated_coverage
            / total_cost
            if total_cost > 0
            else 1.0
        )

        # Tier-2 support is deliberately secondary.
        tier2_support_value = 0.0

        for color in COLOR_ORDER:
            remaining_after_t1 = max(
                0.0,
                post_gold_profile[
                    color
                ]
                - tier1_supply[
                    color
                ],
            )

            tier2_support_value += min(
                remaining_after_t1,
                tier2_support[
                    color
                ],
            )

        viable = (
            coverage_ratio
            >= self.tier3_min_coverage_ratio
            or residual_after_tier1
            <= self.tier3_max_residual_after_tier1
        )

        # All Tier-3 cards are meaningful scoring cards (3/4/5 VP).
        # Points matter, but route compatibility matters heavily too.
        route_score = (
            card.points
            * 2.0
            + coverage_ratio
            * 10.0
            + tier1_coverage
            * self.tier1_bridge_weight
            + tier2_support_value
            * self.tier2_bridge_weight
            - residual_after_tier1
            * 1.25
        )

        # Already owning the reservation makes it a more stable route.
        if source == "reserved":
            route_score += 2.0

        return {
            "card":
                card,

            "source":
                source,

            "slot":
                slot,

            "reserved_index":
                reserved_index,

            "points":
                card.points,

            "raw_profile":
                raw_profile,

            "post_gold_profile":
                post_gold_profile,

            "total_cost":
                total_cost,

            "current_residual":
                current_residual,

            "tier1_coverage":
                tier1_coverage,

            "tier2_support_value":
                tier2_support_value,

            "residual_after_tier1":
                residual_after_tier1,

            "coverage_ratio":
                coverage_ratio,

            "route_score":
                float(
                    route_score
                ),

            "viable":
                bool(
                    viable
                ),
        }

    # ============================================================
    # CURRENT RESOURCE COVERAGE
    # ============================================================

    def _uncovered_color_profile(
        self,
        player,
        card,
    ):
        """
        Cost still uncovered after CURRENT permanent bonuses and
        CURRENT colored gems.
        """

        profile = {}

        for color in COLOR_ORDER:
            profile[
                color
            ] = float(
                max(
                    0,
                    card.cost.get(
                        color,
                        0,
                    )
                    - player.bonuses[
                        color
                    ]
                    - player.gems[
                        color
                    ],
                )
            )

        return profile

    def _gold_count(
        self,
        player,
    ):
        try:
            return int(
                player.gems[
                    GemColor.GOLD
                ]
            )
        except Exception:
            return 0

    def _apply_gold_to_profile(
        self,
        profile,
        gold,
    ):
        """
        Gold is flexible.

        Allocate each gold against the currently largest remaining
        colored deficit. This gives a deterministic approximation of
        the best flexible coverage.
        """

        adjusted = {
            color:
                float(
                    value
                )
            for color, value
            in profile.items()
        }

        for _ in range(
            max(
                0,
                int(
                    gold
                ),
            )
        ):
            color = max(
                COLOR_ORDER,
                key=lambda c:
                    adjusted[
                        c
                    ],
            )

            if adjusted[
                color
            ] <= 0:
                break

            adjusted[
                color
            ] -= 1.0

        return adjusted

    # ============================================================
    # TIER-1 BRIDGE SUPPLY
    # ============================================================

    def _tier1_bridge_supply(
        self,
        state,
        player,
    ):
        """
        Estimate obtainable permanent Tier-1 bonus supply by color.

        A visible Tier-1 card contributes more when it is already
        affordable or only a small gem distance away.
        """

        supply = {
            color: 0.0
            for color in COLOR_ORDER
        }

        for card in state.visible_cards[
            1
        ]:
            if card is None:
                continue

            color = card.bonus_color

            distance = (
                self._distance_to_card(
                    player,
                    card,
                )
            )

            supply[
                color
            ] += (
                self._affordability_weight(
                    distance
                )
            )

        # Own reserved Tier-1 cards can also become bridge pieces.
        for card in player.reserved_cards:
            if card is None:
                continue

            if (
                self._card_tier(
                    card
                )
                != 1
            ):
                continue

            color = card.bonus_color

            distance = (
                self._distance_to_card(
                    player,
                    card,
                )
            )

            supply[
                color
            ] += (
                self._affordability_weight(
                    distance
                )
            )

        return supply

    # ============================================================
    # TIER-2 BRIDGE + POINT SUPPORT
    # ============================================================

    def _tier2_bridge_supply(
        self,
        state,
        player,
    ):
        """
        Tier-2 is not the core Tier-3 feasibility test.

        Instead it contributes secondary bridge support, boosted by
        the fact that Tier-2 can score points while improving the
        engine.
        """

        support = {
            color: 0.0
            for color in COLOR_ORDER
        }

        cards = list(
            state.visible_cards[
                2
            ]
        )

        cards.extend(
            card
            for card
            in player.reserved_cards
            if (
                card is not None
                and self._card_tier(
                    card
                )
                == 2
            )
        )

        for card in cards:
            if card is None:
                continue

            distance = (
                self._distance_to_card(
                    player,
                    card,
                )
            )

            affordability = (
                self._affordability_weight(
                    distance
                )
            )

            point_multiplier = (
                1.0
                + card.points
                * 0.20
            )

            support[
                card.bonus_color
            ] += (
                affordability
                * point_multiplier
                * 0.75
            )

        return support

    def _affordability_weight(
        self,
        distance,
    ):
        if distance <= 0:
            return 1.00
        if distance == 1:
            return 0.85
        if distance == 2:
            return 0.65
        if distance == 3:
            return 0.40
        if distance == 4:
            return 0.20

        return 0.05

    # ============================================================
    # PRIMARY ↔ SECONDARY COMPATIBILITY
    # ============================================================

    def _tier3_target_overlap(
        self,
        target_a,
        target_b,
    ):
        profile_a = target_a[
            "post_gold_profile"
        ]

        profile_b = target_b[
            "post_gold_profile"
        ]

        shared_demand = sum(
            min(
                profile_a[
                    color
                ],
                profile_b[
                    color
                ],
            )
            for color in COLOR_ORDER
        )

        denominator = max(
            1.0,
            min(
                sum(
                    profile_a.values()
                ),
                sum(
                    profile_b.values()
                ),
            ),
        )

        ratio = (
            shared_demand
            / denominator
        )

        shared_colors = {
            color
            for color in COLOR_ORDER
            if (
                profile_a[
                    color
                ] > 0
                and profile_b[
                    color
                ] > 0
            )
        }

        return {
            "shared_demand":
                float(
                    shared_demand
                ),

            "ratio":
                float(
                    ratio
                ),

            "shared_colors":
                shared_colors,
        }

    def _build_tier3_bridge_priority(
        self,
        primary,
        secondary,
    ):
        priority = {
            color: 0.0
            for color in COLOR_ORDER
        }

        primary_profile = (
            primary[
                "post_gold_profile"
            ]
        )

        for color in COLOR_ORDER:
            priority[
                color
            ] += (
                primary_profile[
                    color
                ]
            )

        if secondary is not None:
            secondary_profile = (
                secondary[
                    "post_gold_profile"
                ]
            )

            for color in COLOR_ORDER:
                priority[
                    color
                ] += (
                    0.80
                    * secondary_profile[
                        color
                    ]
                )

                if (
                    primary_profile[
                        color
                    ] > 0
                    and secondary_profile[
                        color
                    ] > 0
                ):
                    # Reusable engine bonus.
                    priority[
                        color
                    ] += 2.0

        return priority

    # ============================================================
    # MANDATORY TIER-3 RESERVE ACTIONS
    # ============================================================

    def _collect_mandatory_tier3_reserves(
        self,
        env,
        state,
        tier3_plan,
        excluded_ids,
    ):
        """
        If a qualified primary/secondary Tier-3 target is visible and
        reservable, force its RESERVE_VISIBLE action into rollout.
        """

        legal_actions = env._legal_actions(
            state
        )

        target_cards = []

        primary = tier3_plan[
            "primary"
        ]

        secondary = tier3_plan[
            "secondary"
        ]

        if (
            primary is not None
            and primary[
                "source"
            ] == "visible"
        ):
            target_cards.append(
                (
                    "primary",
                    primary,
                )
            )

        if (
            secondary is not None
            and secondary[
                "source"
            ] == "visible"
        ):
            target_cards.append(
                (
                    "secondary",
                    secondary,
                )
            )

        records = []

        for label, target in target_cards:
            target_card = target[
                "card"
            ]

            for action in legal_actions:

                if (
                    action.action_type
                    != ActionType.RESERVE_VISIBLE
                ):
                    continue

                identity = (
                    self._action_identity(
                        action
                    )
                )

                if identity in excluded_ids:
                    continue

                card = (
                    self._get_card_from_action(
                        state,
                        action,
                    )
                )

                if not self._same_card(
                    card,
                    target_card,
                ):
                    continue

                # High enough to be useful as a tie-break, but
                # terminal result still decides.
                score = (
                    25.0
                    + target[
                        "route_score"
                    ]
                    + (
                        2.0
                        if label
                        == "primary"
                        else 0.0
                    )
                )

                records.append(
                    {
                        "action":
                            action,

                        "score":
                            float(
                                score
                            ),

                        "sources":
                            {
                                f"tier3_reserve_{label}"
                            },

                        "mandatory":
                            True,

                        "tier3_target":
                            label,
                    }
                )

                break

        return records

    # ============================================================
    # TIER-3 BRIDGE CANDIDATES
    # ============================================================

    def _collect_tier3_bridge_records(
        self,
        env,
        state,
        tier3_plan,
        noble_info,
        excluded_ids,
    ):
        legal_actions = env._legal_actions(
            state
        )

        records = []

        for action in legal_actions:

            identity = (
                self._action_identity(
                    action
                )
            )

            if identity in excluded_ids:
                continue

            score = (
                self._score_tier3_bridge_action(
                    state=state,
                    action=action,
                    tier3_plan=tier3_plan,
                    noble_info=noble_info,
                )
            )

            if score is None:
                continue

            sources = {
                "tier3_bridge"
            }

            # Mark explicit cross-route actions.
            if (
                noble_info is not None
                and self._action_supports_noble_route(
                    state=state,
                    action=action,
                    noble_info=noble_info,
                )
            ):
                sources.add(
                    "noble"
                )

            records.append(
                {
                    "action":
                        action,

                    "score":
                        float(
                            score
                        ),

                    "sources":
                        sources,

                    "mandatory":
                        False,
                }
            )

        records.sort(
            key=lambda record:
                record[
                    "score"
                ],
            reverse=True,
        )

        return records[
            :self.num_tier3_bridge_moves
        ]

    def _score_tier3_bridge_action(
        self,
        state,
        action,
        tier3_plan,
        noble_info,
    ):
        player = state.players[
            state.current_player
        ]

        bridge_priority = tier3_plan[
            "bridge_priority"
        ]

        action_type = (
            action.action_type
        )

        # --------------------------------------------------------
        # BUY TIER-1 / TIER-2 BRIDGE
        # --------------------------------------------------------

        if action_type in (
            ActionType.BUY_VISIBLE,
            ActionType.BUY_RESERVED,
        ):
            card = (
                self._get_card_from_action(
                    state,
                    action,
                )
            )

            if card is None:
                return None

            tier = self._action_tier(
                state,
                action,
            )

            if tier not in (
                1,
                2,
            ):
                return None

            demand = bridge_priority.get(
                card.bonus_color,
                0.0,
            )

            if demand <= 0:
                return None

            distance = (
                self._distance_to_card(
                    player,
                    card,
                )
            )

            if tier == 1:
                score = (
                    demand
                    * 3.0
                    + 4.0
                    - distance
                )

            else:
                # Tier-2 = bridge + points.
                score = (
                    demand
                    * 2.4
                    + card.points
                    * self.tier2_point_weight
                    + 2.0
                    - distance
                    * 0.75
                )

            score += (
                self._cross_route_card_bonus(
                    player=player,
                    card=card,
                    noble_info=noble_info,
                )
            )

            return score

        # --------------------------------------------------------
        # TAKE_GEMS
        # --------------------------------------------------------

        if (
            action_type
            == ActionType.TAKE_GEMS
        ):
            gems_after = dict(
                player.gems
            )

            for color in action.gem_colors:
                gems_after[
                    color
                ] += 1

            best_bridge_progress = 0.0

            # T1/T2 cards that build the engine.
            for tier in (
                1,
                2,
            ):
                for card in state.visible_cards[
                    tier
                ]:
                    if card is None:
                        continue

                    demand = bridge_priority.get(
                        card.bonus_color,
                        0.0,
                    )

                    if demand <= 0:
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
                            gems_after,
                        )
                    )

                    progress = (
                        before
                        - after
                    )

                    if progress <= 0:
                        continue

                    if tier == 1:
                        card_value = (
                            demand
                            * 3.0
                            + 3.0
                        )

                    else:
                        card_value = (
                            demand
                            * 2.2
                            + card.points
                            * self.tier2_point_weight
                        )

                    card_value += (
                        self._cross_route_card_bonus(
                            player=player,
                            card=card,
                            noble_info=noble_info,
                        )
                    )

                    best_bridge_progress = max(
                        best_bridge_progress,
                        card_value
                        + progress
                        * 4.0,
                    )

            # Also allow direct gem progress toward the actual T3
            # destinations once the engine is close enough.
            best_direct_t3_progress = 0.0

            for key in (
                "primary",
                "secondary",
            ):
                target = tier3_plan.get(
                    key,
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
                        gems_after,
                    )
                )

                progress = (
                    before
                    - after
                )

                if progress <= 0:
                    continue

                direct_value = (
                    target[
                        "route_score"
                    ]
                    * 0.35
                    + progress
                    * 4.0
                    * self.direct_tier3_progress_weight
                )

                best_direct_t3_progress = max(
                    best_direct_t3_progress,
                    direct_value,
                )

            best = max(
                best_bridge_progress,
                best_direct_t3_progress,
            )

            return (
                best
                if best > 0
                else None
            )

        # --------------------------------------------------------
        # RESERVE TIER-2 BRIDGE / POINT CARD
        # --------------------------------------------------------

        if (
            action_type
            == ActionType.RESERVE_VISIBLE
        ):
            card = (
                self._get_card_from_action(
                    state,
                    action,
                )
            )

            if card is None:
                return None

            tier = self._action_tier(
                state,
                action,
            )

            # Tier-3 target reserves are handled separately and
            # guaranteed when route-qualified.
            if tier == 3:
                return None

            if tier != 2:
                return None

            demand = bridge_priority.get(
                card.bonus_color,
                0.0,
            )

            if demand <= 0:
                return None

            distance = (
                self._distance_to_card(
                    player,
                    card,
                )
            )

            score = (
                demand
                * 1.5
                + card.points
                * self.tier2_point_weight
                - distance
                * 0.75
            )

            score += (
                self._cross_route_card_bonus(
                    player=player,
                    card=card,
                    noble_info=noble_info,
                )
            )

            return score

        return None

    # ============================================================
    # NOBLE ↔ TIER-3 CROSS ROUTE
    # ============================================================

    def _cross_route_card_bonus(
        self,
        player,
        card,
        noble_info,
    ):
        if noble_info is None:
            return 0.0

        targets = noble_info.get(
            "target_nobles",
            [],
        )

        if not targets:
            return 0.0

        alignment = (
            self._noble_card_alignment(
                player=player,
                card=card,
                target_nobles=targets,
                overlap_colors=noble_info.get(
                    "overlap_colors",
                    set(),
                ),
                overlap_count=noble_info.get(
                    "overlap_count",
                    0,
                ),
            )
        )

        if alignment <= 0:
            return 0.0

        return min(
            self.cross_route_synergy_bonus,
            alignment
            * 0.50,
        )

    def _action_supports_noble_route(
        self,
        state,
        action,
        noble_info,
    ):
        if noble_info is None:
            return False

        if action.action_type not in (
            ActionType.BUY_VISIBLE,
            ActionType.BUY_RESERVED,
            ActionType.RESERVE_VISIBLE,
        ):
            return False

        card = (
            self._get_card_from_action(
                state,
                action,
            )
        )

        if card is None:
            return False

        player = state.players[
            state.current_player
        ]

        return (
            self._cross_route_card_bonus(
                player=player,
                card=card,
                noble_info=noble_info,
            )
            > 0
        )

    # ============================================================
    # EXTRA RECORD MERGING
    # ============================================================

    def _merge_extra_records(
        self,
        records,
    ):
        merged = {}

        for record in records:
            identity = (
                self._action_identity(
                    record[
                        "action"
                    ]
                )
            )

            if identity not in merged:
                merged[
                    identity
                ] = {
                    "action":
                        record[
                            "action"
                        ],

                    "score":
                        float(
                            record[
                                "score"
                            ]
                        ),

                    "sources":
                        set(
                            record.get(
                                "sources",
                                set(),
                            )
                        ),

                    "mandatory":
                        bool(
                            record.get(
                                "mandatory",
                                False,
                            )
                        ),
                }

                if (
                    "tier3_target"
                    in record
                ):
                    merged[
                        identity
                    ][
                        "tier3_target"
                    ] = record[
                        "tier3_target"
                    ]

            else:
                existing = merged[
                    identity
                ]

                existing[
                    "score"
                ] = max(
                    existing[
                        "score"
                    ],
                    float(
                        record[
                            "score"
                        ]
                    ),
                )

                existing[
                    "sources"
                ].update(
                    record.get(
                        "sources",
                        set(),
                    )
                )

                existing[
                    "mandatory"
                ] = (
                    existing[
                        "mandatory"
                    ]
                    or bool(
                        record.get(
                            "mandatory",
                            False,
                        )
                    )
                )

        return list(
            merged.values()
        )

    # ============================================================
    # CARD HELPERS
    # ============================================================

    def _card_tier(
        self,
        card,
    ):
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

    def _same_card(
        self,
        card_a,
        card_b,
    ):
        if (
            card_a is None
            or card_b is None
        ):
            return False

        if card_a is card_b:
            return True

        try:
            return (
                card_a == card_b
            )
        except Exception:
            return False

    # ============================================================
    # DEBUG API
    # ============================================================

    def get_kangaroo_reservation_debug(
        self,
    ):
        return {
            "strategy":
                self.last_strategy_debug,

            "candidates":
                self.last_candidate_debug,
        }


class HeuristicAgent14Diagnostics(
    HeuristicAgent14
):
    """
    Diagnostic version of H14.

    Gameplay logic is identical.
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

    def reset_diagnostic_stats(
        self,
    ):
        self._diag14 = {
            "strategic_decisions":
                0,

            "forced_node_calls":
                0,

            "route_state_counts":
                Counter(),

            "selected_source_counts":
                Counter(),

            "extra_source_counts":
                Counter(),

            "extra_action_type_counts":
                Counter(),

            "tier3_primary_only":
                0,

            "tier3_with_secondary":
                0,

            "mandatory_t3_reserve_candidates":
                0,

            "mandatory_t3_reserve_selected":
                0,

            "extras_available":
                0,

            "extras_proposed":
                0,
        }

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

        self._record_h14_decision(
            state=state,
            selected_action=action,
        )

        return action

    def _record_h14_decision(
        self,
        state,
        selected_action,
    ):
        if (
            state.node_type
            != NodeType.MAIN_DECISION
        ):
            self._diag14[
                "forced_node_calls"
            ] += 1
            return

        self._diag14[
            "strategic_decisions"
        ] += 1

        strategy = (
            self.last_strategy_debug
            or {}
        )

        candidates = (
            self.last_candidate_debug
            or {}
        )

        noble = bool(
            strategy.get(
                "noble_qualified",
                False,
            )
        )

        tier3 = bool(
            strategy.get(
                "tier3_qualified",
                False,
            )
        )

        if noble and tier3:
            route_state = "both"

        elif noble:
            route_state = "noble_only"

        elif tier3:
            route_state = "tier3_only"

        else:
            route_state = "neither"

        self._diag14[
            "route_state_counts"
        ][route_state] += 1

        tier3_plan = strategy.get(
            "tier3",
            {},
        )

        if tier3:
            if (
                tier3_plan.get(
                    "secondary",
                    None,
                )
                is not None
            ):
                self._diag14[
                    "tier3_with_secondary"
                ] += 1

            else:
                self._diag14[
                    "tier3_primary_only"
                ] += 1

        extra_records = candidates.get(
            "extra_records",
            [],
        )

        if extra_records:
            self._diag14[
                "extras_available"
            ] += 1

        self._diag14[
            "extras_proposed"
        ] += len(
            extra_records
        )

        mandatory_records = [
            record
            for record in extra_records
            if record.get(
                "mandatory",
                False,
            )
        ]

        self._diag14[
            "mandatory_t3_reserve_candidates"
        ] += len(
            mandatory_records
        )

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
            in candidates.get(
                "base_candidates",
                [],
            )
        }

        selected_record = None

        for record in extra_records:
            if (
                self._action_identity(
                    record[
                        "action"
                    ]
                )
                == selected_id
            ):
                selected_record = (
                    record
                )
                break

        if selected_id in base_ids:
            source = "h4_base"

        elif selected_record is not None:
            source = "strategic_extra"

        else:
            source = "unknown"

        self._diag14[
            "selected_source_counts"
        ][source] += 1

        if selected_record is not None:
            for route_source in selected_record.get(
                "sources",
                set(),
            ):
                self._diag14[
                    "extra_source_counts"
                ][route_source] += 1

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

            self._diag14[
                "extra_action_type_counts"
            ][action_type_name] += 1

            if selected_record.get(
                "mandatory",
                False,
            ):
                self._diag14[
                    "mandatory_t3_reserve_selected"
                ] += 1

    def get_diagnostic_stats(
        self,
    ):
        decisions = self._diag14[
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

        return {
            "strategic_decisions":
                decisions,

            "forced_node_calls":
                self._diag14[
                    "forced_node_calls"
                ],

            "route_state_counts":
                dict(
                    self._diag14[
                        "route_state_counts"
                    ]
                ),

            "route_state_percentages": {
                key:
                    pct(
                        value
                    )
                for key, value
                in self._diag14[
                    "route_state_counts"
                ].items()
            },

            "selected_source_counts":
                dict(
                    self._diag14[
                        "selected_source_counts"
                    ]
                ),

            "selected_source_percentages": {
                key:
                    pct(
                        value
                    )
                for key, value
                in self._diag14[
                    "selected_source_counts"
                ].items()
            },

            "extra_source_counts":
                dict(
                    self._diag14[
                        "extra_source_counts"
                    ]
                ),

            "extra_action_type_counts":
                dict(
                    self._diag14[
                        "extra_action_type_counts"
                    ]
                ),

            "tier3_primary_only":
                self._diag14[
                    "tier3_primary_only"
                ],

            "tier3_with_secondary":
                self._diag14[
                    "tier3_with_secondary"
                ],

            "mandatory_t3_reserve_candidates":
                self._diag14[
                    "mandatory_t3_reserve_candidates"
                ],

            "mandatory_t3_reserve_selected":
                self._diag14[
                    "mandatory_t3_reserve_selected"
                ],

            "extras_available":
                self._diag14[
                    "extras_available"
                ],

            "extras_available_pct":
                pct(
                    self._diag14[
                        "extras_available"
                    ]
                ),

            "average_extras_proposed":
                (
                    self._diag14[
                        "extras_proposed"
                    ]
                    / decisions
                    if decisions
                    else 0.0
                ),
        }

    def format_diagnostic_summary(
        self,
    ):
        stats = (
            self.get_diagnostic_stats()
        )

        lines = [
            "=== HeuristicAgent14 Kangaroo Reservation Diagnostics ===",
            f"Strategic decisions: {stats['strategic_decisions']}",
            f"Forced-node calls: {stats['forced_node_calls']}",
            "",
            "Route qualification:",
        ]

        for key in (
            "noble_only",
            "tier3_only",
            "both",
            "neither",
        ):
            count = stats[
                "route_state_counts"
            ].get(
                key,
                0,
            )

            pct = stats[
                "route_state_percentages"
            ].get(
                key,
                0.0,
            )

            lines.append(
                f"  {key}: {count} ({pct:.2f}%)"
            )

        lines.extend(
            [
                "",
                "Final selected action source:",
            ]
        )

        for key in (
            "h4_base",
            "strategic_extra",
            "unknown",
        ):
            count = stats[
                "selected_source_counts"
            ].get(
                key,
                0,
            )

            if (
                key == "unknown"
                and count == 0
            ):
                continue

            pct = stats[
                "selected_source_percentages"
            ].get(
                key,
                0.0,
            )

            lines.append(
                f"  {key}: {count} ({pct:.2f}%)"
            )

        lines.extend(
            [
                "",
                "Selected strategic extras by route source:",
            ]
        )

        if stats[
            "extra_source_counts"
        ]:
            for key, count in sorted(
                stats[
                    "extra_source_counts"
                ].items()
            ):
                lines.append(
                    f"  {key}: {count}"
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

        if stats[
            "extra_action_type_counts"
        ]:
            for key, count in sorted(
                stats[
                    "extra_action_type_counts"
                ].items()
            ):
                lines.append(
                    f"  {key}: {count}"
                )
        else:
            lines.append(
                "  none"
            )

        lines.extend(
            [
                "",
                "Tier-3 route shape:",
                (
                    "  Primary only: "
                    f"{stats['tier3_primary_only']}"
                ),
                (
                    "  Primary + secondary: "
                    f"{stats['tier3_with_secondary']}"
                ),
                "",
                "Mandatory Tier-3 reserve investigation:",
                (
                    "  Reserve candidates inserted: "
                    f"{stats['mandatory_t3_reserve_candidates']}"
                ),
                (
                    "  Mandatory reserve actions actually selected: "
                    f"{stats['mandatory_t3_reserve_selected']}"
                ),
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
