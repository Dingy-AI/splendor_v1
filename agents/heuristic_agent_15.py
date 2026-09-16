import random
from collections import Counter

import numpy as np

from splendor_v1.agents.heuristic_agent_3 import HeuristicAgent3
from splendor_v1.env.core.constants import COLOR_ORDER
from splendor_v1.env.core.enums import GemColor, NodeType
from splendor_v1.env.core.actions import ActionType


class HeuristicAgent15:
    """
    HeuristicAgent15 — Unified Route Planner
    =========================================

    H15 is intentionally NOT an H12/H14 subclass.

    It keeps only the low-level ideas that have repeatedly tested well:

        - HeuristicAgent3 as the cheap full-game continuation policy
        - full terminal rollouts
        - hidden-deck determinization
        - common random numbers across root candidates

    Candidate generation is rebuilt from scratch around one idea:

        CURRENT RESOURCES
            ↓
        REACHABLE ENGINE
            ↓
        REACHABLE SCORING DESTINATIONS

    There are no mutually exclusive "Noble mode" and "Reserve mode"
    labels.

    Noble routes, Tier-2 bridge/point cards, and Tier-3 routes all
    contribute to a single strategic model of the board.

    ----------------------------------------------------------------
    STRATEGIC MODEL
    ----------------------------------------------------------------

    H15 constructs a reusable COLOR DEMAND vector:

        demand[color] =
            noble-route demand
            + Tier-3 primary demand
            + Tier-3 secondary demand
            + overlap / cross-route synergy

    A permanent bonus is valuable when that color participates in
    multiple reachable future scoring routes.

    Example:

        Blue helps:
            Noble A
            Noble B
            Primary T3
            Secondary T3

        → Blue becomes an extremely valuable engine color.

    This allows Kangaroo-like Noble play or reservation/high-point
    play to EMERGE from the board instead of being hard-selected.

    ----------------------------------------------------------------
    TIER 1 / TIER 2 / TIER 3
    ----------------------------------------------------------------

        Tier 1:
            cheap engine infrastructure

        Tier 2:
            bridge + immediate points

        Tier 3:
            major scoring destinations

    Tier-3 feasibility uses:

        current permanent bonuses
        + current colored gems
        + current gold
        + obtainable Tier-1 bonus bridge

    Tier-2 cards improve the route but are not required for basic
    Tier-3 feasibility.

    ----------------------------------------------------------------
    TIER-3 TARGETS
    ----------------------------------------------------------------

    H15 first finds ONE best viable Tier-3 primary.

    Then it searches for a secondary that:

        - is also viable
        - is only slightly worse than the primary
        - can reuse enough of the same engine

    The secondary is a route-strengthening signal, not a requirement.

    If a visible primary Tier-3 qualifies:
        its RESERVE_VISIBLE action is a mandatory rollout candidate.

    If a compatible visible secondary qualifies:
        its RESERVE_VISIBLE action is also a mandatory rollout candidate.

    ----------------------------------------------------------------
    TIER-3 ARCHETYPES
    ----------------------------------------------------------------

    Base-game Tier-3 cards are explicitly recognized as:

        3 VP: 5 + 3 + 3 + 3     → BROAD
        4 VP: 7                 → CONCENTRATED
        4 VP: 6 + 3 + 3         → THREE_COLOR
        5 VP: 7 + 3             → TWO_COLOR

    The route scorer includes an engine-shape-fit term so the same
    player state can value these archetypes differently.

    ----------------------------------------------------------------
    NOBLES
    ----------------------------------------------------------------

    Nobles are not a separate strategy.

    Every remaining Noble contributes demand according to:

        - how close the current permanent-bonus engine is
        - which colors are still missing
        - whether two promising Nobles overlap on colors

    If Noble demand and Tier-3 demand point toward the same color,
    that cross-route agreement naturally raises the value of buying
    that permanent bonus.

    ----------------------------------------------------------------
    ACTIONS
    ----------------------------------------------------------------

    BUY:
        immediate points
        + permanent bonus route value
        + Noble completion
        + T1/T2/T3 role value

    TAKE:
        progress toward high-value bridge cards
        + direct progress toward T3 destinations
        + color-demand value
        + small two-player bank-denial value

    RESERVE:
        secure T3 primary/secondary
        + gold access
        + useful T2 bridge/point reservation
        + opponent denial
        - low-value reserve penalty

    These are CHEAP ROOT SCORES ONLY.

    Full terminal rollout remains the final decision-maker.

    ----------------------------------------------------------------
    CANDIDATE SELECTION
    ----------------------------------------------------------------

    H9 showed that broad top-N pruning can be dangerous.

    H15 therefore uses a DIVERSE shortlist:

        mandatory:
            immediate winning buys
            qualified primary T3 reserve
            qualified secondary T3 reserve

        soft seed allocation:
            BUY     up to 3
            TAKE    up to 2
            RESERVE up to 1

        remaining slots:
            highest unified score regardless of category

    Default:
        num_calc_moves = 8
        num_rollouts = 16

    Mandatory strategic actions may exceed num_calc_moves.

    ----------------------------------------------------------------
    PURPOSE
    ----------------------------------------------------------------

    H15 is a fresh candidate generator.

    The benchmark to beat is H12_16.
    """

    WIN_POINTS = 15

    T3_BROAD = "broad_5_3_3_3"
    T3_CONCENTRATED = "concentrated_7"
    T3_THREE_COLOR = "three_color_6_3_3"
    T3_TWO_COLOR = "two_color_7_3"
    T3_OTHER = "other"

    def __init__(
        self,
        num_rollouts=16,
        num_calc_moves=8,
        max_rollout_steps=200,
        random_seed=None,

        # Diverse shortlist.
        buy_seed_slots=3,
        take_seed_slots=2,
        reserve_seed_slots=1,
        candidate_score_tolerance=10.0,

        # Tier-3 feasibility / pair.
        tier3_min_coverage_ratio=0.55,
        tier3_max_residual_after_tier1=5.0,
        secondary_score_tolerance=4.0,
        secondary_min_overlap_ratio=0.30,

        # Unified-route weights.
        noble_demand_weight=1.0,
        tier3_primary_demand_weight=1.0,
        tier3_secondary_demand_weight=0.80,
        shared_t3_color_bonus=2.0,
        noble_overlap_bonus=1.75,
        cross_route_color_bonus=1.5,

        # Action scoring.
        permanent_bonus_weight=2.8,
        tier2_point_weight=3.0,
        tier3_point_weight=6.0,
        noble_completion_bonus=12.0,
        reserve_gold_bonus=2.0,
        bank_denial_weight=0.75,
    ):
        self.num_rollouts = max(
            1,
            int(num_rollouts),
        )

        self.num_calc_moves = max(
            1,
            int(num_calc_moves),
        )

        self.max_rollout_steps = max(
            1,
            int(max_rollout_steps),
        )

        self.rng = random.Random(
            random_seed
        )

        self.buy_seed_slots = max(
            0,
            int(buy_seed_slots),
        )

        self.take_seed_slots = max(
            0,
            int(take_seed_slots),
        )

        self.reserve_seed_slots = max(
            0,
            int(reserve_seed_slots),
        )

        self.candidate_score_tolerance = float(
            candidate_score_tolerance
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

        self.noble_demand_weight = float(
            noble_demand_weight
        )

        self.tier3_primary_demand_weight = float(
            tier3_primary_demand_weight
        )

        self.tier3_secondary_demand_weight = float(
            tier3_secondary_demand_weight
        )

        self.shared_t3_color_bonus = float(
            shared_t3_color_bonus
        )

        self.noble_overlap_bonus = float(
            noble_overlap_bonus
        )

        self.cross_route_color_bonus = float(
            cross_route_color_bonus
        )

        self.permanent_bonus_weight = float(
            permanent_bonus_weight
        )

        self.tier2_point_weight = float(
            tier2_point_weight
        )

        self.tier3_point_weight = float(
            tier3_point_weight
        )

        self.noble_completion_bonus = float(
            noble_completion_bonus
        )

        self.reserve_gold_bonus = float(
            reserve_gold_bonus
        )

        self.bank_denial_weight = float(
            bank_denial_weight
        )

        self.h3 = HeuristicAgent3()

        self.last_board_model = None
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

        action, _, _ = max(
            evaluated,
            key=lambda item: (
                item[1],
                item[2],
            ),
        )

        return action

    def get_policy(
        self,
        env,
        state,
        action_size=1139,
        temperature=0.25,
    ):
        if temperature <= 0:
            raise ValueError(
                "temperature must be > 0"
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

        values = np.array(
            [
                value
                for _, value, _
                in evaluated
            ],
            dtype=np.float64,
        )

        logits = values / temperature
        logits -= np.max(
            logits
        )

        probs = np.exp(
            logits
        )

        probs /= probs.sum()

        for (
            action,
            _,
            _,
        ), prob in zip(
            evaluated,
            probs,
        ):
            action_id = env.action_to_id(
                action
            )

            policy[
                action_id
            ] += prob

        return policy

    # ============================================================
    # ROOT CANDIDATES
    # ============================================================

    def _get_root_candidates(
        self,
        env,
        state,
    ):
        legal_actions = env._legal_actions(
            state
        )

        if not legal_actions:
            return []

        # Forced discard/noble transitions are not strategic route
        # decisions. Use H3's proven forced-node handling.
        if (
            state.node_type
            != NodeType.MAIN_DECISION
        ):
            candidates = (
                self.h3._get_scored_candidates(
                    env,
                    state,
                )
            )

            self.last_candidate_debug = {
                "forced_node": True,
                "selected_count": len(
                    candidates
                ),
            }

            return candidates

        board_model = (
            self._build_board_model(
                state
            )
        )

        self.last_board_model = (
            board_model
        )

        records = []

        for action in legal_actions:
            record = (
                self._score_legal_action(
                    state=state,
                    action=action,
                    board_model=board_model,
                )
            )

            if record is not None:
                records.append(
                    record
                )

        if not records:
            return []

        mandatory = [
            record
            for record in records
            if record[
                "mandatory"
            ]
        ]

        mandatory = (
            self._dedupe_records(
                mandatory
            )
        )

        selected = list(
            mandatory
        )

        selected_ids = {
            self._action_identity(
                record[
                    "action"
                ]
            )
            for record in selected
        }

        normal_budget = max(
            0,
            self.num_calc_moves
            - len(
                selected
            ),
        )

        groups = {
            "buy": [],
            "take": [],
            "reserve": [],
            "other": [],
        }

        for record in records:
            identity = (
                self._action_identity(
                    record[
                        "action"
                    ]
                )
            )

            if identity in selected_ids:
                continue

            groups[
                record[
                    "group"
                ]
            ].append(
                record
            )

        for group_records in groups.values():
            group_records.sort(
                key=lambda record:
                    record[
                        "score"
                    ],
                reverse=True,
            )

        seed_plan = [
            (
                "buy",
                self.buy_seed_slots,
            ),
            (
                "take",
                self.take_seed_slots,
            ),
            (
                "reserve",
                self.reserve_seed_slots,
            ),
        ]

        # First pass: diverse category seeds.
        for group_name, slots in seed_plan:
            if normal_budget <= 0:
                break

            plausible = (
                self._plausible_group_records(
                    groups[
                        group_name
                    ]
                )
            )

            add_count = min(
                slots,
                normal_budget,
                len(
                    plausible
                ),
            )

            for record in plausible[
                :add_count
            ]:
                selected.append(
                    record
                )

                selected_ids.add(
                    self._action_identity(
                        record[
                            "action"
                        ]
                    )
                )

            normal_budget -= (
                add_count
            )

        # Second pass: unified global ranking.
        if normal_budget > 0:
            leftovers = []

            for group_records in groups.values():
                for record in group_records:
                    identity = (
                        self._action_identity(
                            record[
                                "action"
                            ]
                        )
                    )

                    if identity in selected_ids:
                        continue

                    leftovers.append(
                        record
                    )

            leftovers.sort(
                key=lambda record:
                    record[
                        "score"
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
                        record[
                            "action"
                        ]
                    )
                )

                normal_budget -= 1

        self.last_candidate_debug = {
            "forced_node":
                False,

            "legal_count":
                len(
                    legal_actions
                ),

            "record_count":
                len(
                    records
                ),

            "mandatory_count":
                len(
                    mandatory
                ),

            "selected_count":
                len(
                    selected
                ),

            "num_calc_moves":
                self.num_calc_moves,

            "group_counts": {
                group:
                    len(
                        rows
                    )
                for group, rows
                in groups.items()
            },

            "selected_records":
                selected,
        }

        return [
            (
                record[
                    "action"
                ],
                float(
                    record[
                        "score"
                    ]
                ),
            )
            for record in selected
        ]

    def _plausible_group_records(
        self,
        records,
    ):
        if not records:
            return []

        best = records[
            0
        ][
            "score"
        ]

        cutoff = (
            best
            - self.candidate_score_tolerance
        )

        return [
            record
            for record in records
            if record[
                "score"
            ] >= cutoff
        ]

    def _dedupe_records(
        self,
        records,
    ):
        best = {}

        for record in records:
            identity = (
                self._action_identity(
                    record[
                        "action"
                    ]
                )
            )

            if (
                identity not in best
                or record[
                    "score"
                ] > best[
                    identity
                ][
                    "score"
                ]
            ):
                best[
                    identity
                ] = record

        return list(
            best.values()
        )

    # ============================================================
    # BOARD MODEL
    # ============================================================

    def _build_board_model(
        self,
        state,
    ):
        player = state.players[
            state.current_player
        ]

        tier1_supply = (
            self._tier1_bridge_supply(
                state,
                player,
            )
        )

        noble_model = (
            self._build_noble_model(
                state,
                player,
            )
        )

        tier3_model = (
            self._build_tier3_model(
                state=state,
                player=player,
                tier1_supply=tier1_supply,
            )
        )

        color_demand = {
            color: 0.0
            for color in COLOR_ORDER
        }

        noble_demand = (
            noble_model[
                "color_demand"
            ]
        )

        tier3_demand = (
            tier3_model[
                "color_demand"
            ]
        )

        for color in COLOR_ORDER:
            noble_component = (
                noble_demand[
                    color
                ]
                * self.noble_demand_weight
            )

            tier3_component = (
                tier3_demand[
                    color
                ]
            )

            demand = (
                noble_component
                + tier3_component
            )

            # If BOTH major route families want the same color,
            # permanent bonuses of that color have extra reuse value.
            if (
                noble_component > 0
                and tier3_component > 0
            ):
                demand += (
                    self.cross_route_color_bonus
                )

            color_demand[
                color
            ] = float(
                demand
            )

        return {
            "player":
                player,

            "tier1_supply":
                tier1_supply,

            "noble":
                noble_model,

            "tier3":
                tier3_model,

            "color_demand":
                color_demand,
        }

    # ============================================================
    # NOBLE MODEL
    # ============================================================

    def _build_noble_model(
        self,
        state,
        player,
    ):
        nobles = [
            noble
            for noble in state.nobles
            if noble is not None
        ]

        color_demand = {
            color: 0.0
            for color in COLOR_ORDER
        }

        noble_records = []

        for noble in nobles:
            missing = {
                color:
                    max(
                        0,
                        noble.requirement[
                            color
                        ]
                        - player.bonuses[
                            color
                        ],
                    )
                for color in COLOR_ORDER
            }

            total_missing = sum(
                missing.values()
            )

            # Close nobles should contribute much more than distant
            # ones, but distant nobles still provide weak structural
            # information.
            closeness = max(
                0.20,
                1.75
                - 0.10
                * total_missing,
            )

            for color in COLOR_ORDER:
                if missing[
                    color
                ] > 0:
                    color_demand[
                        color
                    ] += (
                        closeness
                        * min(
                            2.0,
                            missing[
                                color
                            ],
                        )
                    )

            noble_records.append(
                {
                    "noble":
                        noble,

                    "missing":
                        missing,

                    "total_missing":
                        total_missing,

                    "closeness":
                        closeness,
                }
            )

        # Explicitly identify the strongest overlapping pair.
        best_pair = None

        for i in range(
            len(
                noble_records
            )
        ):
            for j in range(
                i + 1,
                len(
                    noble_records
                ),
            ):
                a = noble_records[
                    i
                ]

                b = noble_records[
                    j
                ]

                overlap_colors = {
                    color
                    for color in COLOR_ORDER
                    if (
                        a[
                            "noble"
                        ].requirement[
                            color
                        ] > 0
                        and b[
                            "noble"
                        ].requirement[
                            color
                        ] > 0
                    )
                }

                rank = (
                    len(
                        overlap_colors
                    ),
                    -(
                        a[
                            "total_missing"
                        ]
                        + b[
                            "total_missing"
                        ]
                    ),
                )

                if (
                    best_pair is None
                    or rank
                    > best_pair[
                        "rank"
                    ]
                ):
                    best_pair = {
                        "rank":
                            rank,

                        "a":
                            a,

                        "b":
                            b,

                        "overlap_colors":
                            overlap_colors,

                        "overlap_count":
                            len(
                                overlap_colors
                            ),
                    }

        # Base-game Noble overlap maxes at 2 colors. Two-color
        # overlap is useful, but not sufficient by itself to create
        # a separate "mode". It simply raises reusable color demand.
        if (
            best_pair is not None
            and best_pair[
                "overlap_count"
            ] >= 2
        ):
            for color in best_pair[
                "overlap_colors"
            ]:
                still_needed = (
                    best_pair[
                        "a"
                    ][
                        "missing"
                    ][
                        color
                    ] > 0
                    or best_pair[
                        "b"
                    ][
                        "missing"
                    ][
                        color
                    ] > 0
                )

                if still_needed:
                    color_demand[
                        color
                    ] += (
                        self.noble_overlap_bonus
                    )

        return {
            "records":
                noble_records,

            "best_pair":
                best_pair,

            "color_demand":
                color_demand,
        }

    # ============================================================
    # TIER-3 MODEL
    # ============================================================

    def _build_tier3_model(
        self,
        state,
        player,
        tier1_supply,
    ):
        targets = []

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
                    source="visible",
                    slot=slot,
                )
            )

        for reserved_index, card in enumerate(
            player.reserved_cards
        ):
            if card is None:
                continue

            if (
                self._card_tier(
                    card
                )
                != 3
            ):
                continue

            targets.append(
                self._score_tier3_target(
                    player=player,
                    card=card,
                    tier1_supply=tier1_supply,
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

        viable = [
            target
            for target in targets
            if target[
                "viable"
            ]
        ]

        primary = (
            viable[
                0
            ]
            if viable
            else None
        )

        secondary = None

        if primary is not None:
            best_rank = None

            for candidate in viable[
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
                    best_rank is None
                    or rank > best_rank
                ):
                    best_rank = (
                        rank
                    )

                    secondary = dict(
                        candidate
                    )

                    secondary[
                        "overlap_with_primary"
                    ] = overlap

        color_demand = {
            color: 0.0
            for color in COLOR_ORDER
        }

        if primary is not None:
            for color in COLOR_ORDER:
                color_demand[
                    color
                ] += (
                    primary[
                        "post_gold_profile"
                    ][
                        color
                    ]
                    * self.tier3_primary_demand_weight
                )

        if secondary is not None:
            for color in COLOR_ORDER:
                color_demand[
                    color
                ] += (
                    secondary[
                        "post_gold_profile"
                    ][
                        color
                    ]
                    * self.tier3_secondary_demand_weight
                )

                if (
                    primary[
                        "post_gold_profile"
                    ][
                        color
                    ] > 0
                    and secondary[
                        "post_gold_profile"
                    ][
                        color
                    ] > 0
                ):
                    color_demand[
                        color
                    ] += (
                        self.shared_t3_color_bonus
                    )

        return {
            "targets":
                targets,

            "primary":
                primary,

            "secondary":
                secondary,

            "qualified":
                primary is not None,

            "color_demand":
                color_demand,
        }

    def _score_tier3_target(
        self,
        player,
        card,
        tier1_supply,
        source,
        slot=None,
        reserved_index=None,
    ):
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

        archetype = (
            self._tier3_archetype(
                card
            )
        )

        archetype_fit = (
            self._tier3_archetype_fit(
                player=player,
                card=card,
                archetype=archetype,
                tier1_supply=tier1_supply,
            )
        )

        viable = (
            coverage_ratio
            >= self.tier3_min_coverage_ratio
            or residual_after_tier1
            <= self.tier3_max_residual_after_tier1
        )

        route_score = (
            card.points
            * 2.25
            + coverage_ratio
            * 10.0
            + tier1_coverage
            * 2.5
            + archetype_fit
            - residual_after_tier1
            * 1.25
        )

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

            "archetype":
                archetype,

            "archetype_fit":
                float(
                    archetype_fit
                ),

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

    def _tier3_archetype(
        self,
        card,
    ):
        costs = sorted(
            [
                int(
                    card.cost.get(
                        color,
                        0,
                    )
                )
                for color in COLOR_ORDER
                if card.cost.get(
                    color,
                    0,
                ) > 0
            ],
            reverse=True,
        )

        if (
            card.points == 3
            and costs
            == [
                5,
                3,
                3,
                3,
            ]
        ):
            return self.T3_BROAD

        if (
            card.points == 4
            and costs
            == [
                7
            ]
        ):
            return self.T3_CONCENTRATED

        if (
            card.points == 4
            and costs
            == [
                6,
                3,
                3,
            ]
        ):
            return self.T3_THREE_COLOR

        if (
            card.points == 5
            and costs
            == [
                7,
                3,
            ]
        ):
            return self.T3_TWO_COLOR

        return self.T3_OTHER

    def _tier3_archetype_fit(
        self,
        player,
        card,
        archetype,
        tier1_supply,
    ):
        ratios = []

        for color in COLOR_ORDER:
            cost = card.cost.get(
                color,
                0,
            )

            if cost <= 0:
                continue

            coverage = (
                player.bonuses[
                    color
                ]
                + player.gems[
                    color
                ]
                + tier1_supply[
                    color
                ]
            )

            ratios.append(
                min(
                    1.0,
                    coverage
                    / cost,
                )
            )

        if not ratios:
            return 0.0

        average = sum(
            ratios
        ) / len(
            ratios
        )

        minimum = min(
            ratios
        )

        maximum = max(
            ratios
        )

        if (
            archetype
            == self.T3_CONCENTRATED
        ):
            return (
                maximum
                * 3.0
            )

        if (
            archetype
            == self.T3_TWO_COLOR
        ):
            return (
                average
                * 2.75
                + minimum
                * 0.75
            )

        if (
            archetype
            == self.T3_THREE_COLOR
        ):
            return (
                average
                * 2.25
                + minimum
                * 0.75
            )

        if (
            archetype
            == self.T3_BROAD
        ):
            return (
                average
                * 1.75
                + minimum
                * 1.25
            )

        return (
            average
            * 2.0
        )

    def _tier3_target_overlap(
        self,
        target_a,
        target_b,
    ):
        a = target_a[
            "post_gold_profile"
        ]

        b = target_b[
            "post_gold_profile"
        ]

        shared_demand = sum(
            min(
                a[
                    color
                ],
                b[
                    color
                ],
            )
            for color in COLOR_ORDER
        )

        denominator = max(
            1.0,
            min(
                sum(
                    a.values()
                ),
                sum(
                    b.values()
                ),
            ),
        )

        shared_colors = {
            color
            for color in COLOR_ORDER
            if (
                a[
                    color
                ] > 0
                and b[
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
                    shared_demand
                    / denominator
                ),

            "shared_colors":
                shared_colors,
        }

    # ============================================================
    # TIER-1 SUPPLY
    # ============================================================

    def _tier1_bridge_supply(
        self,
        state,
        player,
    ):
        supply = {
            color: 0.0
            for color in COLOR_ORDER
        }

        cards = list(
            state.visible_cards[
                1
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
                == 1
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

            supply[
                card.bonus_color
            ] += (
                self._affordability_weight(
                    distance
                )
            )

        return supply

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
    # ACTION SCORING
    # ============================================================

    def _score_legal_action(
        self,
        state,
        action,
        board_model,
    ):
        action_type = (
            action.action_type
        )

        group = (
            self._action_group(
                action
            )
        )

        mandatory = False
        tags = set()

        if action_type in (
            ActionType.BUY_VISIBLE,
            ActionType.BUY_RESERVED,
        ):
            score = (
                self._score_buy_action(
                    state,
                    action,
                    board_model,
                )
            )

            if (
                self._is_threshold_crossing_buy(
                    state,
                    action,
                )
            ):
                mandatory = True
                score += 100.0
                tags.add(
                    "immediate_win"
                )

        elif (
            action_type
            == ActionType.TAKE_GEMS
        ):
            score = (
                self._score_take_action(
                    state,
                    action,
                    board_model,
                )
            )

        elif action_type in (
            ActionType.RESERVE_VISIBLE,
            ActionType.RESERVE_TOP_DECK,
        ):
            (
                score,
                reserve_mandatory,
                reserve_tags,
            ) = (
                self._score_reserve_action(
                    state,
                    action,
                    board_model,
                )
            )

            mandatory = (
                mandatory
                or reserve_mandatory
            )

            tags.update(
                reserve_tags
            )

        else:
            score = 0.0

        if not np.isfinite(
            score
        ):
            return None

        return {
            "action":
                action,

            "group":
                group,

            "score":
                float(
                    score
                ),

            "mandatory":
                bool(
                    mandatory
                ),

            "tags":
                tags,
        }

    def _score_buy_action(
        self,
        state,
        action,
        board_model,
    ):
        player = board_model[
            "player"
        ]

        card = (
            self._get_card_from_action(
                state,
                action,
            )
        )

        if card is None:
            return float(
                "-inf"
            )

        tier = (
            self._action_tier(
                state,
                action,
            )
        )

        demand = board_model[
            "color_demand"
        ].get(
            card.bonus_color,
            0.0,
        )

        score = (
            card.points
            * 4.0
            + demand
            * self.permanent_bonus_weight
        )

        score += (
            self._noble_completion_count_after_card(
                state,
                player,
                card,
            )
            * self.noble_completion_bonus
        )

        if tier == 1:
            # Pure infrastructure: route reuse matters more than VP.
            score += (
                demand
                * 1.25
            )

            score -= (
                self._effective_colored_cost(
                    player,
                    card,
                )
                * 0.30
            )

        elif tier == 2:
            # Bridge + points.
            score += (
                card.points
                * self.tier2_point_weight
                + demand
                * 0.80
            )

        elif tier == 3:
            score += (
                card.points
                * self.tier3_point_weight
            )

            target_role = (
                self._tier3_card_role(
                    card,
                    board_model[
                        "tier3"
                    ],
                )
            )

            if (
                target_role
                == "primary"
            ):
                score += 12.0

            elif (
                target_role
                == "secondary"
            ):
                score += 9.0

            else:
                # Other T3 cards can still be excellent immediate
                # point conversions.
                score += 3.0

        if (
            action.action_type
            == ActionType.BUY_RESERVED
        ):
            score += 1.5

        return score

    def _score_take_action(
        self,
        state,
        action,
        board_model,
    ):
        player = board_model[
            "player"
        ]

        gems_after = dict(
            player.gems
        )

        for color in action.gem_colors:
            gems_after[
                color
            ] += 1

        color_demand = board_model[
            "color_demand"
        ]

        # Small direct preference for taking colors central to the
        # current strategic graph.
        score = sum(
            color_demand.get(
                color,
                0.0,
            )
            * 0.35
            for color in action.gem_colors
            if color in COLOR_ORDER
        )

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
                        gems_after,
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
                    gems_after,
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

        score += (
            self._take_bank_denial_bonus(
                state,
                action,
            )
        )

        return score

    def _score_reserve_action(
        self,
        state,
        action,
        board_model,
    ):
        tags = set()

        if (
            action.action_type
            == ActionType.RESERVE_TOP_DECK
        ):
            # Blind reserve still gains gold, but has no known route.
            return (
                -3.0
                + self._reserve_gold_value(
                    state
                ),
                False,
                tags,
            )

        card = (
            self._get_card_from_action(
                state,
                action,
            )
        )

        if card is None:
            return (
                float(
                    "-inf"
                ),
                False,
                tags,
            )

        player = board_model[
            "player"
        ]

        tier = (
            self._action_tier(
                state,
                action,
            )
        )

        demand = board_model[
            "color_demand"
        ].get(
            card.bonus_color,
            0.0,
        )

        distance = (
            self._distance_to_card(
                player,
                card,
            )
        )

        score = (
            self._reserve_gold_value(
                state
            )
        )

        mandatory = False

        if tier == 3:
            role = (
                self._tier3_card_role(
                    card,
                    board_model[
                        "tier3"
                    ],
                )
            )

            score += (
                card.points
                * 3.0
                - distance
                * 0.75
            )

            if role == "primary":
                score += 18.0
                mandatory = True
                tags.add(
                    "tier3_primary_reserve"
                )

            elif role == "secondary":
                score += 13.0
                mandatory = True
                tags.add(
                    "tier3_secondary_reserve"
                )

            else:
                score += 1.0

        elif tier == 2:
            # Reserve a T2 only if it is a useful bridge/point card.
            score += (
                card.points
                * 2.0
                + demand
                * 1.25
                - distance
                * 0.75
            )

            tags.add(
                "tier2_bridge_reserve"
            )

        elif tier == 1:
            # Rarely worth spending a reserve slot on pure engine.
            score += (
                demand
                * 0.75
                - distance
                - 4.0
            )

        score += (
            self._opponent_denial_value(
                state,
                card,
            )
        )

        return (
            score,
            mandatory,
            tags,
        )

    # ============================================================
    # ACTION-SCORING HELPERS
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

    def _tier3_card_role(
        self,
        card,
        tier3_model,
    ):
        for role in (
            "primary",
            "secondary",
        ):
            target = tier3_model.get(
                role,
                None,
            )

            if (
                target is not None
                and self._same_card(
                    card,
                    target[
                        "card"
                    ],
                )
            ):
                return role

        return None

    def _effective_colored_cost(
        self,
        player,
        card,
    ):
        return sum(
            max(
                0,
                card.cost.get(
                    color,
                    0,
                )
                - player.bonuses[
                    color
                ],
            )
            for color in COLOR_ORDER
        )

    def _noble_completion_count_after_card(
        self,
        state,
        player,
        card,
    ):
        completed = 0

        for noble in state.nobles:
            if noble is None:
                continue

            qualifies = True

            for color in COLOR_ORDER:
                bonuses = (
                    player.bonuses[
                        color
                    ]
                )

                if (
                    color
                    == card.bonus_color
                ):
                    bonuses += 1

                if (
                    bonuses
                    < noble.requirement[
                        color
                    ]
                ):
                    qualifies = False
                    break

            if qualifies:
                completed += 1

        return completed

    def _reserve_gold_value(
        self,
        state,
    ):
        try:
            gold = state.bank[
                GemColor.GOLD
            ]
        except Exception:
            gold = 0

        return (
            self.reserve_gold_bonus
            if gold > 0
            else 0.0
        )

    def _take_bank_denial_bonus(
        self,
        state,
        action,
    ):
        """
        In 2p only 4 normal gems of each color exist.

        Taking one from a bank of exactly 4 turns off the opponent's
        legal take-two option in that color.

        This is deliberately a SMALL term; terminal rollout remains
        the judge.
        """

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

            if bank_count == 4:
                bonus += (
                    self.bank_denial_weight
                )

        return bonus

    def _opponent_denial_value(
        self,
        state,
        card,
    ):
        if (
            len(
                state.players
            )
            != 2
        ):
            return 0.0

        opponent_index = (
            1
            - state.current_player
        )

        opponent = state.players[
            opponent_index
        ]

        distance = (
            self._distance_to_card(
                opponent,
                card,
            )
        )

        if distance <= 0:
            return (
                2.0
                + card.points
                * 0.75
            )

        if distance == 1:
            return (
                1.0
                + card.points
                * 0.40
            )

        return 0.0

    # ============================================================
    # TIER-3 RESOURCE HELPERS
    # ============================================================

    def _uncovered_color_profile(
        self,
        player,
        card,
    ):
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
    # WIN CHECK
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

        projected = (
            getattr(
                player,
                "points",
                0,
            )
            + card.points
        )

        projected += (
            self._best_noble_points_after_card(
                state,
                player,
                card,
            )
        )

        return (
            projected
            >= self.WIN_POINTS
        )

    def _best_noble_points_after_card(
        self,
        state,
        player,
        card,
    ):
        best = 0

        for noble in state.nobles:
            if noble is None:
                continue

            qualifies = True

            for color in COLOR_ORDER:
                bonuses = (
                    player.bonuses[
                        color
                    ]
                )

                if (
                    color
                    == card.bonus_color
                ):
                    bonuses += 1

                if (
                    bonuses
                    < noble.requirement[
                        color
                    ]
                ):
                    qualifies = False
                    break

            if qualifies:
                best = max(
                    best,
                    getattr(
                        noble,
                        "points",
                        3,
                    ),
                )

        return best

    # ============================================================
    # CARD / DISTANCE HELPERS
    # ============================================================

    def _get_card_from_action(
        self,
        state,
        action,
    ):
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

    def _action_tier(
        self,
        state,
        action,
    ):
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

        card = (
            self._get_card_from_action(
                state,
                action,
            )
        )

        return self._card_tier(
            card
        )

    def _card_tier(
        self,
        card,
    ):
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

    def _distance_to_card(
        self,
        player,
        card,
        gems=None,
    ):
        """
        Exact colored shortfall after permanent bonuses and current
        gems, with gold reducing the aggregate shortfall.

        This supports hypothetical TAKE evaluations.
        """

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
                - player.bonuses[
                    color
                ],
            )

            missing += max(
                0,
                required
                - gems[
                    color
                ],
            )

        try:
            gold = gems[
                GemColor.GOLD
            ]
        except Exception:
            gold = 0

        return max(
            0,
            missing
            - gold,
        )

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

        if len(
            candidates
        ) == 1:
            action, cheap_score = (
                candidates[
                    0
                ]
            )

            self.last_rollout_debug = {
                "candidate_count": 1,
                "num_rollouts_per_candidate": 0,
                "total_terminal_rollouts": 0,
                "evaluated": [
                    (
                        action,
                        0.0,
                        cheap_score,
                    )
                ],
            }

            return [
                (
                    action,
                    0.0,
                    cheap_score,
                )
            ]

        root_player = (
            state.current_player
        )

        worlds = (
            self._sample_hidden_worlds(
                state
            )
        )

        evaluated = []

        for (
            action,
            cheap_score,
        ) in candidates:
            values = []

            for sampled_state in worlds:
                value = (
                    self._rollout_from_action(
                        env=env,
                        sampled_state=sampled_state,
                        first_action=action,
                        root_player=root_player,
                    )
                )

                values.append(
                    value
                )

            evaluated.append(
                (
                    action,
                    float(
                        np.mean(
                            values
                        )
                    ),
                    cheap_score,
                )
            )

        self.last_rollout_debug = {
            "candidate_count":
                len(
                    candidates
                ),

            "num_rollouts_per_candidate":
                self.num_rollouts,

            "total_terminal_rollouts":
                len(
                    candidates
                )
                * self.num_rollouts,

            "evaluated":
                evaluated,
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
        sim_env = (
            env.clone()
            if hasattr(
                env,
                "clone",
            )
            else env
        )

        rollout_state = (
            sampled_state.clone()
        )

        rollout_state = (
            self._step(
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
                self._step(
                    sim_env,
                    rollout_state,
                    action,
                )
            )

            steps += 1

        if (
            sim_env._check_terminated(
                rollout_state
            )
        ):
            return (
                self._terminal_value(
                    rollout_state,
                    root_player,
                )
            )

        return 0.0

    def _step(
        self,
        env,
        state,
        action,
    ):
        result = env.step(
            action,
            state,
        )

        if (
            result is not None
            and hasattr(
                result,
                "players",
            )
            and hasattr(
                result,
                "current_player",
            )
        ):
            return result

        return state

    # ============================================================
    # HIDDEN WORLDS
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
                    decks[
                        tier
                    ]
                )

        else:
            for deck in decks:
                self.rng.shuffle(
                    deck
                )

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

        if winners is None:
            winner = getattr(
                state,
                "winner",
                None,
            )

            if winner is None:
                return 0.0

            winners = [
                winner
            ]

        if not winners:
            return 0.0

        if len(
            winners
        ) > 1:
            return 0.0

        return (
            1.0
            if root_player
            in winners
            else -1.0
        )

    # ============================================================
    # ACTION IDENTITY / DEBUG
    # ============================================================

    def _action_identity(
        self,
        action,
    ):
        try:
            hash(
                action
            )

            return action

        except TypeError:
            return repr(
                action
            )

    def get_board_model_debug(
        self,
    ):
        return self.last_board_model

    def get_candidate_debug(
        self,
    ):
        return self.last_candidate_debug

    def get_rollout_debug(
        self,
    ):
        return self.last_rollout_debug


class HeuristicAgent15Diagnostics(
    HeuristicAgent15
):
    """
    Diagnostic H15 with identical gameplay logic.

    Reports:
        - primary / secondary T3 availability
        - Tier-3 archetypes selected
        - selected action category
        - selected strategic tags
        - mandatory T3 reserves proposed / selected
        - average candidate count
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
        self._diag15 = {
            "strategic_decisions":
                0,

            "forced_node_calls":
                0,

            "primary_t3_count":
                0,

            "secondary_t3_count":
                0,

            "primary_archetypes":
                Counter(),

            "secondary_archetypes":
                Counter(),

            "selected_groups":
                Counter(),

            "selected_tags":
                Counter(),

            "mandatory_t3_reserve_candidates":
                0,

            "mandatory_t3_reserve_selected":
                0,

            "candidate_count_sum":
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

        self._record_diagnostic_decision(
            state,
            action,
        )

        return action

    def _record_diagnostic_decision(
        self,
        state,
        action,
    ):
        if (
            state.node_type
            != NodeType.MAIN_DECISION
        ):
            self._diag15[
                "forced_node_calls"
            ] += 1

            return

        self._diag15[
            "strategic_decisions"
        ] += 1

        board = (
            self.last_board_model
            or {}
        )

        tier3 = board.get(
            "tier3",
            {},
        )

        primary = tier3.get(
            "primary",
            None,
        )

        secondary = tier3.get(
            "secondary",
            None,
        )

        if primary is not None:
            self._diag15[
                "primary_t3_count"
            ] += 1

            self._diag15[
                "primary_archetypes"
            ][
                primary[
                    "archetype"
                ]
            ] += 1

        if secondary is not None:
            self._diag15[
                "secondary_t3_count"
            ] += 1

            self._diag15[
                "secondary_archetypes"
            ][
                secondary[
                    "archetype"
                ]
            ] += 1

        candidate_debug = (
            self.last_candidate_debug
            or {}
        )

        selected_records = (
            candidate_debug.get(
                "selected_records",
                [],
            )
        )

        self._diag15[
            "candidate_count_sum"
        ] += len(
            selected_records
        )

        selected_id = (
            self._action_identity(
                action
            )
        )

        selected_record = None

        for record in selected_records:
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

        if selected_record is not None:
            self._diag15[
                "selected_groups"
            ][
                selected_record[
                    "group"
                ]
            ] += 1

            for tag in selected_record.get(
                "tags",
                set(),
            ):
                self._diag15[
                    "selected_tags"
                ][tag] += 1

            if (
                "tier3_primary_reserve"
                in selected_record.get(
                    "tags",
                    set(),
                )
                or "tier3_secondary_reserve"
                in selected_record.get(
                    "tags",
                    set(),
                )
            ):
                self._diag15[
                    "mandatory_t3_reserve_selected"
                ] += 1

        mandatory_t3 = sum(
            1
            for record in selected_records
            if (
                "tier3_primary_reserve"
                in record.get(
                    "tags",
                    set(),
                )
                or "tier3_secondary_reserve"
                in record.get(
                    "tags",
                    set(),
                )
            )
        )

        self._diag15[
            "mandatory_t3_reserve_candidates"
        ] += mandatory_t3

    def get_diagnostic_stats(
        self,
    ):
        decisions = self._diag15[
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
                self._diag15[
                    "forced_node_calls"
                ],

            "primary_t3_count":
                self._diag15[
                    "primary_t3_count"
                ],

            "primary_t3_pct":
                pct(
                    self._diag15[
                        "primary_t3_count"
                    ]
                ),

            "secondary_t3_count":
                self._diag15[
                    "secondary_t3_count"
                ],

            "secondary_t3_pct":
                pct(
                    self._diag15[
                        "secondary_t3_count"
                    ]
                ),

            "primary_archetypes":
                dict(
                    self._diag15[
                        "primary_archetypes"
                    ]
                ),

            "secondary_archetypes":
                dict(
                    self._diag15[
                        "secondary_archetypes"
                    ]
                ),

            "selected_groups":
                dict(
                    self._diag15[
                        "selected_groups"
                    ]
                ),

            "selected_group_percentages": {
                key:
                    pct(
                        value
                    )
                for key, value
                in self._diag15[
                    "selected_groups"
                ].items()
            },

            "selected_tags":
                dict(
                    self._diag15[
                        "selected_tags"
                    ]
                ),

            "mandatory_t3_reserve_candidates":
                self._diag15[
                    "mandatory_t3_reserve_candidates"
                ],

            "mandatory_t3_reserve_selected":
                self._diag15[
                    "mandatory_t3_reserve_selected"
                ],

            "average_candidate_count":
                (
                    self._diag15[
                        "candidate_count_sum"
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
            "=== HeuristicAgent15 Unified Route Planner Diagnostics ===",
            f"Strategic decisions: {stats['strategic_decisions']}",
            f"Forced-node calls: {stats['forced_node_calls']}",
            "",
            (
                "Primary T3 available: "
                f"{stats['primary_t3_count']} "
                f"({stats['primary_t3_pct']:.2f}%)"
            ),
            (
                "Compatible secondary T3 available: "
                f"{stats['secondary_t3_count']} "
                f"({stats['secondary_t3_pct']:.2f}%)"
            ),
            "",
            "Primary T3 archetypes:",
        ]

        if stats[
            "primary_archetypes"
        ]:
            for key, value in sorted(
                stats[
                    "primary_archetypes"
                ].items()
            ):
                lines.append(
                    f"  {key}: {value}"
                )
        else:
            lines.append(
                "  none"
            )

        lines.extend(
            [
                "",
                "Secondary T3 archetypes:",
            ]
        )

        if stats[
            "secondary_archetypes"
        ]:
            for key, value in sorted(
                stats[
                    "secondary_archetypes"
                ].items()
            ):
                lines.append(
                    f"  {key}: {value}"
                )
        else:
            lines.append(
                "  none"
            )

        lines.extend(
            [
                "",
                "Final selected action groups:",
            ]
        )

        for key in (
            "buy",
            "take",
            "reserve",
            "other",
        ):
            count = stats[
                "selected_groups"
            ].get(
                key,
                0,
            )

            pct_value = stats[
                "selected_group_percentages"
            ].get(
                key,
                0.0,
            )

            lines.append(
                f"  {key}: "
                f"{count} "
                f"({pct_value:.2f}%)"
            )

        lines.extend(
            [
                "",
                "Selected strategic tags:",
            ]
        )

        if stats[
            "selected_tags"
        ]:
            for key, value in sorted(
                stats[
                    "selected_tags"
                ].items()
            ):
                lines.append(
                    f"  {key}: {value}"
                )
        else:
            lines.append(
                "  none"
            )

        lines.extend(
            [
                "",
                "Mandatory Tier-3 reserve investigation:",
                (
                    "  Candidates inserted: "
                    f"{stats['mandatory_t3_reserve_candidates']}"
                ),
                (
                    "  Reserve actions selected: "
                    f"{stats['mandatory_t3_reserve_selected']}"
                ),
                "",
                (
                    "Average root candidate count: "
                    f"{stats['average_candidate_count']:.3f}"
                ),
            ]
        )

        return "\n".join(
            lines
        )
