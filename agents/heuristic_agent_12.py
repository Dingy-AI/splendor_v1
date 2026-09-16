import random
import numpy as np

from splendor_v1.agents.heuristic_agent_3 import HeuristicAgent3
from splendor_v1.env.core.constants import COLOR_ORDER
from splendor_v1.env.core.enums import NodeType
from splendor_v1.env.core.actions import ActionType


class HeuristicAgent12:
    """
    HeuristicAgent12
    =================

    Protected-H4 hybrid strategy agent.

    Core principle
    --------------
    H4 is empirically stronger than H9, so H12 does NOT replace
    H4's root candidate logic.

    Instead:

        H4 candidates are ALWAYS protected
                +
        inspect current board state
                ↓
        choose one strategic route:
            1. NOBLE / KANGAROO
            2. RESERVE / HIGH-POINT
            3. NEUTRAL
                ↓
        add only a few strategy-specific extra root actions
                ↓
        full H3-vs-H3 terminal rollouts for EVERY candidate
                ↓
        choose best terminal result

    This directly targets H4's main blind spot:

        H3 uses a hard hierarchy:
            BUY > TAKE > RESERVE

    Therefore, if any BUY exists, H4 never investigates whether
    a TAKE or RESERVE would actually have the better terminal
    outcome.

    H12 keeps every action H4 would have searched and only adds
    a small number of actions that a board-aware strategy says
    may be worth investigating.

    ------------------------------------------------------------
    Strategy 1: Noble / Kangaroo route
    ------------------------------------------------------------

    Favor this route when:
        - two nobles share useful colors
        - especially >= 2 overlapping required colors
        - our current permanent bonuses are progressing toward them
        - visible Tier 1 cards provide those shared colors
        - the engine is still early enough to exploit those bonuses

    Strategic extras can include:
        - TAKE actions that progress toward noble-aligned cards
        - RESERVE_VISIBLE for unusually useful noble-aligned cards
        - any noble-aligned BUY that was somehow not already present

    ------------------------------------------------------------
    Strategy 2: Reserve / high-point route
    ------------------------------------------------------------

    Favor this route when:
        - strong Tier 2 / Tier 3 point cards are visible
        - those cards are reasonably close to affordable
        - our engine is already developed
        - our score is moving into the mid/late game
        - noble structure is weak / fragmented

    Strategic extras can include:
        - RESERVE_VISIBLE on valuable Tier 2 / Tier 3 cards
        - TAKE actions that progress toward high-point targets
        - point-heavy BUY actions not already searched

    ------------------------------------------------------------
    Neutral
    ------------------------------------------------------------

    If neither strategic route is compelling enough, H12 becomes
    H4 for that decision.

    ------------------------------------------------------------
    Rollouts
    ------------------------------------------------------------

    Continuation policy is explicitly HeuristicAgent3 for BOTH
    players all the way to terminal.

    Hidden deck order is shuffled independently for each sampled
    world, and the SAME sampled worlds are reused for every root
    candidate (common random numbers).

    Default:
        num_rollouts = 8
        num_strategy_moves = 2

    Thus if H4 normally searches 4 root actions, H12 searches at
    most ~6, rather than replacing them with an arbitrary global
    shortlist.
    """

    NOBLE = "noble"
    RESERVE = "reserve"
    NEUTRAL = "neutral"

    WIN_POINTS = 15

    def __init__(
        self,
        num_rollouts=8,
        num_strategy_moves=2,
        strategy_margin=2.0,
        min_strategy_score=8.0,
        max_rollout_steps=200,
        random_seed=None,
    ):
        self.num_rollouts = max(
            1,
            int(num_rollouts),
        )


        self.num_strategy_moves = max(
            0,
            int(num_strategy_moves),
        )

        self.strategy_margin = float(
            strategy_margin
        )

        self.min_strategy_score = float(
            min_strategy_score
        )

        self.max_rollout_steps = max(
            1,
            int(max_rollout_steps),
        )

        self.rng = random.Random(
            random_seed
        )

        # H4's root candidate restriction and rollout continuation
        # are both based on H3.
        self.h3 = HeuristicAgent3()

        self.last_strategy_debug = None
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
                item[1],   # terminal rollout result
                item[2],   # cheap score tie-break
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
        logits -= np.max(logits)

        probs = np.exp(logits)
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

            policy[action_id] += prob

        return policy

    # ============================================================
    # ROOT CANDIDATES
    # ============================================================

    def _get_root_candidates(
        self,
        env,
        state,
    ):
        """
        Start from EXACTLY the H3 candidate set that gives H4 its
        hard BUY > TAKE > RESERVE root hierarchy.

        Never remove these candidates.

        Then append up to num_strategy_moves board-aware extras.
        """

        base_candidates = (
            self.h3._get_scored_candidates(
                env,
                state,
            )
        )

        if not base_candidates:
            return []

        # Forced nodes are not strategic choices.
        if (
            state.node_type
            != NodeType.MAIN_DECISION
        ):
            self.last_strategy_debug = {
                "strategy": self.NEUTRAL,
                "reason": "forced_node",
            }

            self.last_candidate_debug = {
                "base_count":
                    len(base_candidates),
                "extra_count":
                    0,
                "total_count":
                    len(base_candidates),
            }

            return base_candidates

        strategy_info = (
            self._choose_strategy(
                state
            )
        )

        strategy = strategy_info[
            "strategy"
        ]

        extras = []

        if (
            self.num_strategy_moves > 0
            and strategy != self.NEUTRAL
        ):
            extras = (
                self._get_strategy_extras(
                    env=env,
                    state=state,
                    strategy=strategy,
                    excluded_actions=[
                        action
                        for action, _
                        in base_candidates
                    ],
                )
            )

        # Protect every H4 candidate.
        combined = list(
            base_candidates
        )

        existing = {
            self._action_identity(
                action
            )
            for action, _
            in combined
        }

        for action, score in extras:
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
                    score,
                )
            )

            existing.add(
                identity
            )

        self.last_strategy_debug = (
            strategy_info
        )

        self.last_candidate_debug = {
            "strategy":
                strategy,

            "base_count":
                len(
                    base_candidates
                ),

            "extra_count":
                len(
                    combined
                )
                - len(
                    base_candidates
                ),

            "total_count":
                len(
                    combined
                ),

            "base_candidates":
                base_candidates,

            "strategy_extras":
                extras,
        }

        return combined

    # ============================================================
    # STRATEGY CLASSIFIER
    # ============================================================

    def _choose_strategy(
        self,
        state,
    ):
        noble_info = (
            self._score_noble_strategy(
                state
            )
        )

        reserve_info = (
            self._score_reserve_strategy(
                state,
                noble_info=noble_info,
            )
        )

        noble_score = noble_info[
            "score"
        ]

        reserve_score = reserve_info[
            "score"
        ]

        best_score = max(
            noble_score,
            reserve_score,
        )

        # Conservative fallback:
        # if neither board pattern is strong, behave exactly like H4.
        if (
            best_score
            < self.min_strategy_score
        ):
            strategy = self.NEUTRAL
            reason = (
                "neither_strategy_strong_enough"
            )

        elif (
            noble_score
            >= reserve_score
            + self.strategy_margin
        ):
            strategy = self.NOBLE
            reason = (
                "noble_score_clear_winner"
            )

        elif (
            reserve_score
            >= noble_score
            + self.strategy_margin
        ):
            strategy = self.RESERVE
            reason = (
                "reserve_score_clear_winner"
            )

        else:
            # If scores are nearly tied, do not inject our own
            # uncertain strategic bias into H4.
            strategy = self.NEUTRAL
            reason = (
                "strategy_scores_too_close"
            )

        return {
            "strategy":
                strategy,

            "reason":
                reason,

            "noble_score":
                noble_score,

            "reserve_score":
                reserve_score,

            "noble":
                noble_info,

            "reserve":
                reserve_info,
        }

    # ============================================================
    # NOBLE / KANGAROO STRATEGY SCORE
    # ============================================================

    def _score_noble_strategy(
        self,
        state,
    ):
        player = state.players[
            state.current_player
        ]

        pair = (
            self._best_noble_pair(
                state
            )
        )

        if not pair[
            "target_nobles"
        ]:
            return {
                "score": 0.0,
                **pair,
                "aligned_tier1_count": 0,
                "shared_tier1_count": 0,
            }

        overlap_count = pair[
            "overlap_count"
        ]

        total_missing = pair[
            "total_missing"
        ]

        # Strong nonlinear reward at the user's requested
        # "two overlapping colors" boundary.
        if overlap_count >= 3:
            overlap_score = 16.0

        elif overlap_count == 2:
            overlap_score = 12.0

        elif overlap_count == 1:
            overlap_score = 4.0

        else:
            overlap_score = 0.0

        # A pair is more attractive if the current permanent-bonus
        # engine is already moving toward it.
        closeness_score = max(
            0.0,
            12.0
            - 0.55
            * total_missing,
        )

        overlap_colors = pair[
            "overlap_colors"
        ]

        target_nobles = pair[
            "target_nobles"
        ]

        aligned_tier1_count = 0
        shared_tier1_count = 0

        for card in state.visible_cards[
            1
        ]:
            if card is None:
                continue

            color = card.bonus_color

            relevant = any(
                noble.requirement[
                    color
                ]
                > player.bonuses[
                    color
                ]
                for noble in target_nobles
            )

            if relevant:
                aligned_tier1_count += 1

            if (
                overlap_count >= 2
                and color in overlap_colors
                and any(
                    noble.requirement[
                        color
                    ]
                    > player.bonuses[
                        color
                    ]
                    for noble in target_nobles
                )
            ):
                shared_tier1_count += 1

        board_support_score = (
            aligned_tier1_count
            * 1.0
            + shared_tier1_count
            * 2.0
        )

        # Noble-engine route is naturally more attractive before
        # the game is nearly over.
        points = getattr(
            player,
            "points",
            0,
        )

        if points >= 10:
            phase_adjustment = -5.0

        elif points >= 7:
            phase_adjustment = -2.0

        else:
            phase_adjustment = 2.0

        score = (
            overlap_score
            + closeness_score
            + board_support_score
            + phase_adjustment
        )

        return {
            "score":
                float(
                    score
                ),

            **pair,

            "aligned_tier1_count":
                aligned_tier1_count,

            "shared_tier1_count":
                shared_tier1_count,

            "overlap_score":
                overlap_score,

            "closeness_score":
                closeness_score,

            "board_support_score":
                board_support_score,

            "phase_adjustment":
                phase_adjustment,
        }

    # ============================================================
    # RESERVE / POINT STRATEGY SCORE
    # ============================================================

    def _score_reserve_strategy(
        self,
        state,
        noble_info,
    ):
        player = state.players[
            state.current_player
        ]

        targets = (
            self._get_high_point_targets(
                state
            )
        )

        if targets:
            best_target_score = (
                targets[0][
                    "route_score"
                ]
            )

            second_target_score = (
                targets[1][
                    "route_score"
                ]
                if len(
                    targets
                ) >= 2
                else 0.0
            )

        else:
            best_target_score = 0.0
            second_target_score = 0.0

        # Do not let raw target scores dominate completely.
        target_component = (
            max(
                0.0,
                best_target_score,
            )
            + 0.4
            * max(
                0.0,
                second_target_score,
            )
        )

        total_bonuses = sum(
            player.bonuses[
                color
            ]
            for color in COLOR_ORDER
        )

        if total_bonuses >= 8:
            engine_score = 6.0

        elif total_bonuses >= 5:
            engine_score = 3.0

        else:
            engine_score = 0.0

        points = getattr(
            player,
            "points",
            0,
        )

        if points >= 10:
            point_pressure = 7.0

        elif points >= 7:
            point_pressure = 4.0

        elif points >= 4:
            point_pressure = 2.0

        else:
            point_pressure = 0.0

        # If nobles have little overlap, direct high-point conversion
        # becomes more attractive.
        overlap_count = noble_info.get(
            "overlap_count",
            0,
        )

        if overlap_count == 0:
            weak_noble_bonus = 4.0

        elif overlap_count == 1:
            weak_noble_bonus = 2.0

        else:
            weak_noble_bonus = 0.0

        # Scale target component down to classifier scale.
        target_component *= 0.65

        score = (
            target_component
            + engine_score
            + point_pressure
            + weak_noble_bonus
        )

        return {
            "score":
                float(
                    score
                ),

            "targets":
                targets,

            "target_component":
                target_component,

            "engine_score":
                engine_score,

            "point_pressure":
                point_pressure,

            "weak_noble_bonus":
                weak_noble_bonus,

            "total_bonuses":
                total_bonuses,
        }

    # ============================================================
    # STRATEGY EXTRA ACTIONS
    # ============================================================

    def _get_strategy_extras(
        self,
        env,
        state,
        strategy,
        excluded_actions,
    ):
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

            if strategy == self.NOBLE:
                score = (
                    self._score_noble_extra_action(
                        state,
                        action,
                    )
                )

            else:
                score = (
                    self._score_reserve_extra_action(
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

    # ============================================================
    # NOBLE EXTRA ACTION SCORING
    # ============================================================

    def _score_noble_extra_action(
        self,
        state,
        action,
    ):
        player = state.players[
            state.current_player
        ]

        pair = self._best_noble_pair(
            state
        )

        targets = pair[
            "target_nobles"
        ]

        if not targets:
            return None

        overlap_colors = pair[
            "overlap_colors"
        ]

        overlap_count = pair[
            "overlap_count"
        ]

        action_type = (
            action.action_type
        )

        # --------------------------------------------------------
        # BUY
        # --------------------------------------------------------

        if action_type in (
            ActionType.BUY_VISIBLE,
            ActionType.BUY_RESERVED,
        ):
            card = self._get_card_from_action(
                state,
                action,
            )

            if card is None:
                return None

            score = self._noble_card_alignment(
                player=player,
                card=card,
                target_nobles=targets,
                overlap_colors=overlap_colors,
                overlap_count=overlap_count,
            )

            tier = self._action_tier(
                state,
                action,
            )

            if (
                tier == 1
                and score > 0
            ):
                score += 4.0

            return score

        # --------------------------------------------------------
        # TAKE
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

            best = 0.0

            for tier in (
                1,
                2,
                3,
            ):
                for card in state.visible_cards[
                    tier
                ]:
                    if card is None:
                        continue

                    alignment = (
                        self._noble_card_alignment(
                            player=player,
                            card=card,
                            target_nobles=targets,
                            overlap_colors=overlap_colors,
                            overlap_count=overlap_count,
                        )
                    )

                    if alignment <= 0:
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

                    if progress > 0:
                        best = max(
                            best,
                            alignment
                            + 4.0
                            * progress,
                        )

            return (
                best
                if best > 0
                else None
            )

        # --------------------------------------------------------
        # RESERVE
        # --------------------------------------------------------

        if (
            action_type
            == ActionType.RESERVE_VISIBLE
        ):
            card = self._get_card_from_action(
                state,
                action,
            )

            if card is None:
                return None

            alignment = (
                self._noble_card_alignment(
                    player=player,
                    card=card,
                    target_nobles=targets,
                    overlap_colors=overlap_colors,
                    overlap_count=overlap_count,
                )
            )

            # Kangaroo route does not reserve much, so only unusually
            # aligned cards should become H4 expansion candidates.
            return (
                alignment - 4.0
                if alignment >= 8.0
                else None
            )

        return None

    # ============================================================
    # RESERVE / POINT EXTRA ACTION SCORING
    # ============================================================

    def _score_reserve_extra_action(
        self,
        state,
        action,
    ):
        player = state.players[
            state.current_player
        ]

        action_type = (
            action.action_type
        )

        # --------------------------------------------------------
        # RESERVE VISIBLE
        # --------------------------------------------------------

        if (
            action_type
            == ActionType.RESERVE_VISIBLE
        ):
            card = self._get_card_from_action(
                state,
                action,
            )

            if card is None:
                return None

            tier = self._action_tier(
                state,
                action,
            )

            if tier not in (
                2,
                3,
            ):
                return None

            distance = (
                self._distance_to_card(
                    player,
                    card,
                )
            )

            score = (
                card.points
                * 4.0
                - distance
                * 1.5
            )

            if tier == 2:
                score += 3.0

            elif tier == 3:
                score += 7.0

            return score

        # --------------------------------------------------------
        # TAKE
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

            best = 0.0

            for target in self._get_high_point_targets(
                state
            ):
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

                value = (
                    target[
                        "route_score"
                    ]
                    + progress
                    * 4.0
                )

                best = max(
                    best,
                    value,
                )

            return (
                best
                if best > 0
                else None
            )

        # --------------------------------------------------------
        # BUY
        # --------------------------------------------------------

        if action_type in (
            ActionType.BUY_VISIBLE,
            ActionType.BUY_RESERVED,
        ):
            card = self._get_card_from_action(
                state,
                action,
            )

            if card is None:
                return None

            tier = self._action_tier(
                state,
                action,
            )

            if (
                tier not in (
                    2,
                    3,
                )
                and card.points <= 0
            ):
                return None

            score = (
                card.points
                * 4.0
            )

            if tier == 2:
                score += 3.0

            elif tier == 3:
                score += 7.0

            if (
                action_type
                == ActionType.BUY_RESERVED
            ):
                score += 4.0

            return score

        return None

    # ============================================================
    # NOBLE PAIR
    # ============================================================

    def _best_noble_pair(
        self,
        state,
    ):
        player = state.players[
            state.current_player
        ]

        nobles = [
            noble
            for noble in state.nobles
            if noble is not None
        ]

        if not nobles:
            return {
                "target_nobles": [],
                "overlap_colors": set(),
                "overlap_count": 0,
                "total_missing": 0,
                "individual_missing": [],
            }

        if len(nobles) == 1:
            missing = (
                self._noble_missing(
                    player,
                    nobles[0],
                )
            )

            return {
                "target_nobles": [
                    nobles[0]
                ],
                "overlap_colors": set(),
                "overlap_count": 0,
                "total_missing": missing,
                "individual_missing": [
                    missing
                ],
            }

        best = None

        for i in range(
            len(nobles)
        ):
            for j in range(
                i + 1,
                len(nobles),
            ):
                noble_a = nobles[i]
                noble_b = nobles[j]

                overlap_colors = {
                    color
                    for color in COLOR_ORDER
                    if (
                        noble_a.requirement[
                            color
                        ] > 0
                        and noble_b.requirement[
                            color
                        ] > 0
                    )
                }

                missing_a = (
                    self._noble_missing(
                        player,
                        noble_a,
                    )
                )

                missing_b = (
                    self._noble_missing(
                        player,
                        noble_b,
                    )
                )

                total_missing = (
                    missing_a
                    + missing_b
                )

                rank = (
                    len(
                        overlap_colors
                    ),
                    -total_missing,
                    -max(
                        missing_a,
                        missing_b,
                    ),
                )

                if (
                    best is None
                    or rank > best[
                        "rank"
                    ]
                ):
                    best = {
                        "rank":
                            rank,

                        "target_nobles":
                            [
                                noble_a,
                                noble_b,
                            ],

                        "overlap_colors":
                            overlap_colors,

                        "overlap_count":
                            len(
                                overlap_colors
                            ),

                        "total_missing":
                            total_missing,

                        "individual_missing":
                            [
                                missing_a,
                                missing_b,
                            ],
                    }

        return best

    def _noble_missing(
        self,
        player,
        noble,
    ):
        return sum(
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
        )

    def _noble_card_alignment(
        self,
        player,
        card,
        target_nobles,
        overlap_colors,
        overlap_count,
    ):
        color = card.bonus_color

        relevant_count = sum(
            1
            for noble in target_nobles
            if (
                noble.requirement[
                    color
                ]
                > player.bonuses[
                    color
                ]
            )
        )

        if relevant_count == 0:
            return 0.0

        score = (
            relevant_count
            * 3.0
        )

        if relevant_count >= 2:
            score += 6.0

        if (
            overlap_count >= 2
            and color in overlap_colors
        ):
            score += 6.0

        return score

    # ============================================================
    # HIGH-POINT TARGETS
    # ============================================================

    def _get_high_point_targets(
        self,
        state,
    ):
        player = state.players[
            state.current_player
        ]

        targets = []

        for tier in (
            2,
            3,
        ):
            for card in state.visible_cards[
                tier
            ]:
                if card is None:
                    continue

                distance = (
                    self._distance_to_card(
                        player,
                        card,
                    )
                )

                route_score = (
                    card.points
                    * 3.0
                    - distance
                    * 1.5
                )

                if tier == 2:
                    route_score += 1.5

                elif tier == 3:
                    route_score += 3.0

                targets.append(
                    {
                        "card":
                            card,

                        "tier":
                            tier,

                        "distance":
                            distance,

                        "route_score":
                            float(
                                route_score
                            ),
                    }
                )

        targets.sort(
            key=lambda item:
                item[
                    "route_score"
                ],
            reverse=True,
        )

        return targets[:6]

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

            try:
                return method(
                    player,
                    card,
                    gems,
                )
            except TypeError:
                pass

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

        # Avoid importing a gold enum name dependency here by using
        # H3 whenever available. If H3's distance helper does not
        # accept alternate gems, this fallback simply ignores gold
        # for the hypothetical comparison rather than leaking state.
        return missing

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
            action, cheap_score = (
                candidates[0]
            )

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
                len(candidates),

            "num_rollouts_per_candidate":
                self.num_rollouts,

            "total_terminal_rollouts":
                len(candidates)
                * self.num_rollouts,

            "evaluated":
                evaluated,
        }

        return evaluated

    # ============================================================
    # FULL H3 TERMINAL ROLLOUT
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
            return self._terminal_value(
                rollout_state,
                root_player,
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

        if len(winners) > 1:
            return 0.0

        return (
            1.0
            if root_player
            in winners
            else -1.0
        )

    # ============================================================
    # ACTION IDENTITY
    # ============================================================

    def _action_identity(
        self,
        action,
    ):
        try:
            hash(action)
            return action

        except TypeError:
            return repr(
                action
            )

    # ============================================================
    # DEBUG API
    # ============================================================

    def get_strategy_debug(
        self,
    ):
        return self.last_strategy_debug

    def get_candidate_debug(
        self,
    ):
        return self.last_candidate_debug

    def get_rollout_debug(
        self,
    ):
        return self.last_rollout_debug
