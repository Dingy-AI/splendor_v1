from splendor_v1.agents.heuristic_agent_9 import HeuristicAgent9
from splendor_v1.env.core.constants import COLOR_ORDER
from splendor_v1.env.core.enums import GemColor
from splendor_v1.env.core.actions import ActionType


class HeuristicAgent10(HeuristicAgent9):
    """
    HeuristicAgent10 = Kangaroo-style noble acquisition strategy.

    H10 keeps H9's search architecture:

        all legal actions
            ↓
        cheap candidate scoring
            ↓
        phase-aware shortlist
            ↓
        num_calc_moves cap / rollover
            ↓
        H3 full terminal rollouts
            ↓
        choose best terminal result

    H10 changes ONLY the cheap root-candidate priorities.

    Strategic identity
    ------------------
    The Kangaroo route is based on the observed elite-player pattern:

        - many Tier 1 purchases
        - large permanent-bonus engine
        - roughly two nobles per game
        - relatively little reserving
        - low dependence on temporary token stock
        - later conversion of the engine into Tier 2 / Tier 3 points

    H10 therefore tries to ensure that noble-building actions survive
    root pruning and receive rollout calculation.

    Noble pair logic
    ----------------
    H10 assumes the player may be building toward TWO nobles.

    For every pair of available nobles:

        1. Count how many required colors overlap.
        2. Compute how many permanent bonuses the player is missing
           across the pair.
        3. Prefer:
               more overlap first
               then fewer missing bonuses

    If the best pair has >= 2 overlapping colors:

        cards whose permanent bonus is one of those shared colors
        receive a strong candidate-selection bonus.

    Example:

        Noble A wants:
            3 blue, 3 red, 3 green

        Noble B wants:
            3 blue, 3 red, 3 black

        overlap colors:
            blue, red

        Tier 1 blue / red bonus cards become especially attractive
        because each permanent bonus progresses BOTH noble routes.

    Important:
        This noble logic does NOT directly choose the final move.
        It only changes which actions are likely to make H9's
        num_calc_moves shortlist.

        Final choice still comes from H3 full terminal rollout.

    Designed for the same environment/API as HeuristicAgent9.
    """

    # ============================================================
    # KANGAROO SEARCH ALLOCATION
    # ============================================================

    # Kangaroo buys a lot of cards and reserves relatively little.
    # These are soft targets; unused capacity still rolls over using
    # H9's normal waterfall behavior.
    KANGAROO_EARLY_TARGETS = {
        "buy": 5,
        "take": 3,
        "reserve": 0,
    }

    KANGAROO_MID_TARGETS = {
        "buy": 6,
        "take": 2,
        "reserve": 0,
    }

    KANGAROO_LATE_TARGETS = {
        "buy": 6,
        "take": 1,
        "reserve": 1,
    }

    # ============================================================
    # NOBLE STRATEGY WEIGHTS
    # ============================================================

    # Card bonus advances one currently relevant noble.
    NOBLE_COLOR_BONUS = 3.0

    # Card bonus advances BOTH selected nobles.
    DOUBLE_NOBLE_COLOR_BONUS = 7.0

    # Stronger emphasis when the chosen noble pair shares at least
    # two colors, matching the user's desired "color overlap of 2"
    # behavior.
    TWO_COLOR_OVERLAP_BONUS = 6.0

    # Extra candidate priority when buying a card would complete a
    # noble immediately.
    NOBLE_COMPLETION_BONUS = 14.0

    # Extra priority for Tier 1 cards that contribute to the noble
    # engine. This is search allocation, not final action value.
    TIER1_NOBLE_ENGINE_BONUS = 4.0

    # TAKE_GEMS receives bonus only when it makes concrete progress
    # toward noble-aligned cards.
    TAKE_NOBLE_PROGRESS_WEIGHT = 3.0

    # Cap noble contribution so candidate ranking remains sane.
    MAX_NOBLE_SELECTION_BONUS = 20.0

    # Kangaroo reserves much less than the other elite archetype.
    RESERVE_SELECTION_PENALTY = 3.0

    # ============================================================
    # PHASE ALLOCATION OVERRIDES
    # ============================================================

    def _phase_targets(
        self,
        phase,
    ):
        if phase == self.EARLY:
            return dict(
                self.KANGAROO_EARLY_TARGETS
            )

        if phase == self.MID:
            return dict(
                self.KANGAROO_MID_TARGETS
            )

        return dict(
            self.KANGAROO_LATE_TARGETS
        )

    def _phase_bonus(
        self,
        phase,
        group,
        tier,
        action,
    ):
        """
        Kangaroo-specific phase bias.

        Compared with H9:
            - stronger Tier 1 early emphasis
            - Tier 1 remains relevant longer
            - reserve is less favored
            - Tier 2 / 3 still take over as scoring sources later
        """

        if group == "buy":

            if phase == self.EARLY:
                if tier == 1:
                    return 11.0
                if tier == 2:
                    return 1.0
                if tier == 3:
                    return -7.0
                return 1.0

            if phase == self.MID:
                if tier == 1:
                    return 4.0
                if tier == 2:
                    return 8.0
                if tier == 3:
                    return 2.0
                return 3.0

            # LATE
            if tier == 1:
                return -2.0
            if tier == 2:
                return 6.0
            if tier == 3:
                return 10.0
            return 5.0

        if group == "take":

            if phase == self.EARLY:
                return 6.0

            if phase == self.MID:
                return 3.0

            return 1.0

        if group == "reserve":

            # Kangaroo route intentionally suppresses reservation.
            if phase == self.EARLY:
                return -8.0

            if phase == self.MID:
                return -4.0

            # Late reserve is still allowed tactically.
            if tier == 3:
                return 1.0

            return -3.0

        return 0.0

    # ============================================================
    # ROOT ACTION RECORD OVERRIDE
    # ============================================================

    def _make_action_record(
        self,
        state,
        action,
        phase,
    ):
        """
        Start with H9's normal action record, then add the Kangaroo
        noble-acquisition priority.

        H9 still owns:
            - legal action handling
            - early reserve filtering
            - H3 base score
            - threshold-crossing mandatory buys
            - shortlist tolerance
            - waterfall / rollover
            - terminal rollout
        """

        record = super()._make_action_record(
            state=state,
            action=action,
            phase=phase,
        )

        if record is None:
            return None

        noble_bonus = (
            self._kangaroo_noble_action_bonus(
                state=state,
                action=action,
                phase=phase,
            )
        )

        noble_bonus = max(
            -self.MAX_NOBLE_SELECTION_BONUS,
            min(
                self.MAX_NOBLE_SELECTION_BONUS,
                noble_bonus,
            ),
        )

        record["noble_bonus"] = float(
            noble_bonus
        )

        record["selection_score"] += (
            noble_bonus
        )

        return record

    # ============================================================
    # NOBLE ACTION BONUS
    # ============================================================

    def _kangaroo_noble_action_bonus(
        self,
        state,
        action,
        phase,
    ):
        strategy = (
            self._get_noble_pair_strategy(
                state
            )
        )

        if not strategy[
            "target_nobles"
        ]:
            return 0.0

        action_type = (
            action.action_type
        )

        if action_type in (
            ActionType.BUY_VISIBLE,
            ActionType.BUY_RESERVED,
        ):
            return self._buy_noble_bonus(
                state=state,
                action=action,
                phase=phase,
                strategy=strategy,
            )

        if (
            action_type
            == ActionType.TAKE_GEMS
        ):
            return self._take_noble_bonus(
                state=state,
                action=action,
                phase=phase,
                strategy=strategy,
            )

        if (
            action_type
            == ActionType.RESERVE_VISIBLE
        ):
            # A reserve can preserve access to a noble-aligned card,
            # but Kangaroo's empirical style reserves much less.
            card = self._get_card_from_action(
                state,
                action,
            )

            if card is None:
                return (
                    -self.RESERVE_SELECTION_PENALTY
                )

            alignment = (
                self._card_noble_alignment_bonus(
                    state=state,
                    card=card,
                    strategy=strategy,
                    include_completion=False,
                )
            )

            return (
                alignment * 0.25
                - self.RESERVE_SELECTION_PENALTY
            )

        if (
            action_type
            == ActionType.RESERVE_TOP_DECK
        ):
            return (
                -self.RESERVE_SELECTION_PENALTY
                - 2.0
            )

        return 0.0

    # ============================================================
    # BUY NOBLE BONUS
    # ============================================================

    def _buy_noble_bonus(
        self,
        state,
        action,
        phase,
        strategy,
    ):
        card = self._get_card_from_action(
            state,
            action,
        )

        if card is None:
            return 0.0

        bonus = (
            self._card_noble_alignment_bonus(
                state=state,
                card=card,
                strategy=strategy,
                include_completion=True,
            )
        )

        tier = self._action_tier(
            state,
            action,
        )

        # Kangaroo's observed engine comes primarily from Tier 1.
        # Only give this extra bonus if the card actually helps one
        # of the chosen noble routes.
        if (
            tier == 1
            and bonus > 0.0
            and phase in (
                self.EARLY,
                self.MID,
            )
        ):
            bonus += (
                self.TIER1_NOBLE_ENGINE_BONUS
            )

        return bonus

    # ============================================================
    # TAKE_GEMS NOBLE BONUS
    # ============================================================

    def _take_noble_bonus(
        self,
        state,
        action,
        phase,
        strategy,
    ):
        """
        Do not reward gems merely because their color matches a noble.

        Gems do not satisfy nobles directly.

        Instead:
            find visible / reserved cards whose PERMANENT BONUS helps
            the noble strategy, and reward a TAKE action only if it
            reduces distance to those cards.
        """

        player = state.players[
            state.current_player
        ]

        gems_after = dict(
            player.gems
        )

        for color in action.gem_colors:
            gems_after[color] += 1

        aligned_cards = (
            self._get_noble_aligned_cards(
                state=state,
                strategy=strategy,
            )
        )

        if not aligned_cards:
            return 0.0

        best_progress_value = 0.0

        for card, alignment in aligned_cards:

            before = self._distance_to_card(
                player,
                card,
            )

            after = self._distance_to_card(
                player,
                card,
                gems_after,
            )

            progress = (
                before
                - after
            )

            if progress <= 0:
                continue

            # Prefer progress toward cards that support both target
            # nobles over cards that support only one.
            value = (
                progress
                * self.TAKE_NOBLE_PROGRESS_WEIGHT
                * max(
                    1.0,
                    alignment / 3.0,
                )
            )

            best_progress_value = max(
                best_progress_value,
                value,
            )

        # In late game, noble-engine gem collection should matter
        # less than actual point conversion.
        if phase == self.LATE:
            best_progress_value *= 0.5

        return best_progress_value

    # ============================================================
    # CARD ↔ NOBLE ALIGNMENT
    # ============================================================

    def _card_noble_alignment_bonus(
        self,
        state,
        card,
        strategy,
        include_completion,
    ):
        player = state.players[
            state.current_player
        ]

        color = card.bonus_color

        target_nobles = strategy[
            "target_nobles"
        ]

        overlap_colors = strategy[
            "overlap_colors"
        ]

        relevant_count = 0

        for noble in target_nobles:

            if (
                noble.requirement[color]
                > player.bonuses[color]
            ):
                relevant_count += 1

        if relevant_count == 0:
            return 0.0

        bonus = (
            relevant_count
            * self.NOBLE_COLOR_BONUS
        )

        # Card color advances both selected nobles.
        if relevant_count >= 2:
            bonus += (
                self.DOUBLE_NOBLE_COLOR_BONUS
            )

        # User-requested special emphasis:
        # pair has at least 2 shared colors and this card gives one
        # of those shared permanent bonuses.
        if (
            strategy[
                "overlap_count"
            ] >= 2
            and color in overlap_colors
        ):
            bonus += (
                self.TWO_COLOR_OVERLAP_BONUS
            )

        if include_completion:
            completed = (
                self._count_nobles_completed_after_card(
                    player=player,
                    card=card,
                    nobles=target_nobles,
                )
            )

            bonus += (
                completed
                * self.NOBLE_COMPLETION_BONUS
            )

        return bonus

    # ============================================================
    # FIND NOBLE-ALIGNED CARDS
    # ============================================================

    def _get_noble_aligned_cards(
        self,
        state,
        strategy,
    ):
        cards = []

        player = state.players[
            state.current_player
        ]

        # Visible cards.
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
                    self._card_noble_alignment_bonus(
                        state=state,
                        card=card,
                        strategy=strategy,
                        include_completion=False,
                    )
                )

                if alignment > 0:
                    cards.append(
                        (
                            card,
                            alignment,
                        )
                    )

        # Own reserved cards.
        for card in player.reserved_cards:
            if card is None:
                continue

            alignment = (
                self._card_noble_alignment_bonus(
                    state=state,
                    card=card,
                    strategy=strategy,
                    include_completion=False,
                )
            )

            if alignment > 0:
                cards.append(
                    (
                        card,
                        alignment,
                    )
                )

        return cards

    # ============================================================
    # NOBLE PAIR STRATEGY
    # ============================================================

    def _get_noble_pair_strategy(
        self,
        state,
    ):
        """
        Select up to two nobles.

        Pair ordering:
            1. larger color overlap
            2. lower total missing permanent bonuses
            3. lower worst individual noble distance

        This deliberately makes a two-color overlap very attractive.
        """

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

        # Only one noble remains.
        if len(nobles) == 1:
            noble = nobles[0]

            missing = (
                self._noble_missing_total(
                    player,
                    noble,
                )
            )

            return {
                "target_nobles": [
                    noble
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

                overlap_colors = (
                    self._noble_overlap_colors(
                        noble_a,
                        noble_b,
                    )
                )

                missing_a = (
                    self._noble_missing_total(
                        player,
                        noble_a,
                    )
                )

                missing_b = (
                    self._noble_missing_total(
                        player,
                        noble_b,
                    )
                )

                total_missing = (
                    missing_a
                    + missing_b
                )

                worst_missing = max(
                    missing_a,
                    missing_b,
                )

                # Primary objective:
                # maximize shared colors.
                #
                # Secondary:
                # prefer the pair our current engine is closer to.
                rank = (
                    len(
                        overlap_colors
                    ),
                    -total_missing,
                    -worst_missing,
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

    def _noble_overlap_colors(
        self,
        noble_a,
        noble_b,
    ):
        """
        A color overlaps when BOTH nobles require that color.
        """

        return {
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

    # ============================================================
    # NOBLE DISTANCE / COMPLETION
    # ============================================================

    def _noble_missing_total(
        self,
        player,
        noble,
    ):
        missing = 0

        for color in COLOR_ORDER:
            missing += max(
                0,
                noble.requirement[
                    color
                ]
                - player.bonuses[
                    color
                ],
            )

        return missing

    def _count_nobles_completed_after_card(
        self,
        player,
        card,
        nobles,
    ):
        completed = 0

        for noble in nobles:

            qualifies = True

            for color in COLOR_ORDER:

                bonus_count = (
                    player.bonuses[
                        color
                    ]
                )

                if (
                    color
                    == card.bonus_color
                ):
                    bonus_count += 1

                if (
                    bonus_count
                    < noble.requirement[
                        color
                    ]
                ):
                    qualifies = False
                    break

            if qualifies:
                completed += 1

        return completed

    # ============================================================
    # OPTIONAL DEBUGGING
    # ============================================================

    def get_noble_strategy_debug(
        self,
        state,
    ):
        """
        Inspect the current Kangaroo noble plan.

        Useful before running a large benchmark.

        Example output shape:

            {
                "phase": "early",
                "overlap_count": 2,
                "overlap_colors": {...},
                "total_missing": 11,
                ...
            }
        """

        strategy = (
            self._get_noble_pair_strategy(
                state
            )
        )

        return {
            "phase":
                self._get_game_phase(
                    state
                ),

            "target_noble_count":
                len(
                    strategy[
                        "target_nobles"
                    ]
                ),

            "target_nobles":
                strategy[
                    "target_nobles"
                ],

            "overlap_colors":
                strategy[
                    "overlap_colors"
                ],

            "overlap_count":
                strategy[
                    "overlap_count"
                ],

            "total_missing":
                strategy[
                    "total_missing"
                ],

            "individual_missing":
                strategy[
                    "individual_missing"
                ],
        }
