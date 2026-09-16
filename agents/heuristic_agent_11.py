from splendor_v1.agents.heuristic_agent_9 import HeuristicAgent9
from splendor_v1.env.core.constants import COLOR_ORDER
from splendor_v1.env.core.enums import GemColor
from splendor_v1.env.core.actions import ActionType


class HeuristicAgent11(HeuristicAgent9):
    """
    HeuristicAgent11 = reserve / high-point Tier 2-3 strategy.

    H11 keeps H9's search architecture:

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

    Strategic identity
    ------------------
    H11 is based on the elite-player pattern opposite the
    Kangaroo noble-engine route:

        - smaller Tier 1 engine
        - more frequent reservation
        - more gold acquisition
        - more Tier 2 / Tier 3 point conversion
        - less reliance on nobles
        - secure high-value cards before opponents can take them

    H11 does NOT directly choose "reserve" or "buy Tier 3".

    It changes only which actions are likely to survive H9's
    root pruning and receive full terminal rollout calculation.

    Final move selection still comes from actual H3 terminal
    rollout results.

    Core ideas
    ----------
    1. Early:
        build only enough Tier 1 engine to unlock valuable
        Tier 2 / Tier 3 routes.

    2. Mid:
        emphasize Tier 2 buys and strategic reservation of
        valuable Tier 2 / Tier 3 cards.

    3. Late:
        emphasize direct point conversion through Tier 2 / Tier 3,
        immediate wins, and reserve-based denial / access.

    4. Reservation value:
        reserve becomes attractive when the card is:
            - worth many points
            - relatively close to affordable
            - efficient in points per remaining distance
            - likely to matter soon

    5. TAKE_GEMS value:
        reward takes that move toward the best Tier 2 / Tier 3
        point targets, especially reserved targets.

    Environment/API:
        inherited from HeuristicAgent9.
    """

    # ============================================================
    # SEARCH ALLOCATION
    # ============================================================

    RESERVE_EARLY_TARGETS = {
        "buy": 3,
        "take": 3,
        "reserve": 2,
    }

    RESERVE_MID_TARGETS = {
        "buy": 4,
        "take": 2,
        "reserve": 2,
    }

    RESERVE_LATE_TARGETS = {
        "buy": 4,
        "take": 1,
        "reserve": 3,
    }

    # ============================================================
    # POINT-ROUTE WEIGHTS
    # ============================================================

    # Point-bearing card priority.
    POINT_CARD_WEIGHT = 2.5

    # Extra preference for Tier 2 / Tier 3.
    TIER2_POINT_BONUS = 4.0
    TIER3_POINT_BONUS = 8.0

    # Strong bonus for cards close to affordable.
    NEAR_TARGET_BONUS = 3.0

    # Reserve-specific bonuses.
    RESERVE_POINT_WEIGHT = 2.0
    RESERVE_NEAR_WEIGHT = 2.5
    RESERVE_TIER2_BONUS = 3.0
    RESERVE_TIER3_BONUS = 6.0

    # Buying a previously reserved high-point card is strongly aligned
    # with this strategy.
    BUY_RESERVED_BONUS = 5.0

    # TAKE_GEMS toward Tier 2 / Tier 3 point targets.
    TAKE_PROGRESS_WEIGHT = 3.0
    RESERVED_TARGET_MULTIPLIER = 1.35

    # Cap strategic shortlist adjustment.
    MAX_POINT_ROUTE_BONUS = 20.0

    # Tier 1 is still useful, but should not dominate this archetype.
    TIER1_ENGINE_PENALTY_LATE = 3.0

    # ============================================================
    # PHASE TARGET OVERRIDES
    # ============================================================

    def _phase_targets(
        self,
        phase,
    ):
        if phase == self.EARLY:
            return dict(
                self.RESERVE_EARLY_TARGETS
            )

        if phase == self.MID:
            return dict(
                self.RESERVE_MID_TARGETS
            )

        return dict(
            self.RESERVE_LATE_TARGETS
        )

    def _phase_group_order(
        self,
        phase,
    ):
        if phase == self.EARLY:
            return [
                "buy",
                "take",
                "reserve",
                "other",
            ]

        if phase == self.MID:
            return [
                "buy",
                "reserve",
                "take",
                "other",
            ]

        return [
            "buy",
            "reserve",
            "take",
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
        Reserve-route phase bias.

        Relative to H9:
            - less Tier 1 emphasis
            - more Tier 2 / Tier 3 emphasis
            - reserve is much more viable
        """

        if group == "buy":

            if phase == self.EARLY:
                if tier == 1:
                    return 5.0
                if tier == 2:
                    return 4.0
                if tier == 3:
                    return 0.0
                return 2.0

            if phase == self.MID:
                if tier == 1:
                    return 0.0
                if tier == 2:
                    return 9.0
                if tier == 3:
                    return 6.0
                return 3.0

            # LATE
            if tier == 1:
                return -5.0
            if tier == 2:
                return 7.0
            if tier == 3:
                return 12.0
            return 5.0

        if group == "take":
            if phase == self.EARLY:
                return 5.0
            if phase == self.MID:
                return 3.0
            return 1.0

        if group == "reserve":
            if phase == self.EARLY:
                if tier == 1:
                    return -1.0
                if tier == 2:
                    return 4.0
                if tier == 3:
                    return 3.0
                return -2.0

            if phase == self.MID:
                if tier == 1:
                    return -2.0
                if tier == 2:
                    return 6.0
                if tier == 3:
                    return 7.0
                return -1.0

            # LATE
            if tier == 1:
                return -4.0
            if tier == 2:
                return 5.0
            if tier == 3:
                return 9.0
            return -1.0

        return 0.0

    # ============================================================
    # ROOT RECORD OVERRIDE
    # ============================================================

    def _make_action_record(
        self,
        state,
        action,
        phase,
    ):
        """
        H9 normally excludes Tier 2 / Tier 3 reserves in EARLY.

        H11 must allow them, because reservation is a defining
        part of this strategy.

        Therefore this method reproduces H9's record construction
        without the early Tier 2/3 reserve exclusion.
        """

        group = self._action_group(
            action
        )

        tier = self._action_tier(
            state,
            action,
        )

        base_score = (
            self._cheap_h3_action_score(
                state,
                action,
            )
        )

        try:
            finite = (
                float(base_score)
                != float("inf")
                and float(base_score)
                != float("-inf")
            )
        except Exception:
            finite = False

        if not finite:
            return None

        phase_bonus = (
            self._phase_bonus(
                phase=phase,
                group=group,
                tier=tier,
                action=action,
            )
        )

        strategy_bonus = (
            self._point_route_action_bonus(
                state=state,
                action=action,
                phase=phase,
                tier=tier,
            )
        )

        strategy_bonus = max(
            -self.MAX_POINT_ROUTE_BONUS,
            min(
                self.MAX_POINT_ROUTE_BONUS,
                strategy_bonus,
            ),
        )

        selection_score = (
            base_score
            + phase_bonus
            + strategy_bonus
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

            "strategy_bonus":
                float(
                    strategy_bonus
                ),

            "selection_score":
                float(
                    selection_score
                ),

            "mandatory":
                mandatory,
        }

    # ============================================================
    # STRATEGY BONUS
    # ============================================================

    def _point_route_action_bonus(
        self,
        state,
        action,
        phase,
        tier,
    ):
        action_type = (
            action.action_type
        )

        if action_type in (
            ActionType.BUY_VISIBLE,
            ActionType.BUY_RESERVED,
        ):
            return self._buy_point_route_bonus(
                state=state,
                action=action,
                phase=phase,
                tier=tier,
            )

        if (
            action_type
            == ActionType.TAKE_GEMS
        ):
            return self._take_point_route_bonus(
                state=state,
                action=action,
                phase=phase,
            )

        if (
            action_type
            == ActionType.RESERVE_VISIBLE
        ):
            return self._reserve_point_route_bonus(
                state=state,
                action=action,
                phase=phase,
                tier=tier,
            )

        if (
            action_type
            == ActionType.RESERVE_TOP_DECK
        ):
            # Blind reserves are much less attractive because
            # they do not secure a known high-value card.
            return -4.0

        return 0.0

    # ============================================================
    # BUY BONUS
    # ============================================================

    def _buy_point_route_bonus(
        self,
        state,
        action,
        phase,
        tier,
    ):
        card = self._get_card_from_action(
            state,
            action,
        )

        if card is None:
            return 0.0

        player = state.players[
            state.current_player
        ]

        distance = self._distance_to_card(
            player,
            card,
        )

        bonus = (
            card.points
            * self.POINT_CARD_WEIGHT
        )

        if tier == 2:
            bonus += (
                self.TIER2_POINT_BONUS
            )

        elif tier == 3:
            bonus += (
                self.TIER3_POINT_BONUS
            )

        # Buying a reserved point card means the reserve/gold setup
        # successfully converted into points.
        if (
            action.action_type
            == ActionType.BUY_RESERVED
        ):
            bonus += (
                self.BUY_RESERVED_BONUS
            )

        if distance <= 1:
            bonus += (
                self.NEAR_TARGET_BONUS
            )

        # Tier 1 pointless engine pieces should gradually lose
        # shortlist priority in this archetype.
        if (
            tier == 1
            and card.points == 0
            and phase == self.LATE
        ):
            bonus -= (
                self.TIER1_ENGINE_PENALTY_LATE
            )

        return bonus

    # ============================================================
    # RESERVE BONUS
    # ============================================================

    def _reserve_point_route_bonus(
        self,
        state,
        action,
        phase,
        tier,
    ):
        card = self._get_card_from_action(
            state,
            action,
        )

        if card is None:
            return 0.0

        player = state.players[
            state.current_player
        ]

        distance = self._distance_to_card(
            player,
            card,
        )

        bonus = (
            card.points
            * self.RESERVE_POINT_WEIGHT
        )

        if tier == 2:
            bonus += (
                self.RESERVE_TIER2_BONUS
            )

        elif tier == 3:
            bonus += (
                self.RESERVE_TIER3_BONUS
            )

        # Prefer reserving cards that are not absurdly far away.
        if distance <= 1:
            bonus += (
                self.RESERVE_NEAR_WEIGHT
                * 2.0
            )

        elif distance == 2:
            bonus += (
                self.RESERVE_NEAR_WEIGHT
            )

        elif distance >= 5:
            bonus -= 3.0

        # Late game: points are worth much more than engine.
        if phase == self.LATE:
            bonus += (
                card.points
                * 1.5
            )

        return bonus

    # ============================================================
    # TAKE_GEMS BONUS
    # ============================================================

    def _take_point_route_bonus(
        self,
        state,
        action,
        phase,
    ):
        """
        Reward gem actions that concretely reduce distance to
        high-value Tier 2 / Tier 3 targets.

        Reserved targets receive extra weight because this strategy
        intentionally secures cards before collecting everything
        needed to buy them.
        """

        player = state.players[
            state.current_player
        ]

        gems_after = dict(
            player.gems
        )

        for color in action.gem_colors:
            gems_after[color] += 1

        targets = (
            self._get_point_targets(
                state
            )
        )

        if not targets:
            return 0.0

        best_value = 0.0

        for target in targets:
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
                progress
                * self.TAKE_PROGRESS_WEIGHT
            )

            value *= (
                1.0
                + card.points
                * 0.25
            )

            if target[
                "reserved"
            ]:
                value *= (
                    self.RESERVED_TARGET_MULTIPLIER
                )

            # In late game, direct point-target progress matters more.
            if phase == self.LATE:
                value *= 1.25

            best_value = max(
                best_value,
                value,
            )

        return best_value

    # ============================================================
    # POINT TARGET DISCOVERY
    # ============================================================

    def _get_point_targets(
        self,
        state,
    ):
        """
        Build a small pool of high-value Tier 2 / Tier 3 cards.

        Visible cards and our own reserved cards are included.

        This is only used for cheap shortlist scoring.
        """

        player = state.players[
            state.current_player
        ]

        targets = []

        # Visible Tier 2 / Tier 3.
        for tier in (
            2,
            3,
        ):
            for card in state.visible_cards[
                tier
            ]:
                if card is None:
                    continue

                score = (
                    self._point_target_score(
                        player,
                        card,
                        tier=tier,
                        reserved=False,
                    )
                )

                targets.append(
                    {
                        "card":
                            card,
                        "tier":
                            tier,
                        "reserved":
                            False,
                        "score":
                            score,
                    }
                )

        # Reserved cards.
        for card in player.reserved_cards:
            if card is None:
                continue

            tier = None

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
                    tier = value
                    break

            # If original tier is unavailable, still include point
            # cards because they are central to this archetype.
            if (
                tier not in (
                    2,
                    3,
                )
                and card.points <= 0
            ):
                continue

            score = (
                self._point_target_score(
                    player,
                    card,
                    tier=tier,
                    reserved=True,
                )
            )

            targets.append(
                {
                    "card":
                        card,
                    "tier":
                        tier,
                    "reserved":
                        True,
                    "score":
                        score,
                }
            )

        targets.sort(
            key=lambda item:
                item["score"],
            reverse=True,
        )

        # Keep only the most relevant few for cheap progress scoring.
        return targets[:6]

    def _point_target_score(
        self,
        player,
        card,
        tier,
        reserved,
    ):
        distance = (
            self._distance_to_card(
                player,
                card,
            )
        )

        score = (
            card.points * 10.0
        )

        score -= (
            distance * 3.0
        )

        if tier == 2:
            score += 2.0

        elif tier == 3:
            score += 5.0

        if reserved:
            score += 4.0

        return score

    # ============================================================
    # OPTIONAL DEBUGGING
    # ============================================================

    def get_point_route_debug(
        self,
        state,
    ):
        """
        Inspect current H11 strategic targets.
        """

        targets = (
            self._get_point_targets(
                state
            )
        )

        return {
            "phase":
                self._get_game_phase(
                    state
                ),

            "targets":
                targets,

            "reserve_strategy":
                True,

            "num_calc_moves":
                self.num_calc_moves,

            "num_rollouts":
                self.num_rollouts,
        }
