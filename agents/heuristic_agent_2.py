from splendor_v1.agents.heuristic_agent import HeuristicAgent
from splendor_v1.env.core.constants import COLOR_ORDER
from splendor_v1.env.core.enums import GemColor
from splendor_v1.env.core.actions import ActionType


class HeuristicAgent2(HeuristicAgent):

    # How important stealing useful gems is.
    GEM_DENIAL_WEIGHT = 0.5

    # Buying a card the opponent wants has extra value.
    BUY_DENIAL_WEIGHT = 0.15

    # Reserving is primarily our blocking tool.
    RESERVE_DENIAL_WEIGHT = 0.25

    # If the opponent threat exceeds this,
    # H2 will reserve instead of taking gems.


    def _get_opponent(
        self,
        state,
    ):
        if len(state.players) != 2:
            raise ValueError(
                "HeuristicAgent2 currently "
                "supports only 2-player games."
            )

        opponent_index = (
            1 - state.current_player
        )

        return state.players[
            opponent_index
        ]


    # ============================================================
    # Opponent points
    # ============================================================

    def _opponent_urgency(
        self,
        opponent,
    ):
        """
        Opponent becomes increasingly dangerous
        as they approach 15 points.

        0 points  -> 1.0
        10 points -> 1.67
        15 points -> 2.0
        """

        return (
            1.0
            + opponent.points / 15.0
        )


    # ============================================================
    # Opponent noble progress
    # ============================================================

    def _noble_distance(
        self,
        player,
        noble,
        extra_bonus_color=None,
    ):

        missing = 0

        for color in COLOR_ORDER:

            bonuses = player.bonuses[color]

            if color == extra_bonus_color:
                bonuses += 1

            missing += max(
                0,
                noble.requirement[color]
                - bonuses,
            )

        return missing


    def _opponent_noble_pressure(
        self,
        opponent,
        card,
        nobles,
    ):
        """
        How much would this card help the
        opponent toward nobles?
        """

        score = 0.0

        for noble in nobles:

            if noble is None:
                continue

            before = self._noble_distance(
                opponent,
                noble,
            )

            after = self._noble_distance(
                opponent,
                noble,
                extra_bonus_color=(
                    card.bonus_color
                ),
            )

            # Card actually advances noble.
            if after < before:
                score += 2.0

            # Card immediately completes noble.
            if before > 0 and after == 0:
                score += (
                    8.0
                    + noble.points
                )

        return score


    # ============================================================
    # Opponent card threat
    # ============================================================

    def _opponent_card_threat(
        self,
        state,
        card,
    ):

        opponent = self._get_opponent(
            state
        )

        distance = self._distance_to_card(
            opponent,
            card,
        )

        threat = 0.0

        # --------------------------
        # Gems + bonuses
        # --------------------------

        # Opponent can buy immediately.
        if distance == 0:
            threat += 8.0

        # One gem away.
        elif distance == 1:
            threat += 4.0

        # Close.
        elif distance == 2:
            threat += 1.0

        # --------------------------
        # Card points
        # --------------------------

        threat += (
            card.points * 2.0
        )

        # --------------------------
        # Noble progress
        # --------------------------

        threat += self._opponent_noble_pressure(
            opponent,
            card,
            state.nobles,
        )

        # --------------------------
        # Immediate winning threat
        # --------------------------

        if (
            distance == 0
            and (
                opponent.points
                + card.points
                >= 15
            )
        ):
            threat += 25.0

        # --------------------------
        # Opponent overall score
        # --------------------------

        threat *= self._opponent_urgency(
            opponent
        )

        return threat

    def _opponent_gem_demand(
        self,
        state,
    ):

        opponent = self._get_opponent(
            state
        )

        demand = {
            color: 0.0
            for color in COLOR_ORDER
        }

        target_cards = []

        # Visible cards are possible targets.
        for tier in (1, 2, 3):

            for card in state.visible_cards[
                tier
            ]:

                if card is not None:
                    target_cards.append(
                        (card, 1.0)
                    )

        # Reserved cards are stronger signals:
        # opponent has already committed a turn
        # to acquiring them.
        for card in opponent.reserved_cards:

            if card is not None:
                target_cards.append(
                    (card, 1.5)
                )

        for card, source_weight in target_cards:

            distance = self._distance_to_card(
                opponent,
                card,
            )

            # Ignore cards that are still
            # very far away.
            if distance > 4:
                continue

            card_weight = (
                source_weight
                * (
                    1.0
                    + card.points * 0.5
                )
                / (distance + 1)
            )

            for color in COLOR_ORDER:

                required = max(
                    0,
                    card.cost.get(
                        color,
                        0,
                    )
                    - opponent.bonuses[color],
                )

                shortage = max(
                    0,
                    required
                    - opponent.gems[color],
                )

                if shortage > 0:
                    demand[color] += (
                        card_weight
                        * shortage
                    )

        return demand

    def _score_take_gems(
        self,
        state,
        action,
    ):

        # H1's own-card planning.
        score = super()._score_take_gems(
            state,
            action,
        )

        opponent_demand = (
            self._opponent_gem_demand(
                state
            )
        )

        denial_score = 0.0

        for color in action.gem_colors:

            if color == GemColor.GOLD:
                continue

            denial_score += (
                opponent_demand[color]
            )

        score += (
            denial_score
            * self.GEM_DENIAL_WEIGHT
        )

        return score

    def _score_buy_action(
        self,
        state,
        action,
    ):

        # Original H1 score.
        score = super()._score_buy_action(
            state,
            action,
        )

        # Reserved card belongs to us already,
        # so buying it doesn't deny opponent.
        if (
            action.action_type
            == ActionType.BUY_RESERVED
        ):
            return score

        card = self._get_card_from_action(
            state,
            action,
        )

        if card is None:
            return score

        threat = self._opponent_card_threat(
            state,
            card,
        )

        score += (
            threat
            * self.BUY_DENIAL_WEIGHT
        )

        return score

    def _score_reserve_action(
        self,
        state,
        action,
    ):

        score = super()._score_reserve_action(
            state,
            action,
        )

        if (
            action.action_type
            == ActionType.RESERVE_TOP_DECK
        ):
            return score

        card = self._get_card_from_action(
            state,
            action,
        )

        if card is None:
            return score

        threat = self._opponent_card_threat(
            state,
            card,
        )

        score += (
            threat
            * self.RESERVE_DENIAL_WEIGHT
        )

        return score

    def _is_critical_block(
        self,
        state,
        card,
    ):
        opponent = self._get_opponent(
            state
        )

        # They cannot buy it yet.
        if self._distance_to_card(
            opponent,
            card,
        ) != 0:
            return False

        # Winning card.
        if (
            opponent.points
            + card.points
            >= 15
        ):
            return True

        # Large immediate VP swing.
        if card.points >= 3:
            return True

        # Does buying this card complete a noble?
        for noble in state.nobles:

            if noble is None:
                continue

            before = self._noble_distance(
                opponent,
                noble,
            )

            after = self._noble_distance(
                opponent,
                noble,
                extra_bonus_color=card.bonus_color,
            )

            if before > 0 and after == 0:
                return True

        return False

    def select_action(
        self,
        env,
        state,
    ):

        legal_actions = env._legal_actions(
            state
        )

        if not legal_actions:
            return None

        # -------------------------
        # BUY
        # -------------------------

        buy_actions = [
            action
            for action in legal_actions
            if action.action_type in (
                ActionType.BUY_VISIBLE,
                ActionType.BUY_RESERVED,
            )
        ]

        if buy_actions:
            return max(
                buy_actions,
                key=lambda action:
                    self._score_buy_action(
                        state,
                        action,
                    ),
            )

        # -------------------------
        # TAKE GEMS
        # -------------------------

        take_actions = [
            action
            for action in legal_actions
            if (
                action.action_type
                == ActionType.TAKE_GEMS
            )
        ]

        # -------------------------
        # RESERVE
        # -------------------------

        reserve_actions = [
            action
            for action in legal_actions
            if action.action_type in (
                ActionType.RESERVE_VISIBLE,
                ActionType.RESERVE_TOP_DECK,
            )
        ]

        # -------------------------
        # CRITICAL BLOCK
        # -------------------------

        critical_reserves = [
            action
            for action in reserve_actions
            if (
                action.action_type
                == ActionType.RESERVE_VISIBLE
                and self._is_critical_block(
                    state,
                    self._get_card_from_action(
                        state,
                        action,
                    ),
                )
            )
        ]

        if critical_reserves:
            return max(
                critical_reserves,
                key=lambda action:
                    self._score_reserve_action(
                        state,
                        action,
                    ),
            )

        # -------------------------
        # NORMAL H1 BEHAVIOR
        # -------------------------

        if take_actions:
            return max(
                take_actions,
                key=lambda action:
                    self._score_take_gems(
                        state,
                        action,
                    ),
            )

        if reserve_actions:
            return max(
                reserve_actions,
                key=lambda action:
                    self._score_reserve_action(
                        state,
                        action,
                    ),
            )

        return legal_actions[0]