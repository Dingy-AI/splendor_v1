from splendor_v1.env.core.constants import COLOR_ORDER
from splendor_v1.env.core.enums import GemColor
from splendor_v1.env.core.actions import ActionType

class HeuristicAgent:


    def _distance_to_card(
        self,
        player,
        card,
        gems=None,
    ):

        if gems is None:
            gems = player.gems

        missing = 0

        for color in COLOR_ORDER:

            required = max(
                0,
                card.cost.get(color, 0)
                - player.bonuses[color]
            )

            missing += max(
                0,
                required - gems[color]
            )

        # Gold can cover remaining shortages
        missing = max(
            0,
            missing - gems[GemColor.GOLD]
        )

        return missing


    def _score_card(
        self,
        player,
        card,
        nobles,
    ):

        effective_cost = sum(
            max(
                0,
                card.cost.get(color, 0)
                - player.bonuses[color]
            )
            for color in COLOR_ORDER
        )

        score = 0.0

        # Immediate victory points
        score += card.points * 10.0

        # Cheaper cards are better
        score -= effective_cost * 0.5

        # Permanent bonus always has some value
        score += 2.0

        # Noble progress
        bonus_color = card.bonus_color

        for noble in nobles:

            if noble is None:
                continue

            requirement = noble.requirement[
                bonus_color
            ]

            current_bonus = player.bonuses[
                bonus_color
            ]

            if current_bonus < requirement:
                score += 2.0

        return score

    def _score_take_gems(
        self,
        state,
        action,
    ):

        player = state.players[
            state.current_player
        ]

        gems_after = dict(player.gems)

        for color in action.gem_colors:
            gems_after[color] += 1

        cards = []

        for tier in (1, 2, 3):
            cards.extend(
                card
                for card in state.visible_cards[tier]
                if card is not None
            )
        cards.extend(
            card
            for card in player.reserved_cards
            if card is not None
        )

        best_score = float("-inf")

        for card in cards:

            before = self._distance_to_card(
                player,
                card,
            )

            after = self._distance_to_card(
                player,
                card,
                gems_after,
            )

            progress = before - after

            card_value = self._score_card(
                player,
                card,
                state.nobles,
            )

            score = (
                progress * 10.0
                + card_value
                / (after + 1)
            )

            best_score = max(
                best_score,
                score,
            )

        return best_score


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

        if take_actions:
            return max(
                take_actions,
                key=lambda action:
                    self._score_take_gems(
                        state,
                        action,
                    ),
            )

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

        if reserve_actions:
            return max(
                reserve_actions,
                key=lambda action:
                    self._score_reserve_action(
                        state,
                        action,
                    ),
            )

        # Forced discard / noble / fallback
        return legal_actions[0]

    def _get_card_from_action(
        self,
        state,
        action,
    ):

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

        if action.action_type == ActionType.BUY_RESERVED:
            return player.reserved_cards[
                action.reserved_index
            ]

        return None

    def _score_buy_action(
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
            return float("-inf")

        score = self._score_card(
            player,
            card,
            state.nobles,
        )

        # Prefer payment options that preserve gold.
        if action.gold_payment is not None:
            gold_used = sum(
                action.gold_payment
            )

            score -= gold_used * 1.0

        return score

    def _score_reserve_action(
        self,
        state,
        action,
    ):

        player = state.players[
            state.current_player
        ]

        # Unknown top-deck card
        if (
            action.action_type
            == ActionType.RESERVE_TOP_DECK
        ):
            return -5.0

        card = self._get_card_from_action(
            state,
            action,
        )

        if card is None:
            return float("-inf")

        card_score = self._score_card(
            player,
            card,
            state.nobles,
        )

        distance = self._distance_to_card(
            player,
            card,
        )

        # Reserving is more attractive when
        # the card is valuable AND reasonably
        # attainable.
        score = (
            card_score
            - distance * 2.0
        )

        return score