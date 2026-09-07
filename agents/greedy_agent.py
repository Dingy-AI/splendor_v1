from splendor_v1.env.core.actions import Action
from splendor_v1.env.core.enums import ActionType


class GreedyAgent:

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

        # ---------------------------------
        # 1. Buy highest-point card
        # ---------------------------------

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
                    self._get_card_points(
                        state,
                        action,
                    ),
            )

        # ---------------------------------
        # 2. Prefer taking 3 gems
        # ---------------------------------

        take_three_actions = [
            action
            for action in legal_actions
            if (
                action.action_type
                == ActionType.TAKE_GEMS
                and action.gem_colors is not None
                and len(action.gem_colors) == 3
            )
        ]

        if take_three_actions:
            return take_three_actions[0]

        # ---------------------------------
        # 3. Otherwise take gems
        # ---------------------------------

        take_gem_actions = [
            action
            for action in legal_actions
            if (
                action.action_type
                == ActionType.TAKE_GEMS
            )
        ]

        if take_gem_actions:

            # Prefer taking the largest
            # number of gems available.
            return max(
                take_gem_actions,
                key=lambda action:
                    len(action.gem_colors),
            )

        # ---------------------------------
        # 4. Fallback
        # ---------------------------------

        return legal_actions[0]

    def _get_card_points(
        self,
        state,
        action: Action,
    ) -> int:

        if (
            action.action_type
            == ActionType.BUY_VISIBLE
        ):

            card = state.visible_cards[
                action.tier
            ][
                action.slot
            ]

            return card.points

        if (
            action.action_type
            == ActionType.BUY_RESERVED
        ):

            player = state.players[
                state.current_player
            ]

            card = player.reserved_cards[
                action.reserved_index
            ]

            return card.points

        return -1