from splendor_v1.env.core.constants import COLOR_ORDER
from splendor_v1.env.core.enums import GemColor
from splendor_v1.env.core.actions import ActionType
import numpy as np


class HeuristicAgent5:
    """
    Targeted opponent-aware upgrade of HeuristicAgent3.

    Keeps H3's simple BUY -> TAKE -> RESERVE structure, with one
    exception: an immediately critical opponent card may justify a
    defensive reserve.

    Adds:
      - opponent visible-card threat detection
      - targeted reservation denial
      - gem-denial awareness
      - better future bonus-color relevance
      - better noble progress weighting
      - endgame urgency near 15 points

    No minimax and no speculative multi-turn rollout.
    """

    CRITICAL_RESERVE_THRESHOLD = 30.0

    IMMEDIATE_DISTANCE = 0
    NEAR_DISTANCE = 1
    CLOSE_DISTANCE = 2

    # ============================================================
    # CANDIDATE SELECTION
    # ============================================================

    def _get_scored_candidates(self, env, state):

        legal_actions = env._legal_actions(state)

        if not legal_actions:
            return []

        buy_actions = [
            action
            for action in legal_actions
            if action.action_type in (
                ActionType.BUY_VISIBLE,
                ActionType.BUY_RESERVED,
            )
        ]

        take_actions = [
            action
            for action in legal_actions
            if action.action_type == ActionType.TAKE_GEMS
        ]

        reserve_actions = [
            action
            for action in legal_actions
            if action.action_type in (
                ActionType.RESERVE_VISIBLE,
                ActionType.RESERVE_TOP_DECK,
            )
        ]

        # Find only genuinely urgent reserve actions.
        critical_reserves = []

        for action in reserve_actions:

            if action.action_type != ActionType.RESERVE_VISIBLE:
                continue

            denial_value = self._reserve_denial_value(
                state,
                action,
            )

            if denial_value >= self.CRITICAL_RESERVE_THRESHOLD:

                critical_reserves.append(
                    (
                        action,
                        self._score_reserve_action(
                            state,
                            action,
                        ),
                    )
                )

        # --------------------------------------------------------
        # Preserve H3's hierarchy as the default.
        # Critical denial reserve is the only exception.
        # --------------------------------------------------------

        if buy_actions:

            candidates = [
                (
                    action,
                    self._score_buy_action(
                        state,
                        action,
                    ),
                )
                for action in buy_actions
            ]

            candidates.extend(
                critical_reserves
            )

            return candidates

        if take_actions:

            candidates = [
                (
                    action,
                    self._score_take_gems(
                        state,
                        action,
                    ),
                )
                for action in take_actions
            ]

            candidates.extend(
                critical_reserves
            )

            return candidates

        if reserve_actions:

            return [
                (
                    action,
                    self._score_reserve_action(
                        state,
                        action,
                    ),
                )
                for action in reserve_actions
            ]

        # Forced transition / fallback.
        return [
            (action, 0.0)
            for action in legal_actions
        ]

    # ============================================================
    # PUBLIC API
    # ============================================================

    def select_action(self, env, state):

        scored_actions = self._get_scored_candidates(
            env,
            state,
        )

        if not scored_actions:
            return None

        return max(
            scored_actions,
            key=lambda x: x[1],
        )[0]

    def get_policy(
        self,
        env,
        state,
        action_size=1139,
        temperature=10.0,
    ):

        if temperature <= 0:
            raise ValueError(
                "temperature must be greater than 0"
            )

        scored_actions = self._get_scored_candidates(
            env,
            state,
        )

        policy = np.zeros(
            action_size,
            dtype=np.float32,
        )

        if not scored_actions:
            return policy

        actions = [
            action
            for action, _ in scored_actions
        ]

        scores = np.array(
            [
                score
                for _, score in scored_actions
            ],
            dtype=np.float64,
        )

        if not np.any(np.isfinite(scores)):

            probs = np.ones(
                len(scores),
                dtype=np.float64,
            )

            probs /= probs.sum()

        else:

            scores = np.where(
                np.isfinite(scores),
                scores,
                -1e9,
            )

            scores -= np.max(scores)

            probs = np.exp(
                scores / temperature
            )

            probs /= probs.sum()

        for action, prob in zip(
            actions,
            probs,
        ):

            action_id = env.action_to_id(
                action
            )

            policy[action_id] += prob

        return policy

    # ============================================================
    # BUY SCORING
    # ============================================================

    def _score_buy_action(self, state, action):

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
            state=state,
        )

        current_points = getattr(
            player,
            "points",
            0,
        )

        # Winning / near-winning cards become urgent.
        if current_points + card.points >= 15:

            score += 100.0

        elif current_points >= 10:

            score += (
                card.points * 2.0
            )

        # Buying a contested visible card also denies it.
        if action.action_type == ActionType.BUY_VISIBLE:

            score += (
                self._opponent_card_threat(
                    state,
                    card,
                )
                * 0.20
            )

        # Prefer preserving gold.
        if action.gold_payment is not None:

            gold_used = sum(
                action.gold_payment
            )

            score -= (
                gold_used * 1.0
            )

        return score

    # ============================================================
    # RESERVE SCORING
    # ============================================================

    def _score_reserve_action(self, state, action):

        player = state.players[
            state.current_player
        ]

        if action.action_type == ActionType.RESERVE_TOP_DECK:
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
            state=state,
        )

        distance = self._distance_to_card(
            player,
            card,
        )

        own_value = (
            card_score
            - distance * 2.0
        )

        denial_value = self._reserve_denial_value(
            state,
            action,
        )

        return (
            own_value
            + denial_value
        )

    def _reserve_denial_value(self, state, action):

        if action.action_type != ActionType.RESERVE_VISIBLE:
            return 0.0

        card = self._get_card_from_action(
            state,
            action,
        )

        if card is None:
            return 0.0

        return self._opponent_card_threat(
            state,
            card,
        )

    # ============================================================
    # TAKE-GEM SCORING
    # ============================================================

    def _score_take_gems(self, state, action):

        player = state.players[
            state.current_player
        ]

        opponent = self._get_opponent(
            state
        )

        gems_after = dict(
            player.gems
        )

        for color in action.gem_colors:
            gems_after[color] += 1

        cards = self._all_relevant_cards(
            state,
            player,
        )

        best_score = float("-inf")

        # H3's original "progress toward my best card" logic.
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

            progress = (
                before - after
            )

            card_value = self._score_card(
                player,
                card,
                state.nobles,
                state=state,
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

        if best_score == float("-inf"):
            best_score = 0.0

        # --------------------------------------------------------
        # Opponent-aware gem denial.
        # --------------------------------------------------------

        if opponent is not None:

            opponent_need = self._opponent_color_need(
                state,
                opponent,
            )

            for color in action.gem_colors:

                if color == GemColor.GOLD:
                    continue

                best_score += (
                    opponent_need.get(
                        color,
                        0.0,
                    )
                    * 1.25
                )

            # Taking 2 from exactly 4 removes the opponent's
            # immediate take-two option for that color.
            color_counts = {}

            for color in action.gem_colors:

                color_counts[color] = (
                    color_counts.get(
                        color,
                        0,
                    )
                    + 1
                )

            for color, count in color_counts.items():

                if (
                    color != GemColor.GOLD
                    and count >= 2
                    and state.bank[color] == 4
                ):

                    best_score += (
                        opponent_need.get(
                            color,
                            0.0,
                        )
                        * 2.0
                    )

        return best_score

    # ============================================================
    # CARD SCORING
    # ============================================================

    def _score_card(
        self,
        player,
        card,
        nobles,
        state=None,
    ):

        effective_cost = sum(
            max(
                0,
                card.cost.get(
                    color,
                    0,
                )
                - player.bonuses[color]
            )
            for color in COLOR_ORDER
        )

        score = 0.0

        score += (
            card.points * 10.0
        )

        score -= (
            effective_cost * 0.5
        )

        score += 2.0

        bonus_color = card.bonus_color

        # Better noble-progress signal than H3:
        # bonus is increasingly useful when close to requirement.
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

                remaining = (
                    requirement
                    - current_bonus
                )

                score += (
                    2.0
                    + 1.5 / remaining
                )

        # Future engine value:
        # the bonus color matters more when cards we may want
        # actually require that color.
        if state is not None:

            relevance = self._bonus_color_relevance(
                state,
                player,
                bonus_color,
            )

            score += (
                relevance * 0.75
            )

        return score

    # ============================================================
    # OPPONENT MODEL
    # ============================================================

    def _opponent_card_threat(self, state, card):

        opponent = self._get_opponent(
            state
        )

        if opponent is None:
            return 0.0

        distance = self._distance_to_card(
            opponent,
            card,
        )

        # Ignore speculative targets.
        if distance > self.CLOSE_DISTANCE:
            return 0.0

        opponent_points = getattr(
            opponent,
            "points",
            0,
        )

        threat = 0.0

        threat += (
            card.points * 8.0
        )

        threat += 2.0

        threat += (
            self._bonus_color_relevance(
                state,
                opponent,
                card.bonus_color,
            )
            * 0.75
        )

        threat += self._noble_progress_value(
            opponent,
            card.bonus_color,
            state.nobles,
        )

        if distance == self.IMMEDIATE_DISTANCE:

            threat *= 1.50

        elif distance == self.NEAR_DISTANCE:

            threat *= 0.80

        elif distance == self.CLOSE_DISTANCE:

            threat *= 0.35

        # Immediate endgame threats should dominate.
        if opponent_points + card.points >= 15:

            threat += 100.0

        elif opponent_points >= 10:

            threat += (
                card.points * 3.0
            )

        return threat

    def _opponent_color_need(
        self,
        state,
        opponent,
    ):

        need = {
            color: 0.0
            for color in COLOR_ORDER
        }

        # Visible cards reveal plausible near-term plans.
        for card in self._visible_cards(
            state
        ):

            distance = self._distance_to_card(
                opponent,
                card,
            )

            if distance > 2:
                continue

            card_weight = (
                1.0
                + card.points * 0.75
            )

            if distance == 0:
                distance_weight = 2.0

            elif distance == 1:
                distance_weight = 1.0

            else:
                distance_weight = 0.4

            for color in COLOR_ORDER:

                shortage = max(
                    0,
                    card.cost.get(
                        color,
                        0,
                    )
                    - opponent.bonuses[color]
                    - opponent.gems[color],
                )

                if shortage <= 0:
                    continue

                need[color] += (
                    shortage
                    * card_weight
                    * distance_weight
                )

        # Reserved cards are an even stronger intent signal.
        for card in opponent.reserved_cards:

            if card is None:
                continue

            distance = self._distance_to_card(
                opponent,
                card,
            )

            if distance > 3:
                continue

            for color in COLOR_ORDER:

                shortage = max(
                    0,
                    card.cost.get(
                        color,
                        0,
                    )
                    - opponent.bonuses[color]
                    - opponent.gems[color],
                )

                need[color] += (
                    shortage
                    * (
                        1.0
                        + card.points * 0.5
                    )
                    / (distance + 1)
                )

        return need

    # ============================================================
    # ENGINE / NOBLE HELPERS
    # ============================================================

    def _bonus_color_relevance(
        self,
        state,
        player,
        color,
    ):

        relevance = 0.0

        # Visible cards.
        for card in self._visible_cards(
            state
        ):

            required = max(
                0,
                card.cost.get(
                    color,
                    0,
                )
                - player.bonuses[color],
            )

            if required <= 0:
                continue

            distance = self._distance_to_card(
                player,
                card,
            )

            if distance <= 4:

                relevance += (
                    required
                    * (
                        1.0
                        + card.points * 0.25
                    )
                    / (distance + 1)
                )

        # Reserved cards reveal our own concrete plan.
        for card in player.reserved_cards:

            if card is None:
                continue

            required = max(
                0,
                card.cost.get(
                    color,
                    0,
                )
                - player.bonuses[color],
            )

            if required <= 0:
                continue

            distance = self._distance_to_card(
                player,
                card,
            )

            relevance += (
                required
                * (
                    1.0
                    + card.points * 0.25
                )
                / (distance + 1)
            )

        return relevance

    def _noble_progress_value(
        self,
        player,
        bonus_color,
        nobles,
    ):

        value = 0.0

        for noble in nobles:

            if noble is None:
                continue

            requirement = noble.requirement[
                bonus_color
            ]

            current = player.bonuses[
                bonus_color
            ]

            if current >= requirement:
                continue

            remaining = (
                requirement - current
            )

            value += (
                2.0
                + 2.0 / remaining
            )

        return value

    # ============================================================
    # CORE HELPERS
    # ============================================================

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
                card.cost.get(
                    color,
                    0,
                )
                - player.bonuses[color]
            )

            missing += max(
                0,
                required - gems[color]
            )

        missing = max(
            0,
            missing
            - gems[GemColor.GOLD]
        )

        return missing

    def _all_relevant_cards(
        self,
        state,
        player,
    ):

        cards = self._visible_cards(
            state
        )

        cards.extend(
            card
            for card in player.reserved_cards
            if card is not None
        )

        return cards

    def _visible_cards(self, state):

        cards = []

        for tier in (1, 2, 3):

            cards.extend(
                card
                for card in state.visible_cards[tier]
                if card is not None
            )

        return cards

    def _get_opponent(self, state):

        if len(state.players) != 2:
            return None

        opponent_index = (
            1 - state.current_player
        )

        return state.players[
            opponent_index
        ]

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
