import numpy as np
import gymnasium as gym
from gymnasium import spaces
from splendor_v1.env.core.constants import MAX_GEMS, VICTORY_REQUIREMENT, COLOR_ORDER
from splendor_v1.env.core.action_constants import GEM_ACTION_TO_ID, DISCARD_COLOR_TO_ID, TAKE_GEMS_START, RESERVE_START, RESERVE_DECK_START, BUY_START, BUY_RESERVED_START, DISCARD_START, NOBLE_START, GEM_ACTIONS, ACTION_END, T1_PAYMENT_COUNT, T2_PAYMENT_COUNT, T3_PAYMENT_COUNT, BUY_T1_START, BUY_T2_START, BUY_T3_START, BUY_END, DISCARD_COLORS

from splendor_v1.env.state.base import GameState
from splendor_v1.env.data.data import BASE_TIER_1, BASE_TIER_2, BASE_TIER_3, NOBLES
from splendor_v1.env.core.actions import Action
from splendor_v1.env.core.player import Player
from splendor_v1.env.core.enums import GemColor, NodeType, ActionType, CardType
from splendor_v1.env.core.card import Card
from splendor_v1.env.core.noble import Noble

from splendor_v1.env.core.const_payment_lookup_table import COST_TYPE_PAYMENTS

from splendor_v1.env.core.cost_lookup_table_v3 import T1_PAYMENT_LOOKUP, T2_PAYMENT_LOOKUP, T3_PAYMENT_LOOKUP
from splendor_v1.env.core.cost_lookup_table_v2 import PAYMENT_TABLE


from splendor_v1.env.core.action_constants import ACTION_SPACE_SIZE

from splendor_v1.env.observation.encoder import ObservationEncoder
from itertools import combinations
import random
from copy import deepcopy

class SplendorEnv(gym.Env):
    def __init__ (self, num_players: int = 2, seed = 420):
        #TODO
        self.observation_encoder = ObservationEncoder()
        #I need to define the action space to allow for more expansions in the future
        #Pick Gems (6 gem types) -> Red, Blue, Green, Black, White
        #Pick 3 (10) - Red/Blue/Green, Red/Blue/Black, Red/Blue/White
        #             Red/Green/Black, Red/Green/White, Red/Black/White, 
        #             Blue/Green/Black, Blue/Green/White, Blue/Black/White, 
        #             Green/Black/White
        #Pick 2 different Color  Red/Blue, Red/Green, Red/Black, Red/White,
        #                        Blue/Green, Blue/Black, Blue/White
        #                        Green/Black, Green/White, Black/White
        #Pick 2 same color (5) - Red, Blue, Green, Black, White

        #Buy a card
        #Buy 1 of 12 (12) (15)

        # Reserve a card

        # Reserve 1 of 15 (15)
        # 0-8 -> Pick 3
        # 9-13 -> Pick 2
        # 14-29 -> Buy a Card
        # 30-45 -> Reserve a Card
        # 46-48 -> Pick Noble
        # 49-61 -> Discard 13 #this will be a loop where we will discard 1 and recheck to discard another
        # 62+ -> Expansion
        #Total actions 9 + 5 + 15 + 15 = 61 Actions
        self.num_players = num_players
        self.game_state = None
        self.seed = seed
        self.state = None
        return None 
    
    
    def reset(self) -> GameState:
        #TODO
        self.state = self._build_initial_state()


        obs = self.observation_encoder.encoder(self.state)
        info = self._get_info()
        return obs, info
    
    def _build_initial_state(self):
        decks = self._init_decks()
        visible_cards, decks = self._deal_visible_cards(decks)
        players = self._init_players()
        bank = self._init_bank()
        nobles = self._init_nobles()


        state = GameState(
            node_type=NodeType.MAIN_DECISION,
            players=players,
            bank=bank,
            nobles=nobles,
            visible_cards=visible_cards,
            decks=decks,
            current_player = 0,
            turn_number=0,
            game_over = False,
            winners = []
        )
        return state 

    def _init_decks(self):
        return {

            1: deepcopy(BASE_TIER_1),
            2: deepcopy(BASE_TIER_2),
            3: deepcopy(BASE_TIER_3)
        }

    def _deal_visible_cards(self, decks):
        visible = {}

        for tier, deck in decks.items():
            random.shuffle(deck)
            visible[tier] = [deck.pop() for _ in range(4)]
        return visible, decks

    def _init_players(self):
        return [
            Player(i, {
                GemColor.WHITE: 0,
                GemColor.BLUE: 0,
                GemColor.GREEN: 0,
                GemColor.RED: 0,
                GemColor.BLACK: 0,
                GemColor.GOLD: 0,
            }, {
                GemColor.WHITE: 0,
                GemColor.BLUE: 0,
                GemColor.GREEN: 0,
                GemColor.RED: 0,
                GemColor.BLACK: 0,
            }, [], [], []) for i in range(self.num_players)
        ]

    def _init_bank(self):
        base = 7
        if self.num_players == 2:
            base = 4 
        elif self.num_players == 3:
            base = 5 

        bank = {color: base for color in GemColor}
      


        bank[GemColor.GOLD] = 5

        return bank 

    def _init_nobles(self):
        nobles = deepcopy(NOBLES)
        random.shuffle(nobles)
        return nobles[:self.num_players+1]

    def _get_info(self): # work on this later for debugging purpose 
        return None 

    def _legal_actions(self, state: GameState) -> list[Action]:

        if state.node_type == NodeType.MAIN_DECISION:
            return self._legal_main_actions(state)

        if state.node_type == NodeType.OVERFLOW_DISCARD:
            return self._legal_discard_actions(state)

        if state.node_type == NodeType.NOBLE_CLAIM:
            return self._legal_noble_actions(state)

        raise ValueError(f"Unknown node type: {state.node_type}")

    def _legal_main_actions(self, state:GameState)  -> list[Action]:
        #TODO
        actions = []
        #I actually cant write the mask yet.
        #write now I just want a list of legal actions

        # BUY_VISIBLE
        # BUY_RESERVED
        # RESERVE_VISIBLE
        # RESERVE_TOP_DECK
        # TAKE_NOBLE
        # TAKE_GEMS
        # DISCARD_GEMS


        actions.extend(self._legal_buy_visible(state))
        actions.extend(self._legal_buy_reserved(state))
        actions.extend(self._legal_reserve_visible(state))
        actions.extend(self._legal_reserve_top_deck(state))
        actions.extend(self._legal_take_gems(state))        
        return actions
    
    #TODO right now it just says that it is a legal 'buy'
    # we need it to calculate the different payment options
    # def _legal_buy_visible(self, state:GameState) -> list[Action]:
    #     actions = []

    #     player = state.players[state.current_player]

    #     for tier, cards in state.visible_cards.items():
    #         for slot, card in enumerate(cards):
    #             if self._can_afford(player, card):
    #                 actions.append(
    #                     Action(
    #                         action_type=ActionType.BUY_VISIBLE,
    #                         tier=tier,
    #                         slot=slot,
    #                     )
    #                 )
    #     return actions
    def _legal_buy_visible(self, state: GameState) -> list[Action]:

        actions = []

        player = state.players[state.current_player]

        for tier, cards in state.visible_cards.items():

            payment_lookup = self._get_tier_payment_lookup(tier)

            for slot, card in enumerate(cards):

                color_mapping = self.get_color_mapping(card)

                for payment_id, canonical_payment in enumerate(payment_lookup):

                    actual_payment = self.map_payment_to_card(
                        canonical_payment,
                        color_mapping
                    )

                    if self._can_pay(
                        player,
                        card,
                        actual_payment
                    ):
                        actions.append(
                            Action(
                                action_type=ActionType.BUY_VISIBLE,
                                tier=tier,
                                slot=slot,
                                payment_id=payment_id,
                                gold_payment=actual_payment,
                            )
                        )

        return actions

    def _get_tier_payment_lookup(self, tier: int):

        if tier == 1:
            return T1_PAYMENT_LOOKUP

        if tier == 2:
            return T2_PAYMENT_LOOKUP

        if tier == 3:
            return T3_PAYMENT_LOOKUP

        raise ValueError(f"Unknown tier {tier}")

    def get_card_type(self, card) -> CardType:
        """
        Returns the canonical CardType for a card.

        Canonicalization:
            1. Sort costs descending.
            2. Break ties using
                WHITE > BLUE > GREEN > RED > BLACK.
            3. Remove all zero costs.
            4. Convert to enum name.
        """

        costs = [
            card.cost.get(color,0)
            for color in COLOR_ORDER
        ]

        # Sort descending while preserving COLOR_ORDER for ties.
        sorted_costs = sorted(costs, reverse=True)

        # Remove zeros.
        shape = "".join(
            str(cost)
            for cost in sorted_costs
            if cost > 0
        )

        enum_name = f"T{card.tier}_{shape}"

        return CardType[enum_name]



    def get_color_mapping(self, card):
        """
        Returns mapping from canonical color positions
        to actual card colors.

        Example:
            canonical WHITE -> actual GREEN
        """

        # Pair each color with its cost
        color_costs = [
            (color, card.cost.get(color, 0))
            for color in COLOR_ORDER
        ]

        # Sort:
        # 1. highest cost first
        # 2. COLOR_ORDER breaks ties
        sorted_colors = sorted(
            color_costs,
            key=lambda x: (-x[1], COLOR_ORDER.index(x[0]))
        )

        # Only colors that matter
        actual_colors = [
            color
            for color, cost in sorted_colors
            if cost > 0
        ]

        # Add zero colors at the end
        actual_colors += [
            color
            for color, cost in sorted_colors
            if cost == 0
        ]

        return actual_colors


    def map_payment_to_card(self,
        canonical_payment,
        color_mapping
    ):

        payment = {
            color:0
            for color in COLOR_ORDER
        }

        for idx, gold_amount in enumerate(canonical_payment):

            if gold_amount > 0:

                actual_color = color_mapping[idx]

                payment[actual_color] = gold_amount

        return tuple(
            payment[color]
            for color in COLOR_ORDER
        )
    def _get_payment_options(self, player, card):

        #This converts the GemColor.WHITE:1, etc... into 
        #(1,0,0,0,0) etc...
        # we can then enter this into a lookup table for possible payments
        # then for loop the possible payments and create the possible buy actions
        cost_type = self._get_cost_type(card)

        #TODO need to create the cost type payments look up table 
        # this lookup table will be based on the cards in data.py
        possible_payments = COST_TYPE_PAYMENTS[cost_type]

        legal_payments = []

        for payment in possible_payments:
            if self._can_pay(player, card, payment):
                legal_payments.append(payment)

        return legal_payments


    def _get_cost_type(self, card: Card):

        cost = tuple(
            card.cost[color]
            for color in [
                GemColor.WHITE,
                GemColor.BLUE,
                GemColor.GREEN,
                GemColor.RED,
                GemColor.BLACK,
            ]
        )

        return COST_TO_ID[cost]

    def _legal_buy_reserved(self, state: GameState) -> list[Action]:

        actions = []

        player = state.players[state.current_player]

        for reserved_index, card in enumerate(player.reserved_cards):

            card_type = self.get_card_type(card)

            color_mapping = self.get_color_mapping(card)

            # Only generate payments that apply to this card type
            # payments = PAYMENT_TABLE[card_type]

            # payment_lookup = self._get_tier_payment_lookup(card.tier)
            payment_lookup = T3_PAYMENT_LOOKUP
            for payment_id, canonical_payment in enumerate(payment_lookup):

                actual_payment = self.map_payment_to_card(
                    canonical_payment,
                    color_mapping
                )

                if self._can_pay(
                    player,
                    card,
                    actual_payment
                ):
                    actions.append(
                        Action(
                            action_type=ActionType.BUY_RESERVED,
                            reserved_index=reserved_index,
                            payment_id=payment_id,
                            gold_payment=actual_payment,
                        )
                    )

        return actions

    def _legal_reserve_visible(self, state: GameState) -> list[Action]:
        actions = []

        player = state.players[state.current_player]

        # Reserve limit reached
        if len(player.reserved_cards) >= 3:
            return actions

        for tier, cards in state.visible_cards.items():
            for slot, card in enumerate(cards):

                # Skip empty slots if your implementation uses None
                if card is None:
                    continue

                actions.append(
                    Action(
                        action_type=ActionType.RESERVE_VISIBLE,
                        tier=tier,
                        slot=slot,
                    )
                )

        return actions

    def _legal_reserve_top_deck(self, state: GameState) -> list[Action]:
        actions = []

        player = state.players[state.current_player]

        # Cannot reserve more than 3 cards
        if len(player.reserved_cards) >= 3:
            return actions

        for tier, deck in state.decks.items():

            # Cannot reserve from an empty deck
            if len(deck) == 0:
                continue

            actions.append(
                Action(
                    action_type=ActionType.RESERVE_TOP_DECK,
                    tier=tier,
                )
            )

        return actions
    
    def _legal_take_gems(self, state: GameState) -> list[Action]:
        actions = []

        bank = state.bank

        available_colors = [
            color
            for color in GemColor
            if color != GemColor.GOLD and bank[color] > 0
        ]

        # Take 1 gem
        for color in available_colors:
            actions.append(
                Action(
                    action_type=ActionType.TAKE_GEMS,
                    gem_colors=(color,)
                )
            )

        # Take 2 different gems
        for combo in combinations(available_colors, 2):
            actions.append(
                Action(
                    action_type=ActionType.TAKE_GEMS,
                    gem_colors=combo
                )
            )

        # Take 3 different gems
        for combo in combinations(available_colors, 3):
            actions.append(
                Action(
                    action_type=ActionType.TAKE_GEMS,
                    gem_colors=combo
                )
            )

        # Take 2 of same color
        for color in available_colors:
            if bank[color] >= 4:
                actions.append(
                    Action(
                        action_type=ActionType.TAKE_GEMS,
                        gem_colors=(color, color)
                    )
                )

        return actions

    def _legal_discard_actions(self, state: GameState) -> list[Action]:
        actions = []

        player = state.players[state.current_player]

        total_gems = sum(player.gems.values())
        excess = total_gems - MAX_GEMS  # usually 10

        if excess <= 0:
            return actions

        # One-gem discard actions (repeat until resolved by multiple turns)
        for color, count in player.gems.items():

            if count <= 0:
                continue

            actions.append(
                Action(
                    action_type=ActionType.DISCARD_GEMS,
                    gem_colors=(color,),
                )
            )

        return actions

    def _legal_noble_actions(self, state: GameState) -> list[Action]:
        actions = []

        player = state.players[state.current_player]

        for i, noble in enumerate(state.nobles):

            # Skip missing / already-taken nobles if applicable
            if noble is None:
                continue

            # Check if player qualifies
            if self._qualifies_for_noble(player, noble):
                actions.append(
                    Action(
                        action_type=ActionType.TAKE_NOBLE,
                        noble_index=i
                    )
                )

        return actions
        
    def _qualifies_for_noble(self, player: Player, noble: Noble) -> bool:
        for color, required in noble.requirement.items():
            if player.bonuses.get(color, 0) < required:
                return False
        return True


    def _can_pay(
        self,
        player: Player,
        card: Card,
        gold_payment: tuple[int, ...]
    ) -> bool:

        # Track how much gold is being used
        gold_used = sum(gold_payment)

        # Player does not have enough gold
        # print("gemcheck", player.gems)
        # print("checker for gold", player.gems[GemColor.GOLD])
        if player.gems[GemColor.GOLD] < gold_used:
            return False

        # Check each colored gem
        for color, gold_substitute in zip(
            [
                GemColor.WHITE,
                GemColor.BLUE,
                GemColor.GREEN,
                GemColor.RED,
                GemColor.BLACK,
            ],
            gold_payment,
        ):

            # Original cost
            required = card.cost[color]

            # Gold covers part of this color
            remaining = required - gold_substitute

            # If gold covers more than needed
            if remaining < 0:
                return False

            # Need enough colored gems
            if player.gems[color] < remaining:
                return False

        return True


    # def _can_afford(self, player: Player, card: Card) -> bool:
    #     gold_needed = 0
    #     for color, cost in card.cost.items():
    #         # Apply bonus discount
            
    #         discounted_cost = max(0, cost - player.bonuses.get(color, 0))
    #         # How many colored gems do we actually have?

    #         # there is an issue with card color and gem color
    #         # I will make a quick fix but will need to re-factor in the future


    #         available = player.gems.get(color, 0)
    #         # Missing gems must be covered by gold
    #         gold_needed += max(0, discounted_cost - available)

    #     return gold_needed <= player.gems.get(GemColor.GOLD, 0)
    

    def step(self, action: Action):
        state = self.state

        actor = state.current_player   # FREEZE actor here
        prev_points = state.players[actor].points


        
        # 1. Apply action
        self._apply_action(state, action)

        # 2. Resolve forced transitions (overflow, nobles, etc.)
        self._resolve_transitions(state)


        self._maybe_advance_player(state)


        # 3. Compute observation
        obs = self.observation_encoder.encoder(self.state)



        # 4. Compute reward
        reward = self._compute_reward(state, actor, prev_points)

        # 4.5 Check end triggered

        # 5. Check termination
        terminated = self._check_terminated(state, actor)   # game ended naturally

        # 6. Check truncation
        truncated = False #ptional time limit




        # 7. Info dictionary (debug / logging)
        info = {
            "node_type": state.node_type,
            "current_player": state.current_player,
            "legal_actions_count": len(self._legal_actions(state)),
            "turn_number": state.turn_number,
        }

        if terminated:
            info["winners"] = state.winners
            info["final_scores"] = [
                p.points for p in state.players
                ]

        return obs, reward, terminated, truncated, info

    def _compute_reward(self, state, actor, prev_points):

        curr_points = state.players[actor].points

        reward = curr_points - prev_points
        return reward

    def _check_terminated(self, state, actor):

        if not state.end_triggered:
            return False
        # end when we return to start player of final round
        if actor == len(state.players)-1:
            state.winners = self._compute_winners(state)
            return True

        return False

    def _compute_winners(self, state: GameState) -> list[int]:

        best_points = max(p.points for p in state.players)

        candidates = [
            (i, p) for i, p in enumerate(state.players)
            if p.points == best_points
        ]

        # Splendor rule: FEWEST cards wins
        min_cards = min(len(p.purchased_cards) for _, p in candidates)

        winnerss = [
            i for i, p in candidates
            if len(p.purchased_cards) == min_cards
        ]

        return winnerss

    def _maybe_advance_player(self, state: GameState):

        # DO NOT switch players during forced nodes
        if state.node_type != NodeType.MAIN_DECISION:
            return

        state.current_player = (state.current_player + 1) % self.num_players
        state.turn_number += 1

    def _resolve_transitions(self, state: GameState):

        # IMPORTANT: only called from MAIN_DECISION

        # 1. overflow has highest priority
        if self._check_overflow(state):
            state.node_type = NodeType.OVERFLOW_DISCARD
            return

        # 2. nobles next priority
        if self._check_nobles(state):
            state.node_type = NodeType.NOBLE_CLAIM
            return

        # 3. otherwise continue normal gameplay
        state.node_type = NodeType.MAIN_DECISION
        return

    def _check_overflow(self, state):

        player = state.players[state.current_player]

        return sum(player.gems.values()) > MAX_GEMS

    def _check_nobles(self, state):

        player = state.players[state.current_player]

        for noble in state.nobles:
            if noble and self._qualifies_for_noble(player, noble):
                return True

        return False

    def _apply_action(self, state: GameState, action: Action):

        if state.node_type == NodeType.MAIN_DECISION:
            self._apply_main_action(state, action)

        elif state.node_type == NodeType.OVERFLOW_DISCARD:
            self._apply_discard_action(state, action)

        elif state.node_type == NodeType.NOBLE_CLAIM:
            self._apply_noble_action(state, action)

        else:
            raise ValueError(f"Unknown node type: {state.node_type}")


    def _apply_main_action(self, state: GameState, action: Action):

        if action.action_type == ActionType.BUY_VISIBLE:
            self._buy_visible(state, action)

        elif action.action_type == ActionType.BUY_RESERVED:
            self._buy_reserved(state, action)

        elif action.action_type == ActionType.RESERVE_VISIBLE:
            self._reserve_visible(state, action)

        elif action.action_type == ActionType.RESERVE_TOP_DECK:
            self._reserve_top_deck(state, action)

        elif action.action_type == ActionType.TAKE_GEMS:
            self._take_gems(state, action)

        else:
            raise ValueError(f"Invalid main action: {action}")


    def _buy_visible(self, state: GameState, action: Action):

        player: Player = state.players[state.current_player]

        card: Card = state.visible_cards[action.tier][action.slot]

        # Get canonical payment from tier lookup
        payment_lookup = self._get_tier_payment_lookup(card.tier)

        canonical_payment = payment_lookup[action.payment_id]

        # Convert canonical payment to actual card colors
        color_mapping = self.get_color_mapping(card)

        gold_payment = self.map_payment_to_card(
            canonical_payment,
            color_mapping
        )

        # Pay for the card
        self._pay_for_card(
            state,
            player,
            card,
            gold_payment
        )

        # Gain the card
        player.purchased_cards.append(card)

        player.points += card.points

        player.bonuses[card.bonus_color] += 1

        # Replace bought card
        state.visible_cards[action.tier][action.slot] = self._draw_card(
            state,
            action.tier
        )

        # Trigger end game
        if player.points >= VICTORY_REQUIREMENT:
            state.end_triggered = True

    def _buy_reserved(self, state: GameState, action: Action):

        player: Player = state.players[state.current_player]

        card: Card = player.reserved_cards[action.reserved_index]

        # Get canonical payment from tier lookup
        # payment_lookup = self._get_tier_payment_lookup(card.tier)

        canonical_payment = T3_PAYMENT_LOOKUP[action.payment_id]

        # Convert canonical payment to actual card colors
        color_mapping = self.get_color_mapping(card)

        gold_payment = self.map_payment_to_card(
            canonical_payment,
            color_mapping
        )

        # Pay for the card
        self._pay_for_card(
            state,
            player,
            card,
            gold_payment
        )

        player.reserved_cards.pop(action.reserved_index)

        player.purchased_cards.append(card)

        player.points += card.points

        player.bonuses[card.bonus_color] += 1

        if player.points >= VICTORY_REQUIREMENT:
            state.end_triggered = True

    def _draw_card(self, state, tier):
        deck = state.decks[tier]

        if len(deck) == 0:
            return None

        return deck.pop()


    def _reserve_visible(self, state: GameState, action: Action):

        player = state.players[state.current_player]

        card = state.visible_cards[action.tier][action.slot]

        player.reserved_cards.append(card)

        # refill slot
        state.visible_cards[action.tier][action.slot] = self._draw_card(state, action.tier)

        # gold bonus (optional)
        if state.bank[GemColor.GOLD] > 0:
            player.gems[GemColor.GOLD] += 1
            state.bank[GemColor.GOLD] -= 1


    def _reserve_top_deck(self, state: GameState, action: Action):

        player = state.players[state.current_player]
        card = state.decks[action.tier].pop()
        player.reserved_cards.append(card)

        # gold bonus (optional)
        if state.bank[GemColor.GOLD] > 0:
            player.gems[GemColor.GOLD] += 1
            state.bank[GemColor.GOLD] -= 1
     

    def _take_gems(self, state: GameState, action: Action):

        player = state.players[state.current_player]

        for color in action.gem_colors:
            player.gems[color] += 1
            state.bank[color] -= 1

    def _apply_discard_action(self, state: GameState, action: Action):

        player = state.players[state.current_player]

        for color in action.gem_colors:
            player.gems[color] -= 1

        # AFTER applying discard → check if overflow is resolved
        if sum(player.gems.values()) <= MAX_GEMS:
            state.node_type = NodeType.MAIN_DECISION


    def _apply_noble_action(self, state: GameState, action: Action):

        player = state.players[state.current_player]
        noble = state.nobles[action.noble_index]

        player.nobles.append(noble)
        player.points += noble.points # standard Splendor noble value

        state.nobles[action.noble_index] = None

        state.node_type = NodeType.MAIN_DECISION
        if player.points >= VICTORY_REQUIREMENT:
            state.end_triggered = True        

    def _pay_for_card(
        self,
        state: GameState,
        player: Player,
        card: Card,
        gold_payment: tuple[int,...]
    ):

        gold_used = sum(gold_payment)

        for color in COLOR_ORDER:

            required = card.cost[color]

            gold_substitute = gold_payment[color.value]

            normal_required = required - gold_substitute

            if normal_required > 0:
                player.gems[color] -= normal_required
                state.bank[color] += normal_required

        # Remove gold
        player.gems[GemColor.GOLD] -= gold_used
        state.bank[GemColor.GOLD] += gold_used

    def clone(self):
        return deepcopy(self) 

    def action_to_id(self, action: Action) -> int:

        if action.action_type == ActionType.TAKE_GEMS:

            gem_colors = tuple(sorted(action.gem_colors, key=lambda x: x.value))
            return TAKE_GEMS_START + GEM_ACTION_TO_ID[gem_colors]
        
        elif action.action_type == ActionType.RESERVE_VISIBLE:
            return RESERVE_START + ((action.tier-1) * 4) + action.slot

        elif action.action_type == ActionType.RESERVE_TOP_DECK:
            return RESERVE_DECK_START + (action.tier-1)

        elif action.action_type == ActionType.BUY_VISIBLE:

            payment_counts = {
                1: T1_PAYMENT_COUNT,
                2: T2_PAYMENT_COUNT,
                3: T3_PAYMENT_COUNT,
            }

            tier_starts = {
                1: BUY_T1_START,
                2: BUY_T2_START,
                3: BUY_T3_START,
            }

            payment_count = payment_counts[action.tier]

            return (
                tier_starts[action.tier]
                + action.slot * payment_count
                + action.payment_id
            )


        elif action.action_type == ActionType.BUY_RESERVED:
            return (
                BUY_RESERVED_START
                + action.reserved_index * T3_PAYMENT_COUNT
                + action.payment_id
            )
        elif action.action_type == ActionType.DISCARD_GEMS:
            gem_colors = tuple(sorted(action.gem_colors, key=lambda x: x.value))
            return DISCARD_START + DISCARD_COLOR_TO_ID[gem_colors[0]]

        elif action.action_type == ActionType.TAKE_NOBLE:
            return NOBLE_START + action.noble_index

        raise ValueError(f"Unknown action: {action}")

    #CONTINUE HERE TOMORROW -> Need to do ID to action 
    # Also need to do the masking 

    def id_to_action(self, action_id: int) -> Action:

        if TAKE_GEMS_START <= action_id < RESERVE_START:
            return self._id_to_take_gems(action_id)

        elif RESERVE_START <= action_id < BUY_START:
            return self._id_to_reserve(action_id)

        elif BUY_START <= action_id < DISCARD_START:
            return self._id_to_buy(action_id)

        elif DISCARD_START <= action_id < NOBLE_START:
            return self._id_to_discard(action_id)

        elif NOBLE_START <= action_id < ACTION_END:
            return self._id_to_noble(action_id)

        raise ValueError(action_id)
        

    def _id_to_take_gems(self, action_id: int) -> Action:

        gem_colors = GEM_ACTIONS[action_id]

        return Action(
            action_type=ActionType.TAKE_GEMS,
            gem_colors=gem_colors,
        )
        
    def _id_to_reserve(self, action_id: int) -> Action:

        # ---------- Reserve Visible ----------
        if RESERVE_START <= action_id < RESERVE_DECK_START:

            offset = action_id - RESERVE_START

            tier = (offset // 4) + 1
            slot = offset % 4

            return Action(
                action_type=ActionType.RESERVE_VISIBLE,
                tier=tier,
                slot=slot,
            )

        # ---------- Reserve Top Deck ----------
        if RESERVE_DECK_START <= action_id < BUY_START:

            offset = action_id - RESERVE_DECK_START

            tier = offset + 1

            return Action(
                action_type=ActionType.RESERVE_TOP_DECK,
                tier=tier,
            )

        raise ValueError(f"Invalid reserve action id: {action_id}")


    def _id_to_buy(self, action_id: int) -> Action:

        # ---------- Visible Tier 1 ----------
        if BUY_T1_START <= action_id < BUY_T2_START:

            offset = action_id - BUY_T1_START

            slot = offset // T1_PAYMENT_COUNT
            payment_id = offset % T1_PAYMENT_COUNT

            return Action(
                action_type=ActionType.BUY_VISIBLE,
                tier=1,
                slot=slot,
                payment_id=payment_id,
            )

        # ---------- Visible Tier 2 ----------
        if BUY_T2_START <= action_id < BUY_T3_START:

            offset = action_id - BUY_T2_START

            slot = offset // T2_PAYMENT_COUNT
            payment_id = offset % T2_PAYMENT_COUNT

            return Action(
                action_type=ActionType.BUY_VISIBLE,
                tier=2,
                slot=slot,
                payment_id=payment_id,
            )

        # ---------- Visible Tier 3 ----------
        if BUY_T3_START <= action_id < BUY_RESERVED_START:

            offset = action_id - BUY_T3_START

            slot = offset // T3_PAYMENT_COUNT
            payment_id = offset % T3_PAYMENT_COUNT

            return Action(
                action_type=ActionType.BUY_VISIBLE,
                tier=3,
                slot=slot,
                payment_id=payment_id,
            )

        # ---------- Reserved (All Tiers) ----------
        offset = action_id - BUY_RESERVED_START

        reserved_index = offset // T3_PAYMENT_COUNT
        payment_id = offset % T3_PAYMENT_COUNT

        return Action(
            action_type=ActionType.BUY_RESERVED,
            reserved_index=reserved_index,
            payment_id=payment_id,
        )

    def _id_to_discard(self, action_id: int) -> Action:

        discard_id = action_id - DISCARD_START
        return Action(
            action_type=ActionType.DISCARD_GEMS,
            gem_colors=(DISCARD_COLORS[discard_id],),
        )

    def _id_to_noble(self, action_id: int) -> Action:

        return Action(
            action_type=ActionType.TAKE_NOBLE,
            noble_index=action_id - NOBLE_START,
        )   
        
    def action_mask(self, state):

        mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.int8)

        for action in self._legal_actions(state):

            action_id = self.action_to_id(action)

            mask[action_id] = 1

        return mask
#TODO NEED TO WORK ON ACTION MASKING

#    ↓
# legal_actions(state)
#    ↓
# [Action, Action, Action]
#    ↓
# action_to_id()
#    ↓
# [17, 22, 31]
#    ↓
# get_action_mask()
#    ↓
# [0,0,1,0,1,...]
#    ↓
# Policy Network
#    ↓
# chosen action_id
#    ↓
# id_to_action()
#    ↓
# Action(...)
#    ↓
# env.step(action)

# ✅ legal_actions(state)
# ✅ step(action)
# ✅ helper functions (_can_afford, _take_gems, etc.)
# Later: action_to_id # can live in decoder/encoder 
# Later: id_to_action # can live in decoder / encoder 
# Later: get_action_mask
# Later: legal_action_ids