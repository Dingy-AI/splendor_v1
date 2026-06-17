import numpy as np
import gymnasium as gym
from gymnasium import spaces
from env.core.constants import ACTION_SPACE_SIZE
from env.state.base import GameState
from env.data.data import BASE_TIER_1, BASE_TIER_2, BASE_TIER_3, NOBLES
from env.core.actions import Action
from env.core.player import Player
from env.core.enums import GemColor, NodeType, ActionType
from env.core.card import Card

from observation.encoder import ObservationEncoder

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
        return None 
    
    
    def reset(self) -> GameState:
        #TODO
        self.state = self._build_initial_state()


        obs = self.observation_encoder.encoder(self.state)
        info = self._get_info(self.state)
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
            game_over = False
        )
        return state 

    def _init_decks(self):
        return {

            0: deepcopy(BASE_TIER_1),
            1: deepcopy(BASE_TIER_2),
            2: deepcopy(BASE_TIER_3)
        }

    def _deal_visible_cards(self, decks):
        visible = {}

        for tier, deck in decks.items():
            random.shuffle(deck)
            visible[tier] = [deck.pop() for _ in range(4)]
        return visible, decks

    def _init_players(self):
        return [
            Player(id=i) for i in range(self.num_players)
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

    def step(self, action=None):
        #TODO 
        obs = None
        reward = None 
        terminated = None
        truncated = None
        info = None 
        return obs, reward, terminated, truncated, info

    def _legal_actions(self, state:GameState)  -> list[Action]:
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
    
    def _legal_buy_visible(self, state:GameState) -> list[Action]:
        actions = []

        player = state.players[state.current_player]

        for tier, cards in state.visible_cards.items():
            for slot, card in enumerate(cards):

                if self._can_afford(player, card):
                    actions.append(
                        Action(
                            action_type=ActionType.BUY_VISIBLE,
                            tier=tier,
                            slot=slot,
                        )
                    )
        return actions

    def _legal_buy_reserved(self, state: GameState) -> list[Action]:
        actions = []

        player = state.players[state.current_player]

        for reserved_index, card in enumerate(player.reserved_cards):

            if self._can_afford(player, card):
                actions.append(
                    Action(
                        action_type=ActionType.BUY_RESERVED,
                        reserved_index=reserved_index,
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


    def _can_afford(self, player: Player, card: Card) -> bool:
        gold_needed = 0

        for color, cost in card.cost.items():

            # Apply bonus discount
            discounted_cost = max(0, cost - player.bonuses.get(color, 0))

            # How many colored gems do we actually have?
            available = player.gems.get(color, 0)

            # Missing gems must be covered by gold
            gold_needed += max(0, discounted_cost - available)

        return gold_needed <= player.gems.get(GemColor.GOLD, 0)
    

    # def get_reward(player_id = None):
    #     #TOOD
    #     return None 

    # def clone(self):
    #     #TODO
    #     return None 

    # def current_player(self):
    #     #TODO
    #     return None

    # def is_terminal(self):
    #     #TODO
    #     return None
    
    # def render(self):
    #     #TODO
    #     return None

    # def get_winner(self):
    #     #TODO
    #     return None
    
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