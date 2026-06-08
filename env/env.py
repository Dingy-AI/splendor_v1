import numpy as np
import gymnasium as gym
from gymnasium import spaces
from env.core.constants import ACTION_SPACE_SIZE
from env.state.base import GameState
from env.data.data import BASE_TIER_1, BASE_TIER_2, BASE_TIER_3, NOBLES

from env.core.player import Player
from env.core.enums import GemColor, NodeType

import random
from copy import deepcopy

class SplendorEnv(gym.Env):
    def __init__ (self, num_players: int = 2, seed = 420):
        #TODO
        
        #I need to define the action space to allow for more expansions in the future
        #Pick Gems (6 gem types) -> Red, Blue, Green, Black, White
        #Pick 3 (9) - Red/Blue/Green, Red/Blue/Black, Red/Blue/White
        #             Red/Green/Black, Red/Green/White, Red/Black/White, Blue/Green/Black, Blue/Black/White, Green/Black/White
        #Pick 2 (5) - Red, Blue, Green, Black, White

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


        obs = self._get_observation(self.state)
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

    def _get_observation(self):
        pass

    def _get_info(self):
        pass    

    def step(self, action=None):
        #TODO 
        obs = None
        reward = None 
        terminated = None
        truncated = None
        info = None 
        return obs, reward, terminated, truncated, info

    def legal_actions(self):
        #TODO
        #write the mask that starts everything as false
        #and then sets the valid values to true

        return None
    def can_afford(self, card) -> bool:
        gold_needed = 0

        for color, cost in card.cost.items():
            effective_cost = max(
                0,
                cost - self.bonuses[color]
            )

            available = self.gems[color]

            if available < effective_cost:
                gold_needed += effective_cost - available

        return gold_needed <= self.gems[GemColor.GOLD]
    

    def get_reward(player_id = None):
        #TOOD
        return None 

    def clone(self):
        #TODO
        return None 

    def current_player(self):
        #TODO
        return None

    def is_terminal(self):
        #TODO
        return None
    
    def render(self):
        #TODO
        return None

    def get_winner(self):
        #TODO
        return None
    
